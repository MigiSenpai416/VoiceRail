"""Train tab: build a voice model from raw recordings."""

from __future__ import annotations

import html
import json
import logging
import shutil
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from .. import jobs, paths
from ..assets import DEFAULT_PRETRAINED, PRETRAINED_SETS
from ..audio.io import AUDIO_EXTENSIONS
from ..audio.separation import (
    DENOISE_MODELS,
    DEREVERB_MODELS,
    LEAD_MODELS,
    VOCAL_MODELS,
)
from ..train.config import SAMPLE_RATES, TrainConfig
from ..train.dataset import CleaningOptions, SegmentOptions, load_report
from ..train.job import STAGES, job_file, write_job
from .common import friendly_errors, reset_button

log = logging.getLogger(__name__)

PRESETS = {
    "Songs (vocals with music)": dict(
        isolate=True, backing=True, dereverb=True, denoise=False
    ),
    "Clean voice (studio vocals, podcasts, audiobooks)": dict(
        isolate=False, backing=False, dereverb=False, denoise=False
    ),
    "Noisy or echoey voice (phone, video, streams)": dict(
        isolate=True, backing=False, dereverb=True, denoise=True
    ),
}
FULL_PIPELINE = ["prepare", "features", "train", "index", "publish"]
UPLOAD_TYPES = sorted(AUDIO_EXTENSIONS)
DEFAULT_TRAIN = TrainConfig()
DEFAULT_SEGMENT = SegmentOptions()


def _status_html(name: str | None) -> str:
    if not name:
        return (
            "<div class='vr-card'><h3>Nothing trained yet</h3>"
            "<div class='vr-muted'>Name your voice, add recordings and press "
            "<b>Start training</b>. Progress shows here and keeps running if "
            "you close this page.</div></div>"
        )
    st = jobs.status(name)
    if st.state == "idle":
        return (
            f"<div class='vr-card'><h3>{html.escape(name)}</h3>"
            "<div class='vr-muted'>Not started yet.</div></div>"
        )
    stages = st.stages
    count = max(st.stage_count, 1)
    overall = (
        1.0
        if st.state == "finished"
        else min(1.0, (st.stage_index + st.progress) / count)
    )
    chips = []
    for i, stage in enumerate(stages):
        cls = "done" if i < st.stage_index or st.state == "finished" else ""
        if i == st.stage_index and st.state == "running":
            cls = "current"
        chips.append(f"<span class='vr-step {cls}'>{STAGES.get(stage, stage)}</span>")
    elapsed = ""
    if st.started:
        end = time.time() if st.state == "running" else (st.updated or time.time())
        minutes = int((end - st.started) // 60)
        elapsed = (
            f" · {minutes // 60}h {minutes % 60:02d}m"
            if minutes >= 60
            else (f" · {minutes} min")
        )
    metrics = ""
    if st.last_metrics:
        m = st.last_metrics
        val = m.get("val_mel")
        metrics = (
            f"<div class='vr-muted'>epoch {m.get('epoch')} · step {m.get('step')}"
            + (f" · validation {val:.3f}" if val is not None else "")
            + (f" · best epoch {m['best_epoch']}" if m.get("best_epoch") else "")
            + "</div>"
        )
    error = ""
    if st.error and st.state in ("failed", "crashed"):
        error = (
            "<pre style='white-space:pre-wrap;margin-top:8px'>"
            f"{html.escape(st.error)}</pre>"
        )
    done = ""
    if st.state == "finished" and st.result and st.result.get("model_dir"):
        done = (
            "<div style='margin-top:8px'>✅ The voice is ready in the "
            "<b>Convert</b> tab.</div>"
        )
    return (
        f"<div class='vr-card'><h3>{html.escape(name)} · "
        f"<span class='vr-state {st.state}'>{st.state}</span>"
        f"<span class='vr-muted'>{elapsed}</span></h3>"
        f"<div>{html.escape(st.message)}</div>"
        f"<div class='vr-bar'><div style='width:{overall * 100:.1f}%'></div></div>"
        f"{metrics}<div class='vr-steps'>{''.join(chips)}</div>{error}{done}</div>"
    )


def _loss_frame(name: str | None) -> pd.DataFrame:
    rows = []
    for m in jobs.metrics(name) if name else []:
        if m.get("mel") is not None:
            rows.append(
                {"epoch": m["epoch"], "loss": m["mel"] / 45.0, "series": "train"}
            )
        if m.get("val_mel") is not None:
            rows.append(
                {"epoch": m["epoch"], "loss": m["val_mel"], "series": "validation"}
            )
    return pd.DataFrame(rows, columns=["epoch", "loss", "series"])


def _report_md(name: str | None) -> str:
    if not name:
        return ""
    report = load_report(jobs.experiment_dir(name))
    if report is None:
        return "*The dataset has not been prepared yet.*"
    lines = [
        f"**Clean voice:** {report.total_seconds / 60:.1f} min in "
        f"{report.total_segments} segments at {report.sample_rate // 1000} kHz"
    ]
    lines += [f"⚠️ {w}" for w in report.warnings]
    rows = ["| File | Length | Kept | Notes |", "|---|---|---|---|"]
    for f in report.files:
        notes = f.error or "; ".join(
            f.warnings + [f"dropped {n} {k}" for k, n in f.dropped.items()]
        )
        rows.append(
            f"| {html.escape(f.name)} | {f.seconds / 60:.1f} min | "
            f"{f.kept_seconds / 60:.1f} min | {html.escape(notes) or '—'} |"
        )
    return "\n\n".join(lines) + "\n\n" + "\n".join(rows)


def _samples(name: str | None) -> tuple[str | None, str | None, str | None]:
    """(latest epoch preview, training reference, a cleaned dataset clip)."""
    if not name:
        return None, None, None
    exp = jobs.experiment_dir(name)
    previews = sorted((exp / "samples").glob("epoch_*.wav"))
    reference = exp / "samples" / "reference.wav"
    clips = sorted((exp / "dataset" / "wavs").glob("*.wav"))
    clip = clips[len(clips) // 2] if clips else None
    return (
        str(previews[-1]) if previews else None,
        str(reference) if reference.is_file() else None,
        str(clip) if clip else None,
    )


def _source_summary(name: str | None) -> str:
    if not name:
        return ""
    src = jobs.experiment_dir(name) / "sources"
    files = [p for p in src.glob("*") if p.is_file()] if src.is_dir() else []
    if not files:
        return "*No recordings stored in this session yet.*"
    size = sum(p.stat().st_size for p in files) / 1024**2
    return f"{len(files)} recording(s) stored for this session ({size:.0f} MB)."


def build() -> dict:
    sessions = jobs.experiments()
    current = jobs.latest_experiment()
    with gr.Column(elem_classes="vr-page"):
        name = gr.Textbox(
            label="Voice name",
            placeholder="e.g. MyVoice",
            value=current or "",
            info="Also the name of the training session. To continue an earlier "
            "one, pick it under Options › Session.",
        )
        files = gr.File(
            label="Recordings (mp3, wav, flac, m4a, video, ...)",
            file_count="multiple",
            file_types=UPLOAD_TYPES,
            type="filepath",
        )
        preset = gr.Radio(
            list(PRESETS),
            value=next(iter(PRESETS)),
            label="What kind of recordings?",
            info="Picks the clean-up steps: songs get the music, backing vocals "
            "and reverb removed; clean voice skips all of that; noisy voice also "
            "gets denoised. Fine-tune under Options › Clean-up.",
        )

        with gr.Accordion("Options", open=False):
            with gr.Accordion("Session", open=False):
                with gr.Row():
                    session = gr.Dropdown(
                        sessions,
                        value=current,
                        label="Training session",
                        info="Pick an earlier session to monitor or resume it.",
                        allow_custom_value=False,
                        scale=4,
                    )
                    refresh_sessions = gr.Button("↻", scale=0, min_width=44)
                folder = gr.Textbox(
                    label="Folder of recordings on this computer",
                    placeholder="C:\\Users\\me\\voice-recordings",
                    info="Used in addition to uploaded files.",
                )
                sources_md = gr.Markdown(
                    _source_summary(current), elem_classes="vr-hint"
                )
                with gr.Row():
                    fresh = gr.Checkbox(
                        value=False,
                        label="Start over",
                        info="Discard this session's previous checkpoints.",
                    )
                    prepare_btn = gr.Button("Prepare dataset only", size="sm")
                    publish_btn = gr.Button("Save current model now", size="sm")
            with gr.Accordion("Clean-up", open=False):
                with gr.Row():
                    isolate = gr.Checkbox(value=True, label="Remove music / background")
                    vocal_model = gr.Dropdown(
                        [m.label for m in VOCAL_MODELS],
                        value=VOCAL_MODELS[0].label,
                        label="Vocal separation model",
                    )
                with gr.Row():
                    backing = gr.Checkbox(
                        value=True, label="Remove backing vocals / harmonies"
                    )
                    lead_model = gr.Dropdown(
                        [m.label for m in LEAD_MODELS],
                        value=LEAD_MODELS[0].label,
                        label="Lead vocal model",
                    )
                with gr.Row():
                    dereverb = gr.Checkbox(value=True, label="Remove reverb / echo")
                    dereverb_model = gr.Dropdown(
                        [m.label for m in DEREVERB_MODELS],
                        value=DEREVERB_MODELS[0].label,
                        label="De-reverb model",
                    )
                with gr.Row():
                    denoise = gr.Checkbox(value=False, label="Remove noise")
                    denoise_model = gr.Dropdown(
                        [m.label for m in DENOISE_MODELS],
                        value=DENOISE_MODELS[0].label,
                        label="Denoise model",
                    )
            with gr.Accordion("Training", open=False):
                with gr.Row():
                    sample_rate = gr.Radio(
                        [str(sr) for sr in SAMPLE_RATES],
                        value=str(DEFAULT_TRAIN.sample_rate),
                        label="Sample rate",
                        info="40k is the best-supported choice; 48k for hi-fi "
                        "singing (more VRAM), 32k for very small GPUs.",
                    )
                    pretrained = gr.Dropdown(
                        list(PRETRAINED_SETS),
                        value=DEFAULT_PRETRAINED,
                        label="Base (pretrained) model",
                        info="The voice is fine-tuned from this. TITAN is cleaner "
                        "but a larger download.",
                    )
                with gr.Row():
                    epochs = gr.Slider(
                        0,
                        1000,
                        value=DEFAULT_TRAIN.epochs,
                        step=10,
                        label="Epochs (0 = automatic)",
                        info="More = closer fit of the voice until it overfits "
                        "(early stopping keeps the best epoch); fewer = faster, "
                        "rougher. Automatic aims at ~7000 steps.",
                    )
                    batch = gr.Slider(
                        0,
                        32,
                        value=DEFAULT_TRAIN.batch_size,
                        step=1,
                        label="Batch size (0 = automatic)",
                        info="Larger = faster, smoother training but more VRAM; "
                        "smaller = fits small GPUs, noisier updates. Automatic "
                        "picks from the GPU's memory.",
                    )
                with gr.Row():
                    learning_rate = gr.Number(
                        value=DEFAULT_TRAIN.learning_rate,
                        minimum=1e-5,
                        maximum=1e-3,
                        label="Learning rate",
                        info="Higher = learns faster but can get unstable or "
                        "overfit; lower = slower, safer. 1e-4 is RVC's standard.",
                    )
                    lr_decay = gr.Number(
                        value=DEFAULT_TRAIN.lr_decay,
                        minimum=0.99,
                        maximum=1.0,
                        label="Learning rate decay per epoch",
                        info="Closer to 1 = keeps learning at full speed longer; "
                        "lower = settles down sooner.",
                    )
                with gr.Row():
                    early = gr.Checkbox(
                        value=DEFAULT_TRAIN.early_stopping,
                        label="Stop when it stops improving",
                        info="Scores held-out clips through the real conversion "
                        "path and keeps the best epoch.",
                    )
                    patience = gr.Slider(
                        10,
                        200,
                        value=DEFAULT_TRAIN.patience,
                        step=5,
                        label="Patience (epochs)",
                        info="Epochs without improvement before stopping. Higher "
                        "= rides out plateaus, longer runs; lower = stops sooner.",
                    )
                with gr.Row():
                    precision = gr.Dropdown(
                        ["auto", "fp32", "bf16", "fp16"],
                        value=DEFAULT_TRAIN.precision,
                        label="Precision",
                        info="auto = bf16 on RTX 30xx and newer, else fp32. fp32 "
                        "= most stable, slower; fp16 = fast on older GPUs, can "
                        "overflow.",
                    )
                    save_every = gr.Slider(
                        1,
                        50,
                        value=DEFAULT_TRAIN.save_every,
                        step=1,
                        label="Save every N epochs",
                        info="Lower = more checkpoints and preview clips (more "
                        "disk, slightly slower); higher = fewer.",
                    )
                with gr.Row():
                    c_mel = gr.Slider(
                        1,
                        100,
                        value=DEFAULT_TRAIN.c_mel,
                        step=1,
                        label="Spectrogram loss weight",
                        info="Higher = closer, cleaner timbre, can sound flatter; "
                        "lower = more freedom and more artifacts. 45 is standard.",
                    )
                    c_kl = gr.Slider(
                        0,
                        5,
                        value=DEFAULT_TRAIN.c_kl,
                        step=0.1,
                        label="KL loss weight",
                        info="Higher = smoother, more generic voice; lower = "
                        "sharper detail but less stable. 1 is standard.",
                    )
                with gr.Row():
                    keep_snapshots = gr.Checkbox(
                        value=DEFAULT_TRAIN.keep_snapshots,
                        label="Keep a model file at every save",
                    )
                    low_vram = gr.Checkbox(
                        value=DEFAULT_TRAIN.gradient_checkpointing,
                        label="Low VRAM mode",
                        info="Gradient checkpointing: fits about twice the batch "
                        "in the same memory, ~30% slower.",
                    )
            with gr.Accordion("Dataset slicing", open=False):
                with gr.Row():
                    max_seconds = gr.Slider(
                        2,
                        10,
                        value=DEFAULT_SEGMENT.max_seconds,
                        step=0.1,
                        label="Longest segment (s)",
                        info="Longer = more context per step but more VRAM per "
                        "item; shorter = more, smaller segments.",
                    )
                    min_seconds = gr.Slider(
                        0.3,
                        3,
                        value=DEFAULT_SEGMENT.min_seconds,
                        step=0.1,
                        label="Shortest segment (s)",
                        info="Shorter clips are dropped. Higher = cleaner but "
                        "less data; lower = keeps short words and breaths.",
                    )
                with gr.Row():
                    silence_db = gr.Slider(
                        -60,
                        -20,
                        value=DEFAULT_SEGMENT.silence_db,
                        step=1,
                        label="Silence threshold (dB)",
                        info="Level treated as silence when slicing. Higher = "
                        "cuts more aggressively and drops quiet tails; lower = "
                        "keeps breaths and room tone.",
                    )
                    target_lufs = gr.Slider(
                        -30,
                        -12,
                        value=DEFAULT_SEGMENT.target_lufs,
                        step=1,
                        label="Target loudness (LUFS)",
                        info="Every clip is normalised to this. Higher = louder "
                        "with less headroom; lower = quieter, safer.",
                    )

        options = [
            preset,
            isolate,
            vocal_model,
            backing,
            lead_model,
            dereverb,
            dereverb_model,
            denoise,
            denoise_model,
            sample_rate,
            pretrained,
            epochs,
            batch,
            learning_rate,
            lr_decay,
            early,
            patience,
            precision,
            save_every,
            c_mel,
            c_kl,
            keep_snapshots,
            low_vram,
            fresh,
            max_seconds,
            min_seconds,
            silence_db,
            target_lufs,
        ]
        with gr.Row():
            reset_button("Reset options", options)
            start_btn = gr.Button(
                "Start / resume training",
                variant="primary",
                scale=4,
                elem_classes="vr-big-button",
            )
            stop_btn = gr.Button("Stop", variant="stop", scale=1, min_width=100)

        status = gr.HTML(_status_html(current))
        with gr.Row():
            preview = gr.Audio(
                label="Latest preview of the voice",
                type="filepath",
                interactive=False,
                buttons=["download"],
            )
            reference = gr.Audio(
                label="Original clip",
                type="filepath",
                interactive=False,
                buttons=["download"],
            )
        with gr.Accordion("Details", open=False):
            plot = gr.LinePlot(
                _loss_frame(current),
                x="epoch",
                y="loss",
                color="series",
                title="Mel loss (lower is better)",
                height=240,
            )
            cleaned = gr.Audio(
                label="A cleaned training clip",
                type="filepath",
                interactive=False,
                buttons=["download"],
            )
            report = gr.Markdown(_report_md(current))
            log_box = gr.Textbox(
                label="Log",
                lines=12,
                max_lines=12,
                autoscroll=True,
                elem_classes="vr-log",
                interactive=False,
            )
    timer = gr.Timer(2.0)

    def apply_preset(value):
        p = PRESETS[value]
        return (
            gr.Checkbox(value=p["isolate"]),
            gr.Checkbox(value=p["backing"]),
            gr.Checkbox(value=p["dereverb"]),
            gr.Checkbox(value=p["denoise"]),
        )

    preset.change(apply_preset, preset, [isolate, backing, dereverb, denoise])

    def refresh_view(selected):
        sample, ref, clip = _samples(selected)
        return (
            _status_html(selected),
            _loss_frame(selected),
            sample,
            ref,
            clip,
            _report_md(selected),
            jobs.tail_log(selected) if selected else "",
            _source_summary(selected),
        )

    view_outputs = [
        status,
        plot,
        preview,
        reference,
        cleaned,
        report,
        log_box,
        sources_md,
    ]

    def tick(selected, last_state):
        """Cheap poll: only push heavy updates while something changes."""
        unchanged = [gr.skip()] * len(view_outputs) + [last_state]
        if not selected:
            return unchanged
        try:
            st = jobs.status(selected)
            signature = f"{st.state}|{st.updated}"
            if signature == last_state:
                return unchanged
            return [*refresh_view(selected), signature]
        except Exception:
            # A background poll must never surface errors in the UI.
            log.exception("Status refresh failed")
            return unchanged

    last = gr.State("")
    timer.tick(tick, [session, last], [*view_outputs, last])

    def select_session(selected):
        return [selected or "", *refresh_view(selected)]

    session.change(
        select_session, session, [name, *view_outputs], show_progress="hidden"
    )
    refresh_sessions.click(
        lambda current: gr.Dropdown(
            choices=jobs.experiments(),
            value=current if current in jobs.experiments() else None,
        ),
        session,
        session,
    )

    def _launch(stages, args):
        (
            voice_name,
            uploads,
            folder_path,
            _preset,
            iso,
            voc_m,
            back,
            lead_m,
            derev,
            derev_m,
            den,
            den_m,
            sr,
            pre,
            ep,
            bs,
            lr,
            decay,
            es,
            pat,
            prec,
            save_n,
            mel_w,
            kl_w,
            snaps,
            lowvram,
            restart,
            max_s,
            min_s,
            sil_db,
            lufs,
        ) = args
        jobs.ensure_idle()
        voice_name = (voice_name or "").strip()
        if not voice_name:
            raise gr.Error("Give the voice a name first.")
        safe = paths.safe_name(voice_name, "voice")
        exp = jobs.experiment_dir(safe)
        src_dir = exp / "sources"
        src_dir.mkdir(parents=True, exist_ok=True)
        for upload in uploads or []:
            upload = Path(upload)
            shutil.copy2(upload, src_dir / paths.safe_name(upload.name, "audio"))
        sources = [str(src_dir)]
        if folder_path and folder_path.strip():
            folder_dir = Path(folder_path.strip().strip('"')).expanduser()
            if not folder_dir.is_dir():
                raise gr.Error(f"Folder not found: {folder_dir}")
            sources.append(str(folder_dir))
        needs_data = "prepare" in stages
        if needs_data and not any(
            p.is_file() for s in sources for p in Path(s).rglob("*")
        ):
            raise gr.Error("Add some recordings (upload files or give a folder).")
        cleaning = CleaningOptions(
            isolate_vocals=iso,
            vocal_model=voc_m,
            remove_backing_vocals=back,
            lead_model=lead_m,
            dereverb=derev,
            dereverb_model=derev_m,
            denoise=den,
            denoise_model=den_m,
        )
        segmenting = SegmentOptions(
            target_lufs=float(lufs),
            silence_db=float(sil_db),
            max_seconds=float(max_s),
            min_seconds=float(min_s),
        )
        if segmenting.min_seconds >= segmenting.max_seconds:
            raise gr.Error("The shortest segment must be shorter than the longest.")
        train = TrainConfig(
            sample_rate=int(sr),
            epochs=int(ep),
            batch_size=int(bs),
            pretrained=pre,
            precision=prec,
            save_every=int(save_n),
            keep_snapshots=bool(snaps),
            early_stopping=bool(es),
            patience=int(pat),
            learning_rate=float(lr),
            lr_decay=float(decay),
            c_mel=float(mel_w),
            c_kl=float(kl_w),
            gradient_checkpointing=bool(lowvram),
        )
        write_job(
            exp,
            safe,
            sources,
            cleaning,
            segmenting,
            train,
            stages,
            fresh=bool(restart) and "train" in stages,
        )
        jobs.start(safe)
        choices = jobs.experiments()
        return [
            gr.Dropdown(choices=choices, value=safe),
            None,
            *refresh_view(safe)[: len(view_outputs)],
        ]

    settings_inputs = [name, files, folder, *options]
    launch_outputs = [session, files, *view_outputs]

    @friendly_errors
    def prepare_only(*args):
        return _launch(["prepare"], args)

    @friendly_errors
    def start_training(*args):
        return _launch(FULL_PIPELINE, args)

    @friendly_errors
    def publish_now(selected):
        if not selected:
            raise gr.Error("Select a training session first.")
        if jobs.status(selected).active:
            raise gr.Error("Training is running; stop it first (it saves on stop).")
        jobs.ensure_idle()
        exp = jobs.experiment_dir(selected)
        ckpt = exp / "checkpoints"
        if not (ckpt.is_dir() and any(ckpt.glob("model_*.pth"))):
            raise gr.Error("This session has no trained model yet.")
        spec = json.loads(job_file(exp).read_text())
        spec["stages"] = ["index", "publish"]
        spec["fresh"] = False
        job_file(exp).write_text(json.dumps(spec, indent=2))
        jobs.start(selected)
        gr.Info("Saving the current model to your voice library…")
        return refresh_view(selected)

    @friendly_errors
    def stop(selected):
        if not selected or not jobs.status(selected).active:
            raise gr.Error("Nothing is running.")
        jobs.request_stop(selected)
        gr.Info("Stopping after the current step; a checkpoint will be saved.")

    for button, handler in ((prepare_btn, prepare_only), (start_btn, start_training)):
        button.click(
            handler,
            settings_inputs,
            launch_outputs,
            show_progress="minimal",
            show_progress_on=status,
        )
    stop_btn.click(stop, session, None)
    publish_btn.click(
        publish_now,
        session,
        view_outputs,
        show_progress="minimal",
        show_progress_on=status,
    )

    def on_page_load():
        """Re-attach to the running (or latest) session after a reload."""
        selected = jobs.latest_experiment()
        return [
            gr.Dropdown(choices=jobs.experiments(), value=selected),
            selected or "",
            *refresh_view(selected),
        ]

    return {
        "session": session,
        "on_load": on_page_load,
        "load_outputs": [session, name, *view_outputs],
    }
