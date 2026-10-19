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
from .common import friendly_errors

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


def _status_html(name: str | None) -> str:
    if not name:
        return (
            "<div class='vr-card'><h3>No training session selected</h3>"
            "<div class='vr-muted'>Name your voice, add recordings and press "
            "<b>Start training</b>.</div></div>"
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
            "<div style='margin-top:8px'>✅ The voice model is ready in the "
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
        return "*No recordings added to this session yet.*"
    size = sum(p.stat().st_size for p in files) / 1024**2
    return f"{len(files)} recording(s) stored for this session ({size:.0f} MB)."


def build() -> dict:
    sessions = jobs.experiments()
    current = jobs.latest_experiment()
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            with gr.Group():
                with gr.Row():
                    session = gr.Dropdown(
                        sessions,
                        value=current,
                        label="Training session",
                        info="Pick an existing session to monitor or resume it.",
                        allow_custom_value=False,
                        scale=4,
                    )
                    refresh_sessions = gr.Button("↻", scale=0, min_width=48)
                name = gr.Textbox(
                    label="Voice name",
                    placeholder="e.g. MyVoice",
                    value=current or "",
                )
                files = gr.File(
                    label="Recordings (mp3, wav, flac, m4a, ...)",
                    file_count="multiple",
                    file_types=UPLOAD_TYPES,
                    type="filepath",
                )
                folder = gr.Textbox(
                    label="…or a folder on this computer",
                    placeholder="C:\\Users\\me\\voice-recordings",
                )
                sources_md = gr.Markdown(_source_summary(current))

            preset = gr.Radio(
                list(PRESETS),
                value=next(iter(PRESETS)),
                label="What kind of recordings are these?",
                info="Chooses the clean-up steps below.",
            )
            with gr.Accordion("Clean-up steps", open=False):
                isolate = gr.Checkbox(value=True, label="Remove music / background")
                vocal_model = gr.Dropdown(
                    [m.label for m in VOCAL_MODELS],
                    value=VOCAL_MODELS[0].label,
                    label="Vocal separation model",
                )
                backing = gr.Checkbox(
                    value=True, label="Remove backing vocals / harmonies"
                )
                lead_model = gr.Dropdown(
                    [m.label for m in LEAD_MODELS],
                    value=LEAD_MODELS[0].label,
                    label="Lead vocal model",
                )
                dereverb = gr.Checkbox(value=True, label="Remove reverb / echo")
                dereverb_model = gr.Dropdown(
                    [m.label for m in DEREVERB_MODELS],
                    value=DEREVERB_MODELS[0].label,
                    label="De-reverb model",
                )
                denoise = gr.Checkbox(value=False, label="Remove noise")
                denoise_model = gr.Dropdown(
                    [m.label for m in DENOISE_MODELS],
                    value=DENOISE_MODELS[0].label,
                    label="Denoise model",
                )

            with gr.Accordion("Training settings", open=False):
                sample_rate = gr.Radio(
                    [str(sr) for sr in SAMPLE_RATES],
                    value="40000",
                    label="Sample rate",
                    info="40k is the best-supported choice; 48k for hi-fi singing.",
                )
                pretrained = gr.Dropdown(
                    list(PRETRAINED_SETS),
                    value=DEFAULT_PRETRAINED,
                    label="Base (pretrained) model",
                )
                with gr.Row():
                    epochs = gr.Slider(
                        0, 1000, value=0, step=10, label="Epochs (0 = automatic)"
                    )
                    batch = gr.Slider(
                        0, 32, value=0, step=1, label="Batch size (0 = automatic)"
                    )
                with gr.Row():
                    early = gr.Checkbox(
                        value=True,
                        label="Stop when it stops improving",
                        info="Uses held-out clips to detect the best epoch.",
                    )
                    patience = gr.Slider(
                        10, 200, value=40, step=5, label="Patience (epochs)"
                    )
                with gr.Row():
                    precision = gr.Dropdown(
                        ["auto", "fp32", "bf16", "fp16"],
                        value="auto",
                        label="Precision",
                        info="auto = bf16 on RTX 30xx and newer, else fp32",
                    )
                    save_every = gr.Slider(
                        1, 50, value=10, step=1, label="Save every N epochs"
                    )
                keep_snapshots = gr.Checkbox(
                    value=False, label="Also keep a model file at every save"
                )
                low_vram = gr.Checkbox(
                    value=False,
                    label="Low VRAM mode (gradient checkpointing, slower)",
                )
                fresh = gr.Checkbox(
                    value=False,
                    label="Start over (discard this session's previous training)",
                )

            with gr.Row():
                prepare_btn = gr.Button("Prepare dataset only")
                start_btn = gr.Button(
                    "Start / resume training",
                    variant="primary",
                    elem_classes="vr-big-button",
                )
            with gr.Row():
                stop_btn = gr.Button("Stop", variant="stop")
                publish_btn = gr.Button("Save current model now")

        with gr.Column(scale=5):
            status = gr.HTML(_status_html(current))
            plot = gr.LinePlot(
                _loss_frame(current),
                x="epoch",
                y="loss",
                color="series",
                title="Mel loss (lower is better)",
                height=260,
            )
            with gr.Row():
                preview = gr.Audio(
                    label="Latest preview",
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                )
                reference = gr.Audio(
                    label="Original",
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                )
            cleaned = gr.Audio(
                label="Cleaned training clip",
                type="filepath",
                interactive=False,
                buttons=["download"],
            )
            report = gr.Markdown(_report_md(current))
            log_box = gr.Textbox(
                label="Log",
                lines=14,
                max_lines=14,
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
            es,
            pat,
            prec,
            save_n,
            snaps,
            lowvram,
            restart,
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
            gradient_checkpointing=bool(lowvram),
        )
        write_job(
            exp,
            safe,
            sources,
            cleaning,
            SegmentOptions(),
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

    settings_inputs = [
        name,
        files,
        folder,
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
        early,
        patience,
        precision,
        save_every,
        keep_snapshots,
        low_vram,
        fresh,
    ]
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
