"""Library tab: voice models, base models, storage and system info."""

from __future__ import annotations

import contextlib
import shutil
from pathlib import Path

import gradio as gr
import pandas as pd

from .. import assets, jobs, library, paths
from ..device import describe
from .common import describe_pitch, friendly_errors, heading, model_choices

COLUMNS = ["Name", "Type", "Sample rate", "Epochs", "Index", "Pitch", "Size", "Source"]


def _table() -> pd.DataFrame:
    rows = []
    for m in library.list_models():
        rows.append(
            [
                m.name,
                f"RVC {m.version}" + ("" if m.pitch_guided else " (no pitch)"),
                f"{m.sample_rate // 1000} kHz",
                m.epochs or "—",
                "yes" if m.has_index else "no",
                describe_pitch(m.f0_median),
                f"{m.size_mb:.0f} MB",
                m.source,
            ]
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _system_md() -> str:
    info = describe()
    vram = f" · {info.vram_gb:.1f} GB VRAM" if info.vram_gb else ""
    lines = [
        f"**Device:** {info.name} ({info.device}){vram}",
        f"**PyTorch:** {info.torch_version} · accelerator: {info.accelerator}",
        f"**Data folder:** `{paths.home()}`",
    ]
    if info.device == "cpu":
        lines.append(
            "⚠️ No GPU detected. Conversion works but is slow; training on a CPU "
            "is impractical. Install the CUDA build (see README)."
        )
    return "\n\n".join(lines)


def _assets_table() -> pd.DataFrame:
    rows = []
    for key, asset in assets.ASSETS.items():
        rows.append(
            [
                key,
                f"{asset.size / 1024**2:.0f} MB",
                "✅ ready" if assets.is_ready(key) else "not downloaded",
            ]
        )
    return pd.DataFrame(rows, columns=["File", "Size", "Status"])


def _folder_size(path) -> float:
    total = 0
    for p in path.rglob("*"):
        with contextlib.suppress(OSError):
            if p.is_file():
                total += p.stat().st_size
    return total / 1024**3


def _storage_md() -> str:
    parts = {
        "Voice models": paths.models_dir(),
        "Training sessions": paths.training_dir(),
        "Converted outputs": paths.outputs_dir(),
        "Cache": paths.cache_dir(),
        "Base & separation models": paths.assets_dir(),
    }
    return "\n".join(
        f"- **{label}:** {_folder_size(path):.2f} GB — `{path}`"
        for label, path in parts.items()
    )


def build() -> dict:
    table = gr.Dataframe(_table(), interactive=False, label="Your voices")
    with gr.Row(equal_height=False):
        with gr.Column(min_width=340):
            heading("Add a voice")
            uploads = gr.File(
                label="Model files: .pth (with its .index) or a .zip",
                file_count="multiple",
                file_types=[".pth", ".index", ".zip"],
                type="filepath",
            )
            url = gr.Textbox(
                label="…or a download link",
                placeholder="https://huggingface.co/user/repo/resolve/main/model.zip",
                info="Hugging Face or any direct link to a .pth or .zip.",
            )
            with gr.Row():
                add_name = gr.Textbox(
                    label="Name (optional)",
                    placeholder="defaults to the file name",
                    scale=3,
                )
                add_btn = gr.Button("Add", variant="primary", scale=1)
        with gr.Column(min_width=340):
            heading("Manage")
            choices = model_choices()
            selected = gr.Dropdown(
                choices, value=choices[0] if choices else None, label="Voice"
            )
            with gr.Row():
                new_name = gr.Textbox(label="New name", scale=3)
                rename_btn = gr.Button("Rename", scale=1)
            with gr.Row():
                export_btn = gr.Button("Get the files")
                delete_btn = gr.Button("Delete", variant="stop")
            confirm = gr.Checkbox(value=False, label="Yes, delete it permanently")
            files_out = gr.File(label="Files", interactive=False)
    status = gr.Markdown(elem_classes="vr-hint")

    with gr.Accordion("Base models, storage and system", open=False):
        gr.Markdown(_system_md())
        with gr.Row(equal_height=False):
            with gr.Column(min_width=340):
                heading("Base models")
                gr.Markdown(
                    "Downloaded automatically when first needed. Fetch the "
                    "essentials now (pitch estimator, voice encoder and the "
                    "standard 40 kHz pretrained model, about 0.6 GB) to avoid a "
                    "wait later.",
                    elem_classes="vr-hint",
                )
                assets_table = gr.Dataframe(_assets_table(), interactive=False)
                download_btn = gr.Button("Download essentials", variant="primary")
                download_status = gr.Markdown(elem_classes="vr-hint")
            with gr.Column(min_width=340):
                heading("Storage")
                storage = gr.Markdown(_storage_md())
                with gr.Row():
                    clear_cache = gr.Button("Clear conversion cache", size="sm")
                    refresh = gr.Button("Refresh", size="sm")
                session = gr.Dropdown(
                    jobs.experiments(),
                    label="Delete a training session",
                    info="Removes its dataset, checkpoints and logs; published "
                    "voices are kept.",
                )
                with gr.Row():
                    confirm_session = gr.Checkbox(
                        value=False, label="Yes, delete it", scale=3
                    )
                    delete_session = gr.Button("Delete", variant="stop", scale=1)

    def refreshed(name: str | None = None):
        names = model_choices()
        value = name if name in names else (names[0] if names else None)
        return _table(), gr.Dropdown(choices=names, value=value)

    @friendly_errors
    def do_add(paths_, link, name, progress=gr.Progress()):
        if paths_:
            info = library.import_files([Path(p) for p in paths_], name or None)
        elif link and link.strip():
            info = library.import_url(
                link, name or None, progress=lambda f, m: progress(f, desc=m)
            )
        else:
            raise gr.Error("Choose model files or paste a download link first.")
        return (
            *refreshed(info.name),
            None,
            "",
            f"✅ Added **{info.name}** ({info.summary()})",
        )

    @friendly_errors
    def do_rename(old, new):
        if not old or not new or not new.strip():
            raise gr.Error("Select a voice and type a new name.")
        final = library.rename_model(old, new.strip())
        return *refreshed(final), "", f"Renamed to **{final}**"

    @friendly_errors
    def do_delete(name, sure):
        if not name:
            raise gr.Error("Select a voice.")
        if not sure:
            raise gr.Error("Tick the confirmation box to delete.")
        library.delete_model(name)
        return *refreshed(), False, f"Deleted **{name}**"

    @friendly_errors
    def do_export(name):
        if not name:
            raise gr.Error("Select a voice.")
        model, index = library.model_paths(name)
        return [str(model)] + ([str(index)] if index else [])

    @friendly_errors
    def download(progress=gr.Progress()):
        keys = [
            *assets.CORE_ASSETS,
            *assets.PRETRAINED_SETS[assets.DEFAULT_PRETRAINED][40000],
        ]
        assets.ensure(keys, lambda f, m: progress(f, desc=m))
        return _assets_table(), "✅ Base models are ready."

    @friendly_errors
    def clear():
        shutil.rmtree(paths.cache_dir() / "convert", ignore_errors=True)
        return _storage_md()

    @friendly_errors
    def remove_session(name, sure):
        if not name:
            raise gr.Error("Select a session.")
        if not sure:
            raise gr.Error("Tick the confirmation box first.")
        if jobs.status(name).active:
            raise gr.Error("This session is training right now; stop it first.")
        shutil.rmtree(jobs.experiment_dir(name), ignore_errors=True)
        return (
            _storage_md(),
            gr.Dropdown(choices=jobs.experiments(), value=None),
            False,
        )

    add_btn.click(
        do_add, [uploads, url, add_name], [table, selected, uploads, url, status]
    )
    rename_btn.click(
        do_rename, [selected, new_name], [table, selected, new_name, status]
    )
    delete_btn.click(do_delete, [selected, confirm], [table, selected, confirm, status])
    export_btn.click(do_export, selected, files_out)
    download_btn.click(download, None, [assets_table, download_status])
    clear_cache.click(clear, None, storage)
    refresh.click(
        lambda: (
            _storage_md(),
            _assets_table(),
            gr.Dropdown(choices=jobs.experiments()),
        ),
        None,
        [storage, assets_table, session],
    )
    delete_session.click(
        remove_session,
        [session, confirm_session],
        [storage, session, confirm_session],
    )
    return {"table": table, "selected": selected, "refresh": refreshed}
