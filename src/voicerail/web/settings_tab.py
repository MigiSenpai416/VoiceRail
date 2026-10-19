"""Settings tab: system info, base model downloads and storage."""

from __future__ import annotations

import contextlib
import shutil

import gradio as gr
import pandas as pd

from .. import assets, jobs, paths
from ..device import describe
from .common import friendly_errors


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


def build() -> None:
    gr.Markdown(_system_md())
    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Markdown("### Base models")
            gr.Markdown(
                "Downloaded automatically when first needed. You can also fetch the "
                "essentials now (pitch estimator, voice encoder and the standard "
                "40 kHz pretrained model, about 0.6 GB)."
            )
            table = gr.Dataframe(_assets_table(), interactive=False)
            download_btn = gr.Button("Download essentials", variant="primary")
            download_status = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Storage")
            storage = gr.Markdown(_storage_md())
            with gr.Row():
                clear_cache = gr.Button("Clear conversion cache")
                refresh = gr.Button("Refresh")
            sessions = jobs.experiments()
            session = gr.Dropdown(sessions, label="Training session to delete")
            confirm = gr.Checkbox(
                value=False,
                label="Yes, delete its dataset, checkpoints and logs "
                "(published voice models are kept)",
            )
            delete_session = gr.Button("Delete training session", variant="stop")

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

    download_btn.click(download, None, [table, download_status])
    clear_cache.click(clear, None, storage)
    refresh.click(
        lambda: (
            _storage_md(),
            _assets_table(),
            gr.Dropdown(choices=jobs.experiments()),
        ),
        None,
        [storage, table, session],
    )
    delete_session.click(
        remove_session, [session, confirm], [storage, session, confirm]
    )
