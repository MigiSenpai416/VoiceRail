"""Voice models tab: import, inspect, rename, delete and download models."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from .. import library
from .common import describe_pitch, friendly_errors, model_choices

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


def build() -> dict:
    table = gr.Dataframe(_table(), interactive=False, label="Your voice models")
    with gr.Row(equal_height=False):
        with gr.Column():
            gr.Markdown("### Import a model")
            uploads = gr.File(
                label="RVC model files (.pth, optional .index) or a .zip",
                file_count="multiple",
                file_types=[".pth", ".index", ".zip"],
                type="filepath",
            )
            upload_name = gr.Textbox(label="Name (optional)")
            import_btn = gr.Button("Import files", variant="primary")
            gr.Markdown("…or from a link (Hugging Face or any direct download):")
            url = gr.Textbox(
                label="Download link",
                placeholder="https://huggingface.co/user/repo/resolve/main/model.zip",
            )
            url_name = gr.Textbox(label="Name (optional)")
            url_btn = gr.Button("Download and import")
        with gr.Column():
            gr.Markdown("### Manage")
            choices = model_choices()
            selected = gr.Dropdown(
                choices, value=choices[0] if choices else None, label="Model"
            )
            new_name = gr.Textbox(label="New name")
            with gr.Row():
                rename_btn = gr.Button("Rename")
                export_btn = gr.Button("Get files")
            files_out = gr.File(label="Model files", interactive=False)
            confirm = gr.Checkbox(
                value=False, label="Yes, delete this model permanently"
            )
            delete_btn = gr.Button("Delete", variant="stop")
            status = gr.Markdown()

    def refreshed(name: str | None = None):
        names = model_choices()
        value = name if name in names else (names[0] if names else None)
        return _table(), gr.Dropdown(choices=names, value=value)

    @friendly_errors
    def do_import(paths_, name):
        if not paths_:
            raise gr.Error("Choose at least one .pth or .zip file.")
        info = library.import_files([Path(p) for p in paths_], name or None)
        return (
            *refreshed(info.name),
            None,
            f"✅ Imported **{info.name}** ({info.summary()})",
        )

    @friendly_errors
    def do_url(link, name, progress=gr.Progress()):
        if not link or not link.strip():
            raise gr.Error("Paste a download link first.")
        info = library.import_url(
            link, name or None, progress=lambda f, m: progress(f, desc=m)
        )
        return (
            *refreshed(info.name),
            "",
            f"✅ Imported **{info.name}** ({info.summary()})",
        )

    @friendly_errors
    def do_rename(old, new):
        if not old or not new or not new.strip():
            raise gr.Error("Select a model and type a new name.")
        final = library.rename_model(old, new.strip())
        return *refreshed(final), "", f"Renamed to **{final}**"

    @friendly_errors
    def do_delete(name, sure):
        if not name:
            raise gr.Error("Select a model.")
        if not sure:
            raise gr.Error("Tick the confirmation box to delete.")
        library.delete_model(name)
        return *refreshed(), False, f"Deleted **{name}**"

    @friendly_errors
    def do_export(name):
        if not name:
            raise gr.Error("Select a model.")
        model, index = library.model_paths(name)
        return [str(model)] + ([str(index)] if index else [])

    import_btn.click(
        do_import, [uploads, upload_name], [table, selected, uploads, status]
    )
    url_btn.click(do_url, [url, url_name], [table, selected, url, status])
    rename_btn.click(
        do_rename, [selected, new_name], [table, selected, new_name, status]
    )
    delete_btn.click(do_delete, [selected, confirm], [table, selected, confirm, status])
    export_btn.click(do_export, selected, files_out)
    return {"table": table, "selected": selected, "refresh": refreshed}
