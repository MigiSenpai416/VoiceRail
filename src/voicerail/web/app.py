"""The VoiceRail web UI."""

from __future__ import annotations

import html

import gradio as gr

from .. import __version__, paths
from ..device import describe
from . import convert_tab, library_tab, train_tab
from .common import model_card, refreshed_models
from .theme import CSS, theme

LOGO = (
    "<svg width='34' height='34' viewBox='0 0 24 24' fill='none' aria-hidden='true'>"
    "<rect width='24' height='24' rx='7' fill='url(#vr-g)'/>"
    "<path d='M6 13v-2M9 16V8M12 18V6M15 15V9M18 13v-2' stroke='white' "
    "stroke-width='1.9' stroke-linecap='round'/><defs><linearGradient id='vr-g' "
    "x1='0' y1='0' x2='24' y2='24'><stop stop-color='#7c3aed'/>"
    "<stop offset='1' stop-color='#0ea5e9'/></linearGradient></defs></svg>"
)


def _header() -> str:
    info = describe()
    if info.device == "cpu":
        badge = "<span class='vr-badge cpu'>CPU only · slow</span>"
    else:
        vram = f" · {info.vram_gb:.0f} GB" if info.vram_gb else ""
        badge = f"<span class='vr-badge gpu'>{html.escape(info.name)}{vram}</span>"
    return (
        "<div class='vr-header'><div><h1 class='vr-title'>"
        f"{LOGO}<span>VoiceRail</span></h1>"
        "<p>Give any song or speech a new voice, or train a voice from your "
        f"own recordings.</p></div><div>{badge} "
        f"<span class='vr-badge'>v{__version__}</span></div></div>"
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="VoiceRail", analytics_enabled=False) as app:
        gr.HTML(_header())
        with gr.Tabs():
            with gr.Tab("Convert", id="convert") as convert_page:
                convert_ui = convert_tab.build()
            with gr.Tab("Train", id="train"):
                train_ui = train_tab.build()
            with gr.Tab("Library", id="library") as library_page:
                library_ui = library_tab.build()

        # Voices can appear from training or imports in other tabs.
        convert_page.select(
            refreshed_models, convert_ui["model"], convert_ui["model"]
        ).then(model_card, convert_ui["model"], convert_ui["card"])
        # The page config is built once; refresh live state on every visit.
        app.load(refreshed_models, convert_ui["model"], convert_ui["model"]).then(
            model_card, convert_ui["model"], convert_ui["card"]
        )
        app.load(train_ui["on_load"], None, train_ui["load_outputs"])
        library_page.select(
            lambda name: library_ui["refresh"](name),
            library_ui["selected"],
            [library_ui["table"], library_ui["selected"]],
        )
    return app


def launch(
    host: str = "127.0.0.1",
    port: int | None = 8080,
    share: bool = False,
    open_browser: bool = True,
) -> None:
    app = build_app()
    app.queue(default_concurrency_limit=2)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=open_browser,
        theme=theme(),
        css=CSS,
        allowed_paths=[str(paths.home())],
        max_file_size="4gb",
        show_error=True,
    )
