"""Helpers shared by the UI tabs."""

from __future__ import annotations

import html
import logging
import traceback
from collections.abc import Callable
from functools import wraps

import gradio as gr

from .. import library

log = logging.getLogger(__name__)


def model_choices() -> list[str]:
    try:
        return library.model_names()
    except Exception:
        log.exception("Could not list voice models")
        return []


def refreshed_models(current: str | None = None) -> gr.Dropdown:
    choices = model_choices()
    value = current if current in choices else (choices[0] if choices else None)
    return gr.Dropdown(choices=choices, value=value)


def describe_pitch(hz: float | None) -> str:
    if not hz:
        return "unknown"
    if hz < 145:
        range_name = "low / male"
    elif hz < 200:
        range_name = "mid"
    else:
        range_name = "high / female"
    return f"~{hz:.0f} Hz ({range_name})"


def model_card(name: str | None) -> str:
    if not name:
        return "*No voice model selected. Import or train one first.*"
    try:
        info = library.get_info(name)
    except Exception as error:
        return f"⚠️ {html.escape(str(error))}"
    parts = [f"**{html.escape(info.name)}** · {info.summary()}"]
    if info.f0_median:
        parts.append(f"typical pitch {describe_pitch(info.f0_median)}")
    if not info.has_index:
        parts.append("no index: timbre retrieval disabled")
    return " · ".join(parts)


def friendly_errors(fn: Callable) -> Callable:
    """Show exceptions as readable UI errors instead of a generic failure."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as error:
            log.error("%s failed: %s\n%s", fn.__name__, error, traceback.format_exc())
            raise gr.Error(str(error) or error.__class__.__name__) from error

    return wrapper
