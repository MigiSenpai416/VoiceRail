"""Look and feel of the web UI."""

from __future__ import annotations

import gradio as gr


def theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="sky",
        neutral_hue="slate",
        radius_size="lg",
        font=(
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
    ).set(
        button_large_padding="14px 22px",
        block_title_text_weight="600",
    )


CSS = """
.vr-header { display:flex; align-items:center; justify-content:space-between;
  gap:16px; padding:6px 4px 14px 4px; flex-wrap:wrap; }
.vr-header h1 { margin:0; font-size:1.9rem; letter-spacing:-0.02em; }
.vr-title { display:flex; align-items:center; gap:10px; }
.vr-header p { margin:2px 0 0 0; opacity:.75; }
.vr-badge { padding:6px 12px; border-radius:999px; font-size:.85rem;
  border:1px solid var(--border-color-primary); white-space:nowrap; }
.vr-badge.gpu { background: color-mix(in srgb, var(--color-accent) 12%, transparent); }
.vr-badge.cpu { background: color-mix(in srgb, #f59e0b 18%, transparent); }
.vr-card { border:1px solid var(--border-color-primary); border-radius:14px;
  padding:14px 16px; background: var(--block-background-fill); }
.vr-card h3 { margin:0 0 6px 0; font-size:1.05rem; }
.vr-muted { opacity:.7; font-size:.9rem; }
.vr-bar { height:10px; border-radius:999px; overflow:hidden; margin:10px 0 6px 0;
  background: var(--border-color-primary); }
.vr-bar > div { height:100%; border-radius:999px;
  background: linear-gradient(90deg, var(--color-accent), #38bdf8);
  transition: width .6s ease; }
.vr-state { font-weight:600; text-transform:capitalize; }
.vr-state.running { color: var(--color-accent); }
.vr-state.finished { color:#16a34a; }
.vr-state.failed, .vr-state.crashed { color:#dc2626; }
.vr-state.stopped { color:#d97706; }
.vr-steps { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
.vr-step { font-size:.78rem; padding:3px 9px; border-radius:999px;
  border:1px solid var(--border-color-primary); opacity:.6; }
.vr-step.done { opacity:1; border-color:#16a34a; }
.vr-step.current { opacity:1; border-color: var(--color-accent);
  background: color-mix(in srgb, var(--color-accent) 12%, transparent); }
.vr-big-button { min-height:52px; font-size:1.05rem !important; }
.vr-log textarea { font-family: var(--font-mono); font-size:.8rem !important; }
footer { display:none !important; }
"""
