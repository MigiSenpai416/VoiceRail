"""Look and feel of the web UI."""

from __future__ import annotations

import gradio as gr


def theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue="violet",
        secondary_hue="sky",
        neutral_hue="slate",
        radius_size="md",
        font=(
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
    ).set(
        block_shadow="none",
        block_title_text_weight="600",
        block_label_text_weight="600",
        button_large_padding="14px 22px",
        button_primary_shadow="none",
        button_secondary_shadow="none",
    )


CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.vr-header { display:flex; align-items:center; justify-content:space-between;
  gap:16px; padding:6px 4px 12px 4px; flex-wrap:wrap; }
.vr-header h1 { margin:0; font-size:1.7rem; letter-spacing:-0.02em; }
.vr-title { display:flex; align-items:center; gap:10px; }
.vr-header p { margin:2px 0 0 0; opacity:.7; font-size:.95rem; }
.vr-badge { padding:5px 11px; border-radius:999px; font-size:.82rem;
  border:1px solid var(--border-color-primary); white-space:nowrap; }
.vr-badge.gpu { background: color-mix(in srgb, var(--color-accent) 12%, transparent); }
.vr-badge.cpu { background: color-mix(in srgb, #f59e0b 18%, transparent); }
.vr-card { border:1px solid var(--border-color-primary); border-radius:12px;
  padding:14px 16px; background: var(--block-background-fill); }
.vr-card h3 { margin:0 0 6px 0; font-size:1.05rem; }
.vr-muted { opacity:.7; font-size:.9rem; }
.vr-bar { height:8px; border-radius:999px; overflow:hidden; margin:10px 0 6px 0;
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
.vr-hint p { margin:0; font-size:.85rem; opacity:.75; }
.vr-page { max-width:880px; margin:0 auto; width:100%; }
.vr-heading p { margin:4px 0 0 0; font-size:1rem; font-weight:600; }

/* Inline help becomes a hover tooltip: the label gets a small marker and the
   text only shows while hovering it. */
.gradio-container .has-info { cursor:help; }
.gradio-container .has-info::after { content:" \\24D8"; opacity:.55; font-weight:400; }
.gradio-container *:has(> .info-text) { position:relative; }
.gradio-container .block:has(.info-text),
.gradio-container .form:has(.info-text) { overflow:visible !important; }
.gradio-container .info-text { display:none; position:absolute; left:0; top:100%;
  z-index:60; width:max-content; max-width:320px; padding:8px 10px; margin-top:4px;
  border-radius:8px; background: var(--background-fill-primary);
  border:1px solid var(--border-color-primary);
  box-shadow:0 8px 24px rgba(0,0,0,.25); font-size:.8rem; line-height:1.4;
  font-weight:400; color: var(--body-text-color); white-space:normal; }
.gradio-container .has-info:hover ~ .info-text,
.gradio-container .checkbox-container:hover ~ .info-text,
.gradio-container .info-text:hover { display:block; }
/* Checkboxes label themselves differently: mark and hover the whole label. */
.gradio-container .checkbox-container:has(~ .info-text) { cursor:help; }
.gradio-container .checkbox-container:has(~ .info-text) .label-text::after {
  content:" \\24D8"; opacity:.55; }
footer { display:none !important; }
"""
