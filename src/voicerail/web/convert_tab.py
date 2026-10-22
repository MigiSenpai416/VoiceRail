"""Convert tab: replace the voice in a song or a voice recording."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from ..audio.io import OUTPUT_FORMATS
from ..audio.separation import (
    DENOISE_MODELS,
    DEREVERB_MODELS,
    LEAD_MODELS,
    VOCAL_MODELS,
)
from ..convert import ConvertRequest, convert
from ..rvc.pipeline import ConversionSettings
from .common import (
    friendly_errors,
    model_card,
    model_choices,
    refreshed_models,
    reset_button,
)

SONG = "Song (vocals + music)"
VOICE = "Voice only (speech or a cappella)"
BACKING = {
    "Keep original backing vocals": "keep",
    "Convert all vocals together": "convert",
    "Remove backing vocals": "remove",
}


def build() -> dict:
    with gr.Column(elem_classes="vr-page"):
        source = gr.Audio(
            label="Audio to convert",
            type="filepath",
            sources=["upload", "microphone"],
        )
        mode = gr.Radio(
            [SONG, VOICE],
            value=SONG,
            label="What is it?",
            info="A song is split into vocals and music first; only the vocals are "
            "converted and everything is mixed back together.",
        )
        with gr.Row():
            choices = model_choices()
            model = gr.Dropdown(
                choices,
                value=choices[0] if choices else None,
                label="Voice",
                info="Models you trained or added in the Library tab.",
                scale=6,
            )
            refresh = gr.Button("↻", scale=0, min_width=44)
        card = gr.Markdown(
            model_card(choices[0] if choices else None), elem_classes="vr-hint"
        )

        with gr.Accordion("Options", open=False):
            with gr.Accordion("Pitch", open=False):
                with gr.Row():
                    auto_pitch = gr.Checkbox(
                        value=True,
                        label="Auto-match the voice's range",
                        info="Adds the shift that fits the model's natural pitch: "
                        "whole octaves for songs (to stay in key), any amount "
                        "for speech. Off = use the manual shift only.",
                    )
                    pitch = gr.Slider(
                        -24,
                        24,
                        value=0,
                        step=1,
                        label="Pitch shift (semitones)",
                        info="+12 = one octave up (male → female), −12 = one "
                        "down. Added on top of the automatic match.",
                    )
                with gr.Row():
                    autotune = gr.Slider(
                        0,
                        1,
                        value=0,
                        step=0.05,
                        label="Autotune",
                        info="Higher = snaps the melody harder onto exact notes "
                        "(robotic at 1); 0 = natural pitch.",
                    )
                    pitch_threshold = gr.Slider(
                        0.01,
                        0.2,
                        value=0.03,
                        step=0.01,
                        label="Pitch detection threshold",
                        info="Higher = quiet or breathy parts count as unvoiced "
                        "(fewer stray notes); lower = catches soft notes but "
                        "may add pitch noise.",
                    )
                refine_pitch = gr.Checkbox(
                    value=True,
                    label="Clean up pitch tracking",
                    info="Fixes octave jumps and tiny glitches in the detected "
                    "melody. Turn off only if a real leap gets smoothed away.",
                )
            with gr.Accordion("Voice", open=False):
                with gr.Row():
                    index_rate = gr.Slider(
                        0,
                        1,
                        value=0.5,
                        step=0.05,
                        label="Timbre retrieval (index rate)",
                        info="Higher = more of the trained voice's timbre, can "
                        "slur words; lower = clearer pronunciation, less "
                        "character. Needs an index file.",
                    )
                    protect = gr.Slider(
                        0,
                        0.5,
                        value=0.33,
                        step=0.01,
                        label="Consonant protection",
                        info="Lower = cleaner breaths and 's' sounds but less of "
                        "the model's voice on them; 0.5 = off.",
                    )
                rms_mix = gr.Slider(
                    0,
                    1,
                    value=0.25,
                    step=0.05,
                    label="Volume envelope",
                    info="Lower = follows the input's loudness (more dynamics); "
                    "higher = keeps the model's own levels (more even).",
                )
            with gr.Accordion("Separation & clean-up", open=False):
                with gr.Group(visible=True) as song_group:
                    backing = gr.Radio(
                        list(BACKING),
                        value="Keep original backing vocals",
                        label="Backing vocals / harmonies",
                    )
                    with gr.Row():
                        vocal_model = gr.Dropdown(
                            [m.label for m in VOCAL_MODELS],
                            value=VOCAL_MODELS[0].label,
                            label="Vocal separation model",
                        )
                        lead_model = gr.Dropdown(
                            [m.label for m in LEAD_MODELS],
                            value=LEAD_MODELS[0].label,
                            label="Lead / backing separation model",
                        )
                with gr.Group(visible=False) as voice_group, gr.Row():
                    denoise = gr.Checkbox(
                        value=False,
                        label="Remove background noise",
                        info="For phone, video or street recordings.",
                    )
                    denoise_model = gr.Dropdown(
                        [m.label for m in DENOISE_MODELS],
                        value=DENOISE_MODELS[0].label,
                        label="Denoise model",
                    )
                with gr.Row():
                    dereverb = gr.Checkbox(
                        value=True,
                        label="Remove reverb / echo before converting",
                        info="Dry vocals convert much more cleanly.",
                    )
                    dereverb_model = gr.Dropdown(
                        [m.label for m in DEREVERB_MODELS],
                        value=DEREVERB_MODELS[0].label,
                        label="De-reverb model",
                    )
            with gr.Accordion("Mix & output", open=False):
                with gr.Row():
                    vocal_gain = gr.Slider(
                        -12, 12, value=0, step=0.5, label="Vocals (dB)"
                    )
                    inst_gain = gr.Slider(
                        -12, 12, value=0, step=0.5, label="Music (dB)"
                    )
                    backing_gain = gr.Slider(
                        -12, 12, value=0, step=0.5, label="Backing vocals (dB)"
                    )
                with gr.Row():
                    reverb = gr.Slider(
                        0,
                        1,
                        value=0.15,
                        step=0.05,
                        label="Vocal reverb",
                        info="Higher = more room around the vocal after de-reverb; "
                        "0 = completely dry.",
                    )
                    fmt = gr.Radio(
                        list(OUTPUT_FORMATS), value="wav", label="Output format"
                    )
                shift_music = gr.Checkbox(
                    value=False,
                    label="Shift the music by the same amount",
                    info="For non-octave shifts, so the song stays in tune with "
                    "the vocals.",
                )

        options = [
            auto_pitch,
            pitch,
            autotune,
            pitch_threshold,
            refine_pitch,
            index_rate,
            protect,
            rms_mix,
            backing,
            vocal_model,
            lead_model,
            denoise,
            denoise_model,
            dereverb,
            dereverb_model,
            vocal_gain,
            inst_gain,
            backing_gain,
            reverb,
            fmt,
            shift_music,
        ]
        with gr.Row():
            reset_button("Reset options", options)
            run = gr.Button(
                "Convert", variant="primary", scale=4, elem_classes="vr-big-button"
            )

        result = gr.Audio(
            label="Result", type="filepath", interactive=False, buttons=["download"]
        )
        notes = gr.Markdown(elem_classes="vr-hint")
        with gr.Accordion("Stems", open=False):
            with gr.Row():
                vocals_out = gr.Audio(
                    label="Converted vocals",
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                )
                inst_out = gr.Audio(
                    label="Music",
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                )
            with gr.Row():
                source_vocals_out = gr.Audio(
                    label="Original vocals",
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                )
                backing_out = gr.Audio(
                    label="Backing vocals",
                    type="filepath",
                    interactive=False,
                    buttons=["download"],
                )

    def toggle_mode(value: str):
        song = value == SONG
        return gr.Group(visible=song), gr.Group(visible=not song)

    mode.change(toggle_mode, mode, [song_group, voice_group])
    refresh.click(refreshed_models, model, model).then(model_card, model, card)
    model.change(model_card, model, card)

    @friendly_errors
    def do_convert(
        audio_path,
        mode_value,
        model_name,
        auto,
        pitch_value,
        autotune_value,
        threshold_value,
        refine,
        index_value,
        protect_value,
        rms_value,
        backing_value,
        vocal_model_value,
        lead_value,
        denoise_value,
        denoise_model_value,
        dereverb_value,
        dereverb_model_value,
        vocal_db,
        inst_db,
        backing_db,
        reverb_value,
        fmt_value,
        shift_music_value,
        progress=gr.Progress(),
    ):
        if not audio_path:
            raise gr.Error("Upload or record some audio first.")
        if not model_name:
            raise gr.Error("Select a voice (train or add one in the Library tab).")
        request = ConvertRequest(
            input_path=Path(audio_path),
            model_name=model_name,
            mode="song" if mode_value == SONG else "voice",
            settings=ConversionSettings(
                pitch_shift=float(pitch_value),
                index_rate=float(index_value),
                protect=float(protect_value),
                rms_mix_rate=float(rms_value),
                refine_pitch=bool(refine),
                autotune_strength=float(autotune_value),
                pitch_threshold=float(threshold_value),
            ),
            auto_pitch=bool(auto),
            vocal_model=vocal_model_value,
            backing_vocals=BACKING[backing_value],
            lead_model=lead_value,
            dereverb=bool(dereverb_value),
            dereverb_model=dereverb_model_value,
            denoise=bool(denoise_value),
            denoise_model=denoise_model_value,
            vocal_gain_db=float(vocal_db),
            instrumental_gain_db=float(inst_db),
            backing_gain_db=float(backing_db),
            reverb=float(reverb_value),
            shift_instrumental=bool(shift_music_value),
            output_format=fmt_value,
        )
        res = convert(request, progress=lambda f, m: progress(f, desc=m))
        lines = [f"✅ Saved to `{res.output}`"]
        if res.pitch_shift:
            lines.append(f"Pitch shift used: {res.pitch_shift:+g} semitones")
        lines += res.notes
        return (
            str(res.output),
            str(res.converted_vocals),
            str(res.instrumental) if res.instrumental else None,
            str(res.source_vocals) if res.source_vocals else None,
            str(res.backing) if res.backing else None,
            "\n\n".join(lines),
        )

    run.click(
        do_convert,
        [source, mode, model, *options],
        [result, vocals_out, inst_out, source_vocals_out, backing_out, notes],
        concurrency_limit=1,
        concurrency_id="gpu",
    )
    return {"model": model, "card": card}
