"""Source separation (vocals, backing vocals, reverb, noise) via audio-separator."""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from .. import paths

log = logging.getLogger(__name__)

SEPARATION_SR = 44100


@dataclass(frozen=True)
class SeparationModel:
    label: str
    filename: str
    note: str = ""


VOCAL_MODELS = [
    SeparationModel(
        "Mel-RoFormer Kim (best)",
        "vocals_mel_band_roformer.ckpt",
        "Highest vocal quality; needs a GPU for reasonable speed.",
    ),
    SeparationModel(
        "Mel-RoFormer Kim FT (unwa)",
        "mel_band_roformer_kim_ft_unwa.ckpt",
        "Fine-tuned Kim model; slightly fuller vocals.",
    ),
    SeparationModel(
        "BS-RoFormer ViperX 1297",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "Strong all-rounder.",
    ),
    SeparationModel(
        "MDX-Net Voc FT (fast)",
        "UVR-MDX-NET-Voc_FT.onnx",
        "Much faster, lower quality. Good for CPU-only machines.",
    ),
]
LEAD_MODELS = [
    SeparationModel(
        "Mel-RoFormer Karaoke (aufr33/viperx)",
        "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    ),
    SeparationModel(
        "Mel-RoFormer Karaoke (becruily)",
        "mel_band_roformer_karaoke_becruily.ckpt",
    ),
]
DEREVERB_MODELS = [
    SeparationModel(
        "Mel-RoFormer De-Reverb (anvuew)",
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    ),
    SeparationModel(
        "Mel-RoFormer De-Reverb + Echo (Sucial v2)",
        "dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt",
        "Also removes delay/echo effects.",
    ),
]
DENOISE_MODELS = [
    SeparationModel(
        "Mel-RoFormer Denoise (aufr33)",
        "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    ),
]

_ALL = {
    m.label: m for m in VOCAL_MODELS + LEAD_MODELS + DEREVERB_MODELS + DENOISE_MODELS
}

_WANTED = {
    "vocals",
    "vocal",
    "lead vocals",
    "dry",
    "noreverb",
    "no reverb",
    "no echo",
    "no noise",
}
_UNWANTED = ("instrument", "other", "reverb", "echo", "noise", "karaoke")


def resolve(label_or_file: str) -> SeparationModel:
    if label_or_file in _ALL:
        return _ALL[label_or_file]
    for model in _ALL.values():
        if model.filename == label_or_file:
            return model
    return SeparationModel(label_or_file, label_or_file)


def _is_wanted(stem: str) -> bool:
    """Whether a stem is the one to keep (voice / dry signal)."""
    name = stem.lower().replace("_", " ").strip()
    if name in _WANTED:
        return True
    if name.startswith("no "):
        return False
    return not any(word in name for word in _UNWANTED)


class StemSeparator:
    """Keeps the last separation model loaded between calls."""

    def __init__(self):
        self._separator = None
        self._loaded: str | None = None
        self._lock = threading.Lock()

    def _get(self, output_dir: Path):
        if self._separator is None:
            from audio_separator.separator import Separator

            class _Separator(Separator):
                def check_ffmpeg_installed(self):
                    # Only needed for pydub output; VoiceRail writes with
                    # soundfile and feeds WAV input, so ffmpeg is optional.
                    if shutil.which("ffmpeg"):
                        super().check_ffmpeg_installed()

            self._separator = _Separator(
                log_level=logging.WARNING,
                model_file_dir=str(paths.separator_models_dir()),
                output_dir=str(output_dir),
                output_format="WAV",
                normalization_threshold=1.0,
                sample_rate=SEPARATION_SR,
                use_soundfile=True,
            )
        self._separator.output_dir = str(output_dir)
        instance = getattr(self._separator, "model_instance", None)
        if instance is not None:
            instance.output_dir = str(output_dir)
        return self._separator

    def separate(
        self, wav_path: Path, model: str, output_dir: Path, base_name: str
    ) -> tuple[Path, Path]:
        """Split ``wav_path`` into (kept, removed) stems written as float WAV.

        ``kept`` is vocals / lead vocals / dry signal depending on the model.
        """
        spec = resolve(model)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            separator = self._get(output_dir)
            if self._loaded != spec.filename:
                log.info("Loading separation model %s", spec.filename)
                separator.load_model(spec.filename)
                self._loaded = spec.filename
                separator.model_instance.output_dir = str(output_dir)
            instance = separator.model_instance
            primary = instance.primary_stem_name
            secondary = instance.secondary_stem_name
            names = {primary: f"{base_name}_a", secondary: f"{base_name}_b"}
            files = separator.separate(str(wav_path), custom_output_names=names)

        produced = [Path(f) if Path(f).is_absolute() else output_dir / f for f in files]
        by_stem = {}
        for stem, name in names.items():
            match = [p for p in produced if p.stem == name]
            if not match:
                raise RuntimeError(
                    f"Separation with {spec.label} did not produce the '{stem}' stem."
                )
            by_stem[stem] = match[0]
        if _is_wanted(primary) or not _is_wanted(secondary):
            return by_stem[primary], by_stem[secondary]
        return by_stem[secondary], by_stem[primary]

    def unload(self) -> None:
        with self._lock:
            self._separator = None
            self._loaded = None


_shared: StemSeparator | None = None


def shared() -> StemSeparator:
    global _shared
    if _shared is None:
        _shared = StemSeparator()
    return _shared
