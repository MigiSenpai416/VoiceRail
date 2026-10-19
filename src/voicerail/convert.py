"""Full conversion workflow: separate, convert, post-process and remix."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import library, paths
from .audio import dsp
from .audio.io import load_audio, resample, save_audio
from .audio.separation import (
    DENOISE_MODELS,
    DEREVERB_MODELS,
    LEAD_MODELS,
    SEPARATION_SR,
    VOCAL_MODELS,
    shared,
)
from .rvc import f0 as f0_utils
from .rvc.pipeline import ConversionSettings, VoiceConverter

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]
MIX_SR = SEPARATION_SR


@dataclass
class ConvertRequest:
    input_path: Path
    model_name: str
    mode: str = "song"  # "song" (music + vocals) or "voice" (voice only)
    settings: ConversionSettings = field(default_factory=ConversionSettings)
    auto_pitch: bool = False
    vocal_model: str = VOCAL_MODELS[0].label
    backing_vocals: str = "keep"  # keep | convert | remove
    lead_model: str = LEAD_MODELS[0].label
    dereverb: bool = True
    dereverb_model: str = DEREVERB_MODELS[0].label
    denoise: bool = False
    denoise_model: str = DENOISE_MODELS[0].label
    vocal_gain_db: float = 0.0
    instrumental_gain_db: float = 0.0
    backing_gain_db: float = 0.0
    reverb: float = 0.15
    shift_instrumental: bool = False
    output_format: str = "wav"


@dataclass
class ConvertResult:
    output: Path
    converted_vocals: Path
    instrumental: Path | None
    backing: Path | None
    source_vocals: Path | None
    pitch_shift: float
    notes: list[str]


class _ConverterCache:
    def __init__(self):
        self._key = None
        self._converter: VoiceConverter | None = None
        self._lock = threading.Lock()

    def get(self, model: Path, index: Path | None) -> VoiceConverter:
        key = (str(model), model.stat().st_mtime, str(index))
        with self._lock:
            if self._key != key:
                self._converter = None
                self._converter = VoiceConverter(model, index)
                self._key = key
            return self._converter


_converters = _ConverterCache()
_lock = threading.Lock()


def _digest(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _db(gain_db: float) -> float:
    return float(10 ** (gain_db / 20))


class _Steps:
    """Maps sub-task progress onto one overall progress bar."""

    def __init__(self, progress: ProgressFn | None, total: int):
        self.progress = progress
        self.total = max(total, 1)
        self.index = 0

    def start(self, message: str) -> ProgressFn:
        base = self.index / self.total
        self.index += 1
        if self.progress:
            self.progress(base, message)

        def sub(fraction: float, text: str) -> None:
            if self.progress:
                self.progress(base + fraction / self.total, text)

        return sub


def convert(
    request: ConvertRequest, progress: ProgressFn | None = None
) -> ConvertResult:
    with _lock:
        return _convert(request, progress)


def _convert(request: ConvertRequest, progress: ProgressFn | None) -> ConvertResult:
    model_path, index_path = library.model_paths(request.model_name)
    source = Path(request.input_path)
    work = paths.cache_dir() / "convert" / _digest(source)
    work.mkdir(parents=True, exist_ok=True)
    song = request.mode == "song"
    notes: list[str] = []

    planned = 3 + (1 if song else 0)
    planned += int(song and request.backing_vocals != "convert")
    planned += int(request.dereverb) + int(not song and request.denoise)
    steps = _Steps(progress, planned)

    # 1. Decode with headroom (the separator rescales anything above 0 dBFS).
    steps.start("Reading audio")
    mix, _ = load_audio(source, sr=MIX_SR, mono=False)
    if mix.shape[1] < MIX_SR // 4:
        raise ValueError("The input is too short (needs at least 0.25 s of audio).")
    headroom = min(1.0, 0.8 / max(dsp.peak(mix), 1e-9))
    mix = mix * headroom
    mix_path = save_audio(work / "input.wav", mix, MIX_SR)

    instrumental = backing = source_vocals = None
    voice_path = mix_path
    if song:
        steps.start("Separating vocals from music")
        voice_path, instrumental = _separate(
            mix_path, request.vocal_model, work, "vocals", "instrumental"
        )
        source_vocals = voice_path
        if request.backing_vocals != "convert":
            steps.start("Separating lead and backing vocals")
            voice_path, backing = _separate(
                voice_path, request.lead_model, work, "lead", "backing"
            )
    elif request.denoise:
        steps.start("Removing noise")
        voice_path, _ = _separate(
            voice_path, request.denoise_model, work, "denoised", "noise"
        )
    if request.dereverb:
        steps.start("Removing reverb")
        voice_path, _ = _separate(
            voice_path, request.dereverb_model, work, "dry", "reverb"
        )

    # 2. Voice conversion.
    convert_progress = steps.start("Converting voice")
    converter = _converters.get(model_path, index_path)
    voice, _ = load_audio(voice_path, sr=16000, mono=True)
    settings = ConversionSettings(**vars(request.settings))
    if request.auto_pitch and converter.model.use_f0:
        target = converter.target_median_f0
        if target:
            f0 = converter.analyse_pitch(voice, settings)
            auto = f0_utils.suggest_shift(f0, target, octaves_only=song)
            settings.pitch_shift += auto
            notes.append(f"Auto pitch: {auto:+d} semitones")
        else:
            notes.append(
                "Auto pitch is unavailable for this model (it has no pitch "
                "statistics); using the manual pitch shift."
            )
    converted = converter.convert(voice, settings, convert_progress)
    converted = resample(converted, converter.tgt_sr, MIX_SR)
    converted = dsp.match_length(converted, mix.shape[1])

    # 3. Post-processing and mix.
    steps.start("Mixing")
    vocals = _vocal_fx(converted, request.reverb) * _db(request.vocal_gain_db)
    vocals_stereo = np.stack([vocals, vocals])
    shift = settings.pitch_shift if converter.model.use_f0 else 0.0
    if not converter.model.use_f0 and (settings.pitch_shift or request.auto_pitch):
        notes.append("This model has no pitch guidance; pitch settings were ignored.")
    stems = {}
    if instrumental is not None:
        stems["instrumental"] = (instrumental, request.instrumental_gain_db)
    if backing is not None and request.backing_vocals == "keep":
        stems["backing"] = (backing, request.backing_gain_db)

    out = vocals_stereo.copy()
    for stem_path, gain in stems.values():
        stem, _ = load_audio(stem_path, sr=MIX_SR, mono=False)
        stem = dsp.match_length(stem, mix.shape[1])
        if request.shift_instrumental and shift % 12 != 0:
            stem = _pitch_shift(stem, shift)
        out = out + stem * _db(gain)
    if request.shift_instrumental and shift % 12 != 0 and stems:
        notes.append(f"Music shifted by {shift:+g} semitones to match the vocals.")

    out = _master(out / headroom)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    title = paths.safe_name(source.stem, "audio")[:60].rstrip(" ._")
    voice = paths.safe_name(request.model_name, "voice")[:40].rstrip(" ._")
    stem_name = f"{title} ({voice}) {stamp}"
    out_dir = paths.outputs_dir()
    output = save_audio(out_dir / stem_name, out, MIX_SR, request.output_format)
    if stems:
        vocals_file = save_audio(
            out_dir / f"{stem_name} [vocals]",
            _master(vocals_stereo / headroom),
            MIX_SR,
            request.output_format,
        )
    else:
        vocals_file = output
    if progress:
        progress(1.0, "Done")
    return ConvertResult(
        output=output,
        converted_vocals=vocals_file,
        instrumental=instrumental,
        backing=backing if request.backing_vocals == "keep" else None,
        source_vocals=source_vocals,
        pitch_shift=shift,
        notes=notes,
    )


def _separate(
    path: Path, model: str, work: Path, kept_name: str, removed_name: str
) -> tuple[Path, Path]:
    """Cached separation: results are keyed by input file and model."""
    tag = hashlib.sha1(f"{path.name}|{model}".encode()).hexdigest()[:8]
    kept = work / f"{kept_name}_{tag}.wav"
    removed = work / f"{removed_name}_{tag}.wav"
    if kept.is_file() and removed.is_file():
        return kept, removed
    a, b = shared().separate(path, model, work, f"tmp_{tag}")
    paths.replace_file(a, kept)
    paths.replace_file(b, removed)
    return kept, removed


def _vocal_fx(vocals: np.ndarray, reverb: float) -> np.ndarray:
    if reverb <= 0:
        return vocals
    from pedalboard import Pedalboard, Reverb

    board = Pedalboard(
        [
            Reverb(
                room_size=0.35,
                damping=0.6,
                wet_level=0.35 * reverb,
                dry_level=1.0,
                width=1.0,
            )
        ]
    )
    return board(vocals.astype(np.float32), MIX_SR)


def _pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    from pedalboard import Pedalboard, PitchShift

    return Pedalboard([PitchShift(semitones=semitones)])(
        audio.astype(np.float32), MIX_SR
    )


def _master(audio: np.ndarray) -> np.ndarray:
    """Transparent peak control: limit only if the mix would clip."""
    if dsp.peak(audio) <= 0.98:
        return audio.astype(np.float32)
    from pedalboard import Limiter, Pedalboard

    limited = Pedalboard([Limiter(threshold_db=-1.0, release_ms=80)])(
        audio.astype(np.float32), MIX_SR
    )
    return np.clip(limited, -0.999, 0.999).astype(np.float32)
