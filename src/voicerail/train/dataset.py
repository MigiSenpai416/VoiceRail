"""Turn raw recordings into clean, uniformly levelled training segments.

Per source file:

1. decode (any format ffmpeg understands),
2. optionally isolate the voice: remove music, backing vocals, reverb, noise,
3. downmix, resample, high-pass (rumble) and loudness-normalise,
4. slice at silences into <= 3.7 s segments, dropping unusable ones,
5. write each segment at the training sample rate and at 16 kHz.

Cleaned stems are cached per file and option set, so re-running with other
segmenting options does not separate everything again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..audio import dsp
from ..audio.io import AUDIO_EXTENSIONS, load_audio, resample, save_audio
from ..audio.separation import (
    DENOISE_MODELS,
    DEREVERB_MODELS,
    LEAD_MODELS,
    SEPARATION_SR,
    VOCAL_MODELS,
    shared,
)
from ..audio.slicer import Slicer, split_long
from ..paths import replace_file
from .progress import Reporter, StopRequested

log = logging.getLogger(__name__)

SR_16K = 16000
MAX_SEPARATION_CHUNK_S = 600
OVERLAP_S = 0.3


@dataclass
class CleaningOptions:
    isolate_vocals: bool = True
    vocal_model: str = VOCAL_MODELS[0].label
    remove_backing_vocals: bool = False
    lead_model: str = LEAD_MODELS[0].label
    dereverb: bool = True
    dereverb_model: str = DEREVERB_MODELS[0].label
    denoise: bool = False
    denoise_model: str = DENOISE_MODELS[0].label

    def chain(self) -> list[tuple[str, str]]:
        steps = []
        if self.isolate_vocals:
            steps.append(("Isolating vocals", self.vocal_model))
        if self.remove_backing_vocals:
            steps.append(("Removing backing vocals", self.lead_model))
        if self.dereverb:
            steps.append(("Removing reverb", self.dereverb_model))
        if self.denoise:
            steps.append(("Removing noise", self.denoise_model))
        return steps


@dataclass
class SegmentOptions:
    sample_rate: int = 40000
    target_lufs: float = -20.0
    highpass_hz: float = 50.0
    silence_db: float = -42.0
    max_seconds: float = 3.7
    min_seconds: float = 1.0


@dataclass
class FileReport:
    name: str
    seconds: float
    kept_seconds: float = 0.0
    segments: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class DatasetReport:
    sample_rate: int
    files: list[FileReport] = field(default_factory=list)
    total_segments: int = 0
    total_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> DatasetReport:
        files = [FileReport(**f) for f in data.get("files", [])]
        return cls(
            sample_rate=data["sample_rate"],
            files=files,
            total_segments=data.get("total_segments", 0),
            total_seconds=data.get("total_seconds", 0.0),
            warnings=data.get("warnings", []),
        )


def dataset_dir(exp_dir: Path) -> Path:
    return exp_dir / "dataset"


def find_audio_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
            )
        elif path.is_file():
            files.append(path)
    return files


def _file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _clean_file(
    source: Path,
    file_hash: str,
    options: CleaningOptions,
    cache: Path,
    reporter: Reporter,
    base_progress: float,
    span: float,
) -> Path:
    """Run the separation chain on one file; cached by content and options."""
    chain = options.chain()
    key = hashlib.sha1(json.dumps(chain).encode()).hexdigest()[:8]
    cleaned = cache / f"{file_hash}_{key}.wav"
    if cleaned.is_file():
        return cleaned

    work = cache / f"work_{file_hash}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        audio, sr = load_audio(source, sr=SEPARATION_SR, mono=False)
        # Leave headroom: the separator rescales any input louder than 0 dBFS.
        audio = audio * min(1.0, 0.9 / max(dsp.peak(audio), 1e-9))
        chunk = MAX_SEPARATION_CHUNK_S * sr
        pieces = [audio[:, i : i + chunk] for i in range(0, audio.shape[1], chunk)]
        results = []
        for p_index, piece in enumerate(pieces):
            current = save_audio(work / f"in_{p_index}.wav", piece, sr)
            for s_index, (label, model) in enumerate(chain):
                fraction = (p_index * len(chain) + s_index) / (len(pieces) * len(chain))
                reporter.progress(
                    base_progress + span * fraction, f"{source.name}: {label.lower()}"
                )
                reporter.check_stop()
                current, _ = shared().separate(
                    current, model, work, f"p{p_index}_s{s_index}"
                )
            stem, _ = load_audio(current, sr=SEPARATION_SR, mono=False)
            results.append(dsp.match_length(stem, piece.shape[1]))
        merged = np.concatenate(results, axis=1)
        tmp = save_audio(work / "merged.wav", merged, SEPARATION_SR)
        replace_file(tmp, cleaned)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return cleaned


def _level_segment(piece: np.ndarray) -> np.ndarray:
    """RVC's per-segment levelling: mostly peak-normalised to 0.9."""
    peak = dsp.peak(piece)
    if peak <= 0:
        return piece
    return (piece / peak * (0.9 * 0.75) + 0.25 * piece).astype(np.float32)


def prepare_dataset(
    sources: list[Path],
    exp_dir: Path,
    cleaning: CleaningOptions,
    segmenting: SegmentOptions,
    reporter: Reporter,
) -> DatasetReport:
    files = find_audio_files(sources)
    if not files:
        raise ValueError("No audio files were found in the training data.")

    out = dataset_dir(exp_dir)
    wav_dir, wav16_dir = out / "wavs", out / "wavs16k"
    for directory in (wav_dir, wav16_dir):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
    cache = exp_dir / "cache" / "clean"
    cache.mkdir(parents=True, exist_ok=True)

    sr = segmenting.sample_rate
    slicer = Slicer(sr, threshold_db=segmenting.silence_db)
    report = DatasetReport(sample_rate=sr)
    seen: dict[str, str] = {}
    span = 1.0 / len(files)

    for index, source in enumerate(files):
        reporter.check_stop()
        base = index * span
        reporter.progress(base, f"Preparing {source.name}")
        file_report = FileReport(name=source.name, seconds=0.0)
        report.files.append(file_report)
        try:
            digest = _file_hash(source)
            if digest in seen:
                file_report.error = f"duplicate of {seen[digest]}, skipped"
                continue
            seen[digest] = source.name

            raw, raw_sr = load_audio(source, mono=False)
            file_report.seconds = raw.shape[1] / raw_sr
            if file_report.seconds < 0.5:
                file_report.error = "too short"
                continue
            if dsp.clipped_fraction(raw) > 0.001:
                file_report.warnings.append(
                    "source is clipped/distorted; consider a cleaner copy"
                )
            del raw

            if cleaning.chain():
                path = _clean_file(
                    source, digest, cleaning, cache, reporter, base, span * 0.85
                )
            else:
                path = source
            audio, _ = load_audio(path, sr=sr, mono=True)
            audio = dsp.highpass(audio, sr, segmenting.highpass_hz)
            audio = dsp.normalize_loudness(audio, sr, segmenting.target_lufs, -1.0)
            reporter.progress(base + span * 0.9, f"{source.name}: slicing")

            n_segment = 0
            for clip in slicer.slice(audio):
                pieces = split_long(clip, sr, segmenting.max_seconds, OVERLAP_S)
                for p_index, piece in enumerate(pieces):
                    reason = _reject_reason(piece, sr, segmenting.min_seconds)
                    if reason:
                        file_report.dropped[reason] = (
                            file_report.dropped.get(reason, 0) + 1
                        )
                        continue
                    piece = _level_segment(piece)
                    name = f"{index:03d}_{n_segment:04d}"
                    save_audio(wav_dir / f"{name}.wav", piece, sr)
                    save_audio(
                        wav16_dir / f"{name}.wav", resample(piece, sr, SR_16K), SR_16K
                    )
                    n_segment += 1
                    # Consecutive windows overlap; count each second once.
                    overlap = OVERLAP_S if p_index > 0 else 0.0
                    file_report.kept_seconds += len(piece) / sr - overlap
            file_report.segments = n_segment
            if n_segment == 0:
                file_report.warnings.append("no usable voice was found in this file")
        except StopRequested:
            raise
        except Exception as error:
            log.exception("Failed to prepare %s", source)
            file_report.error = str(error)

    report.total_segments = sum(f.segments for f in report.files)
    report.total_seconds = sum(f.kept_seconds for f in report.files)
    report.warnings = _dataset_warnings(report)
    (out / "report.json").write_text(json.dumps(report.to_json(), indent=2))
    reporter.progress(1.0, "Dataset ready")
    return report


def _reject_reason(piece: np.ndarray, sr: int, min_seconds: float) -> str | None:
    if len(piece) < min_seconds * sr:
        return "too short"
    rms_db = dsp.db(float(np.sqrt(np.mean(piece**2))))
    if rms_db < -50:
        return "silence"
    return None


def _dataset_warnings(report: DatasetReport) -> list[str]:
    minutes = report.total_seconds / 60
    warnings = []
    if report.total_segments == 0:
        warnings.append("No usable audio remained after cleaning.")
    elif minutes < 3:
        warnings.append(
            f"Only {minutes:.1f} min of clean voice. 10+ minutes gives much better "
            "results; expect a less stable voice."
        )
    elif minutes < 10:
        warnings.append(
            f"{minutes:.1f} min of clean voice: usable, but 10-30 minutes is ideal."
        )
    elif minutes > 90:
        warnings.append(
            f"{minutes:.0f} min of audio: training will take long; 30-60 minutes of "
            "varied, clean material is usually enough."
        )
    failed = [f.name for f in report.files if f.error and "duplicate" not in f.error]
    if failed:
        warnings.append(f"{len(failed)} file(s) could not be used: {', '.join(failed)}")
    return warnings


def load_report(exp_dir: Path) -> DatasetReport | None:
    path = dataset_dir(exp_dir) / "report.json"
    if not path.is_file():
        return None
    return DatasetReport.from_json(json.loads(path.read_text()))
