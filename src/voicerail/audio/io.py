"""Audio decoding, encoding and ffmpeg discovery."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
    ".alac",
    ".webm",
    ".mp4",
    ".mkv",
    ".mov",
    ".m4v",
}
OUTPUT_FORMATS = ("wav", "flac", "mp3")


class AudioError(RuntimeError):
    pass


@cache
def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary, preferring the one on ``PATH``."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as error:
        raise AudioError(
            "ffmpeg was not found. Install ffmpeg or reinstall VoiceRail."
        ) from error


_STREAM_RATE = re.compile(r"Stream #\d+:\d+.*?Audio:.*?(\d+) Hz")


def _decode_ffmpeg(path: Path, channels: int) -> tuple[np.ndarray, int]:
    """Decode at the native sample rate (resampling is done with soxr)."""
    cmd = [
        ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-v",
        "info",
        "-i",
        str(path),
        "-vn",
        "-map",
        "0:a:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    stderr = proc.stderr.decode(errors="replace")
    if proc.returncode != 0 or not proc.stdout:
        lines = [line for line in stderr.strip().splitlines() if line.strip()]
        reason = lines[-1] if lines else "no audio stream"
        raise AudioError(f"Could not decode '{path.name}': {reason}")
    match = _STREAM_RATE.search(stderr)
    if not match:
        raise AudioError(f"Could not determine the sample rate of '{path.name}'.")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio.reshape(-1, channels).T.copy(), int(match.group(1))


def load_audio(
    path: str | Path, sr: int | None = None, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Decode any audio/video file to float32.

    Returns ``[samples]`` when ``mono`` else ``[2, samples]``, resampled to
    ``sr`` when given.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioError(f"File not found: {path}")
    channels = 1 if mono else 2
    try:
        data, file_sr = sf.read(path, dtype="float32", always_2d=True)
        audio = data.T
        if audio.shape[1] == 0:
            raise AudioError(f"'{path.name}' contains no audio.")
    except AudioError:
        raise
    except Exception:
        audio, file_sr = _decode_ffmpeg(path, channels)

    if audio.shape[0] != channels:
        if channels == 1:
            audio = audio.mean(axis=0, keepdims=True)
        elif audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        else:
            audio = audio[:2]
    if sr and file_sr != sr:
        audio = resample(audio, file_sr, sr)
        file_sr = sr
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    out = audio[0] if mono else audio
    return np.ascontiguousarray(out, dtype=np.float32), int(file_sr)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High quality resampling along the last axis."""
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 1:
        return soxr.resample(audio, sr_in, sr_out, quality="VHQ").astype(np.float32)
    out = soxr.resample(audio.T, sr_in, sr_out, quality="VHQ")
    return np.ascontiguousarray(out.T, dtype=np.float32)


def save_audio(
    path: str | Path, audio: np.ndarray, sr: int, fmt: str | None = None
) -> Path:
    """Write ``[samples]`` or ``[channels, samples]`` audio."""
    path = Path(path)
    suffix = path.suffix.lstrip(".").lower()
    if fmt is None:
        fmt = suffix if suffix in OUTPUT_FORMATS else "wav"
    fmt = fmt.lower()
    if suffix != fmt:
        # Append rather than replace: names like "01. Intro" contain dots.
        path = path.with_name(f"{path.name}.{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.T if audio.ndim == 2 else audio
    data = np.ascontiguousarray(data, dtype=np.float32)
    if fmt == "wav":
        sf.write(path, data, sr, subtype="FLOAT")
    elif fmt == "flac":
        sf.write(path, np.clip(data, -1.0, 1.0), sr, subtype="PCM_24")
    elif fmt == "mp3":
        _encode_ffmpeg(path, np.clip(data, -1.0, 1.0), sr, ["-b:a", "320k"])
    else:
        raise AudioError(f"Unsupported output format: {fmt}")
    return path


def _encode_ffmpeg(path: Path, data: np.ndarray, sr: int, codec_args: list[str]):
    channels = 1 if data.ndim == 1 else data.shape[1]
    cmd = [
        ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        "-i",
        "-",
        *codec_args,
        str(path),
    ]
    proc = subprocess.run(cmd, input=data.tobytes(), capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioError(
            f"Could not encode '{path.name}': {proc.stderr.decode(errors='replace')}"
        )


def duration_seconds(path: str | Path) -> float:
    try:
        return float(sf.info(path).duration)
    except Exception:
        audio, sr = load_audio(path, sr=16000)
        return len(audio) / sr
