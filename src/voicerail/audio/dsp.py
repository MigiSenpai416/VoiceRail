"""Signal processing helpers (filters, levels, quality metrics)."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import signal


def highpass(audio: np.ndarray, sr: int, cutoff: float = 48.0, order: int = 5):
    """Zero-phase Butterworth high-pass along the last axis."""
    sos = signal.butter(order, cutoff, btype="highpass", fs=sr, output="sos")
    if audio.shape[-1] <= 3 * (2 * len(sos) + 1):
        return audio.astype(np.float32, copy=True)
    return signal.sosfiltfilt(sos, audio, axis=-1).astype(np.float32)


def peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-12)))


def frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """RMS of each frame of a mono signal (centred frames, zero padded)."""
    padded = np.pad(audio.astype(np.float64), (frame // 2, frame // 2))
    n_frames = 1 + max(0, len(padded) - frame) // hop
    energy = np.concatenate([[0.0], np.cumsum(padded**2)])
    starts = np.arange(n_frames) * hop
    sums = energy[np.minimum(starts + frame, len(padded))] - energy[starts]
    return np.sqrt(np.maximum(sums, 0.0) / frame + 1e-12)


def loudness_lufs(audio: np.ndarray, sr: int) -> float | None:
    """Integrated loudness (ITU-R BS.1770) or ``None`` if too short/silent."""
    import pyloudnorm

    data = audio.T if audio.ndim == 2 else audio
    if data.shape[0] < int(0.5 * sr):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = pyloudnorm.Meter(sr).integrated_loudness(data)
    return float(value) if np.isfinite(value) else None


def normalize_loudness(
    audio: np.ndarray, sr: int, target_lufs: float = -20.0, peak_db: float = -1.0
) -> np.ndarray:
    """Gain to ``target_lufs`` without letting the peak exceed ``peak_db``."""
    current = loudness_lufs(audio, sr)
    gain = 1.0 if current is None else 10 ** ((target_lufs - current) / 20)
    max_gain = 10 ** (peak_db / 20) / max(peak(audio), 1e-9)
    return (audio * min(gain, max_gain)).astype(np.float32)


def peak_normalize(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    current = peak(audio)
    if current <= 0:
        return audio
    return (audio * (target / current)).astype(np.float32)


def prevent_clipping(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    current = peak(audio)
    if current > ceiling:
        audio = audio * (ceiling / current)
    return audio.astype(np.float32, copy=False)


def clipped_fraction(audio: np.ndarray, threshold: float = 0.999) -> float:
    """Share of samples sitting at (or very near) digital full scale."""
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold * max(peak(audio), 1e-9)))


def noise_floor_db(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """(quiet floor, loud level) in dBFS from 20 ms frame energies."""
    frame = int(0.02 * sr)
    rms = frame_rms(audio, frame, frame)
    floor = float(np.percentile(rms, 10))
    loud = float(np.percentile(rms, 95))
    return db(floor), db(loud)


def match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[-1] >= length:
        return audio[..., :length]
    pad = [(0, 0)] * (audio.ndim - 1) + [(0, length - audio.shape[-1])]
    return np.pad(audio, pad)


def crossfade_concat(chunks: list[np.ndarray], overlap: int) -> np.ndarray:
    """Join chunks that overlap by ``overlap`` samples.

    The fades sum to one, which is right for overlapping renders of the same
    material (correlated signals).
    """
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    out = chunks[0].astype(np.float32)
    for chunk in chunks[1:]:
        n = min(overlap, len(out), len(chunk))
        if n <= 0:
            out = np.concatenate([out, chunk])
            continue
        t = np.linspace(0, np.pi / 2, n, dtype=np.float32)
        blended = out[-n:] * np.cos(t) ** 2 + chunk[:n] * np.sin(t) ** 2
        out = np.concatenate([out[:-n], blended, chunk[n:]])
    return out.astype(np.float32, copy=False)
