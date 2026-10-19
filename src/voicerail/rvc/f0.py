"""Pitch (F0) helpers: estimation clean-up, quantisation and transposition."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

F0_MIN = 50.0
F0_MAX = 1100.0
_MEL_MIN = 1127.0 * np.log(1.0 + F0_MIN / 700.0)
_MEL_MAX = 1127.0 * np.log(1.0 + F0_MAX / 700.0)


def coarse_f0(f0: np.ndarray) -> np.ndarray:
    """Quantise F0 (Hz) to RVC's 1..255 mel-scale pitch tokens (1 = unvoiced)."""
    f0_mel = 1127.0 * np.log(1.0 + np.maximum(f0, 0.0) / 700.0)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - _MEL_MIN) * 254 / (_MEL_MAX - _MEL_MIN) + 1
    f0_mel = np.clip(f0_mel, 1, 255)
    return np.rint(f0_mel).astype(np.int64)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop indices of consecutive ``True`` runs."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2], strict=True))


def refine_f0(
    f0: np.ndarray,
    *,
    min_voiced_frames: int = 3,
    max_gap_frames: int = 2,
    fix_octave_errors: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    """Remove common pitch tracker glitches without flattening expression.

    * voiced blips shorter than ``min_voiced_frames`` become unvoiced,
    * unvoiced gaps up to ``max_gap_frames`` inside a phrase are bridged,
    * short isolated octave jumps are folded back to the surrounding pitch,
    * a 3-frame median removes single frame jitter (vibrato is preserved).
    """
    f0 = np.asarray(f0, dtype=np.float64).copy()
    f0[(f0 < F0_MIN * 0.8) | ~np.isfinite(f0)] = 0.0
    voiced = f0 > 0
    if not voiced.any():
        return f0

    for start, stop in _runs(voiced):
        if stop - start < min_voiced_frames:
            f0[start:stop] = 0.0
    voiced = f0 > 0

    for start, stop in _runs(~voiced):
        if start > 0 and stop < len(f0) and stop - start <= max_gap_frames:
            left, right = np.log(f0[start - 1]), np.log(f0[stop])
            steps = np.arange(1, stop - start + 1) / (stop - start + 1)
            f0[start:stop] = np.exp(left + (right - left) * steps)
    voiced = f0 > 0

    if fix_octave_errors and voiced.sum() > 25:
        cents = np.zeros_like(f0)
        cents[voiced] = 1200 * np.log2(f0[voiced])
        # Median of the voiced neighbourhood: hold the last voiced value through
        # unvoiced frames so silence does not drag the reference down.
        held = cents.copy()
        last = cents[np.argmax(voiced)]
        for i in range(len(held)):
            if voiced[i]:
                last = held[i]
            else:
                held[i] = last
        reference = median_filter(held, size=25, mode="nearest")
        deviation = cents - reference
        outlier = voiced & (np.abs(deviation) > 800)
        for start, stop in _runs(outlier):
            if stop - start <= 5:
                octaves = np.round(deviation[start:stop] / 1200)
                f0[start:stop] = f0[start:stop] / 2.0**octaves

    if smooth:
        for start, stop in _runs(f0 > 0):
            if stop - start >= 3:
                f0[start:stop] = median_filter(f0[start:stop], size=3, mode="nearest")
    return f0


def shift_f0(f0: np.ndarray, semitones: float) -> np.ndarray:
    return f0 * 2.0 ** (semitones / 12.0)


def voiced_median(f0: np.ndarray) -> float | None:
    voiced = f0[f0 > 0]
    if voiced.size < 10:
        return None
    return float(np.exp(np.median(np.log(voiced))))


def suggest_shift(
    source_f0: np.ndarray,
    target_median_hz: float | None,
    *,
    octaves_only: bool,
    limit: int = 24,
) -> int:
    """Semitones that move the source's typical pitch to the target voice.

    For music, ``octaves_only`` keeps the song in key (shifts of 0 or ±12).
    """
    source = voiced_median(source_f0)
    if not source or not target_median_hz:
        return 0
    semitones = 12 * np.log2(target_median_hz / source)
    if octaves_only:
        shift = 12 * int(np.round(semitones / 12))
    else:
        shift = int(np.round(semitones))
    return int(np.clip(shift, -limit, limit))


def autotune(f0: np.ndarray, strength: float) -> np.ndarray:
    """Pull every voiced frame towards the nearest equal-tempered note."""
    if strength <= 0:
        return f0
    out = f0.copy()
    voiced = out > 0
    midi = 69 + 12 * np.log2(out[voiced] / 440.0)
    target = np.round(midi)
    out[voiced] = 440.0 * 2 ** ((midi + (target - midi) * strength - 69) / 12)
    return out
