"""Silence-based slicing of long recordings into training segments.

Behaviour follows the widely used openvpi audio-slicer (also used by RVC):
cut at the quietest point of every sufficiently long silence and keep a short
margin of silence around each clip.
"""

from __future__ import annotations

import itertools

import numpy as np

from .dsp import frame_rms


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold_db: float = -42.0,
        min_length_ms: int = 1500,
        min_interval_ms: int = 400,
        hop_ms: int = 15,
        max_silence_kept_ms: int = 500,
    ):
        if not min_length_ms >= min_interval_ms >= hop_ms:
            raise ValueError("min_length >= min_interval >= hop is required")
        if max_silence_kept_ms < hop_ms:
            raise ValueError("max_silence_kept >= hop is required")
        min_interval = sr * min_interval_ms / 1000
        self.threshold = 10 ** (threshold_db / 20)
        self.hop = round(sr * hop_ms / 1000)
        self.win = min(round(min_interval), 4 * self.hop)
        self.min_length = round(sr * min_length_ms / 1000 / self.hop)
        self.min_interval = round(min_interval / self.hop)
        self.max_sil_kept = round(sr * max_silence_kept_ms / 1000 / self.hop)

    def _cut(self, audio: np.ndarray, begin: int, end: int) -> np.ndarray:
        return audio[begin * self.hop : min(len(audio), end * self.hop)]

    def slice(self, audio: np.ndarray) -> list[np.ndarray]:
        if (len(audio) + self.hop - 1) // self.hop <= self.min_length:
            return [audio]
        rms = frame_rms(audio, self.win, self.hop)
        tags: list[tuple[int, int]] = []
        silence_start: int | None = None
        clip_start = 0
        for i, level in enumerate(rms):
            if level < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            leading = silence_start == 0 and i > self.max_sil_kept
            long_enough = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not leading and not long_enough:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = int(rms[silence_start : i + 1].argmin()) + silence_start
                tags.append((0, pos) if silence_start == 0 else (pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                lo = i - self.max_sil_kept
                pos = int(rms[lo : silence_start + self.max_sil_kept + 1].argmin()) + lo
                pos_l = (
                    int(
                        rms[
                            silence_start : silence_start + self.max_sil_kept + 1
                        ].argmin()
                    )
                    + silence_start
                )
                pos_r = int(rms[lo : i + 1].argmin()) + lo
                if silence_start == 0:
                    tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    int(
                        rms[
                            silence_start : silence_start + self.max_sil_kept + 1
                        ].argmin()
                    )
                    + silence_start
                )
                lo = i - self.max_sil_kept
                pos_r = int(rms[lo : i + 1].argmin()) + lo
                tags.append((0, pos_r) if silence_start == 0 else (pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        total = len(rms)
        if silence_start is not None and total - silence_start >= self.min_interval:
            end = min(total, silence_start + self.max_sil_kept)
            pos = int(rms[silence_start : end + 1].argmin()) + silence_start
            tags.append((pos, total + 1))

        if not tags:
            return [audio]
        chunks = []
        if tags[0][0] > 0:
            chunks.append(self._cut(audio, 0, tags[0][0]))
        for (_, start), (stop, _) in itertools.pairwise(tags):
            chunks.append(self._cut(audio, start, stop))
        if tags[-1][1] < total:
            chunks.append(self._cut(audio, tags[-1][1], total))
        return [c for c in chunks if len(c)]


def split_long(
    audio: np.ndarray, sr: int, max_seconds: float = 3.7, overlap: float = 0.3
) -> list[np.ndarray]:
    """Cut a clip into windows of at most ``max_seconds`` (RVC style)."""
    step = int(sr * (max_seconds - overlap))
    size = int(sr * max_seconds)
    if len(audio) <= int(sr * (max_seconds + overlap)):
        return [audio]
    pieces = []
    start = 0
    while True:
        if len(audio) - start > int(sr * (max_seconds + overlap)):
            pieces.append(audio[start : start + size])
            start += step
        else:
            pieces.append(audio[start:])
            break
    return pieces
