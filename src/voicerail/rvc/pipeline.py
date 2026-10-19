"""Voice conversion of a vocal/speech signal with an RVC model."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..audio import dsp
from ..device import default_device, empty_cache
from . import f0 as f0_utils
from . import runtime
from .checkpoint import VoiceModel, read_voice_model
from .index import FeatureIndex

log = logging.getLogger(__name__)

SR_16K = 16000
WINDOW = 160  # samples per pitch/feature frame at 16 kHz (100 frames/s)
X_PAD = 1  # seconds of context added on both sides of every chunk
X_QUERY = 6  # search radius (s) for a quiet split point
X_CENTER = 38  # nominal chunk length (s)
X_MAX = 41  # inputs shorter than this are converted in one pass


@dataclass
class ConversionSettings:
    pitch_shift: float = 0.0
    index_rate: float = 0.5
    protect: float = 0.33
    rms_mix_rate: float = 0.25
    refine_pitch: bool = True
    autotune_strength: float = 0.0
    speaker_id: int = 0
    pitch_threshold: float = 0.03


ProgressFn = Callable[[float, str], None]


class VoiceConverter:
    """Loaded voice model (+ optional index) ready to convert audio."""

    def __init__(
        self,
        model_path: str | Path,
        index_path: str | Path | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or default_device()
        self.model: VoiceModel = read_voice_model(model_path)
        self.net = self.model.build(self.device)
        self.tgt_sr = self.model.sample_rate
        self.index = None
        if index_path and Path(index_path).is_file():
            index = FeatureIndex(index_path)
            if index.dim == self.model.embedding_dim:
                self.index = index
            else:
                log.warning(
                    "Ignoring index %s: %d-dim features, model expects %d",
                    index_path,
                    index.dim,
                    self.model.embedding_dim,
                )

    @property
    def target_median_f0(self) -> float | None:
        extra = self.model.metadata.get("voicerail")
        if isinstance(extra, dict) and extra.get("f0_median"):
            return float(extra["f0_median"])
        return None

    def analyse_pitch(self, audio16k: np.ndarray, settings: ConversionSettings):
        audio = dsp.highpass(audio16k, SR_16K)
        return runtime.extract_f0(
            audio, self.device, settings.pitch_threshold, settings.refine_pitch
        )

    def convert(
        self,
        audio16k: np.ndarray,
        settings: ConversionSettings,
        progress: ProgressFn | None = None,
    ) -> np.ndarray:
        """Convert mono 16 kHz audio; returns audio at the model sample rate."""
        audio = dsp.highpass(audio16k.astype(np.float32), SR_16K)
        n = len(audio)
        if n < WINDOW * 4:
            return np.zeros(int(n * self.tgt_sr / SR_16K), dtype=np.float32)

        t_pad = SR_16K * X_PAD
        t_pad_tgt = self.tgt_sr * X_PAD
        splits = self._split_points(audio)
        mode = "reflect" if n > t_pad else "constant"
        audio_pad = np.pad(audio, (t_pad, t_pad), mode=mode)
        p_len = len(audio_pad) // WINDOW

        pitch = pitchf = None
        if self.model.use_f0:
            f0 = runtime.extract_f0(
                audio_pad, self.device, settings.pitch_threshold, settings.refine_pitch
            )
            f0 = dsp.match_length(f0, p_len)
            f0 = f0_utils.autotune(f0, settings.autotune_strength)
            f0 = f0_utils.shift_f0(f0, settings.pitch_shift)
            pitch = torch.from_numpy(f0_utils.coarse_f0(f0)).to(self.device)[None]
            pitchf = torch.from_numpy(f0.astype(np.float32)).to(self.device)[None]

        sid = torch.tensor(
            [min(settings.speaker_id, self.model.config.spk_embed_dim - 1)]
        )
        sid = sid.to(self.device)
        outputs = []
        start = 0
        bounds = [*splits, None]
        for i, split in enumerate(bounds):
            first_frame = start // WINDOW
            if split is None:
                segment = audio_pad[start:]
                last_frame = p_len
            else:
                segment = audio_pad[start : split + 2 * t_pad + WINDOW]
                last_frame = (split + 2 * t_pad) // WINDOW
            frames = slice(first_frame, last_frame)
            out = self._convert_segment(
                segment,
                last_frame - first_frame,
                pitch[:, frames] if pitch is not None else None,
                pitchf[:, frames] if pitchf is not None else None,
                sid,
                settings,
            )
            outputs.append(out[t_pad_tgt : len(out) - t_pad_tgt])
            if split is not None:
                start = split
            if progress:
                progress((i + 1) / len(bounds), "Converting voice")

        converted = np.concatenate(outputs).astype(np.float32)
        converted = dsp.match_length(converted, round(n * self.tgt_sr / SR_16K))
        if settings.rms_mix_rate < 1:
            converted = self._match_envelope(audio, converted, settings.rms_mix_rate)
        empty_cache(self.device)
        return dsp.prevent_clipping(converted, 0.99)

    def _split_points(self, audio: np.ndarray) -> list[int]:
        """Quiet points roughly every ``X_CENTER`` seconds, frame aligned."""
        if len(audio) <= SR_16K * X_MAX:
            return []
        energy = np.convolve(audio.astype(np.float64) ** 2, np.ones(WINDOW), "same")
        query = SR_16K * X_QUERY
        points = []
        for t in range(SR_16K * X_CENTER, len(audio), SR_16K * X_CENTER):
            lo, hi = max(t - query, 0), min(t + query, len(audio))
            point = lo + int(np.argmin(energy[lo:hi]))
            point = point // WINDOW * WINDOW
            if (not points or point > points[-1]) and point < len(audio) - WINDOW:
                points.append(point)
        return points

    @torch.no_grad()
    def _convert_segment(
        self,
        segment: np.ndarray,
        frames: int,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
        settings: ConversionSettings,
    ) -> np.ndarray:
        # HuBERT's 400-sample receptive field yields one frame less than the
        # pitch track for some lengths; a little tail padding makes every
        # chunk produce exactly ``frames`` frames so chunks never drift.
        padded = np.pad(segment, (0, 4 * WINDOW), mode="constant")
        feats = runtime.extract_content(padded, self.device, self.model.version)
        feats = feats.float()
        use_protect = pitch is not None and settings.protect < 0.5
        feats0 = feats.clone() if use_protect else None

        if self.index is not None and settings.index_rate > 0:
            blended = self.index.blend(feats[0].cpu().numpy(), settings.index_rate)
            feats = torch.from_numpy(blended).to(self.device)[None]

        feats = F.interpolate(feats.transpose(1, 2), scale_factor=2).transpose(1, 2)
        p_len = min(frames, feats.shape[1])
        if pitch is not None:
            p_len = min(p_len, pitch.shape[1])
        feats = feats[:, :p_len]

        if pitch is not None:
            pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
            if feats0 is not None:
                feats0 = F.interpolate(feats0.transpose(1, 2), scale_factor=2)
                feats0 = feats0.transpose(1, 2)[:, :p_len]
                mask = torch.where(pitchf > 0, 1.0, settings.protect).unsqueeze(-1)
                feats = feats * mask + feats0 * (1 - mask)

        lengths = torch.tensor([p_len], device=self.device)
        audio = self.net.infer(feats, lengths, pitch, pitchf, sid)
        return audio[0, 0].float().cpu().numpy()

    def _match_envelope(
        self, source16k: np.ndarray, converted: np.ndarray, rate: float
    ) -> np.ndarray:
        """Blend the loudness envelope of the source into the conversion.

        ``rate`` = 1 keeps the converted loudness, 0 follows the source fully.
        Gain is capped so near-silent converted passages are not amplified.
        """
        frame_s, hop_s = 0.2, 0.1
        src = dsp.frame_rms(source16k, int(SR_16K * frame_s), int(SR_16K * hop_s))
        out = dsp.frame_rms(
            converted, int(self.tgt_sr * frame_s), int(self.tgt_sr * hop_s)
        )
        # Work at frame rate, then interpolate the gain once (memory stays
        # proportional to the output even for long songs).
        frames = np.linspace(0, 1, len(out))
        src_env = np.interp(frames, np.linspace(0, 1, len(src)), src)
        gain = (src_env / np.maximum(out, 1e-6)) ** (1 - rate)
        gain = torch.from_numpy(np.minimum(gain, 4.0).astype(np.float32))
        gain = F.interpolate(
            gain.view(1, 1, -1), size=len(converted), mode="linear", align_corners=True
        ).view(-1)
        return converted * gain.numpy()
