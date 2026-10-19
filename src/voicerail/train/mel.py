"""Spectrogram features matching RVC's training front-end."""

from __future__ import annotations

import torch
from librosa.filters import mel as librosa_mel

from .config import AudioConfig


class MelFrontEnd(torch.nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        basis = librosa_mel(
            sr=config.sample_rate,
            n_fft=config.filter_length,
            n_mels=config.n_mel_channels,
            fmin=0.0,
            fmax=None,
        )
        self.register_buffer("mel_basis", torch.from_numpy(basis).float())
        self.register_buffer("window", torch.hann_window(config.win_length))

    def spectrogram(self, wav: torch.Tensor) -> torch.Tensor:
        """``wav [B, N]`` -> linear magnitude ``[B, F, N // hop]``."""
        c = self.config
        pad = (c.filter_length - c.hop_length) // 2
        wav = torch.nn.functional.pad(wav.float().unsqueeze(1), (pad, pad), "reflect")
        spec = torch.stft(
            wav.squeeze(1),
            n_fft=c.filter_length,
            hop_length=c.hop_length,
            win_length=c.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    def mel(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.spectrogram(wav)
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, spec), min=1e-5))
