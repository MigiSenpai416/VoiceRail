"""RMVPE: robust pitch estimation for vocals in polyphonic music.

Network definition follows the original RMVPE release so that the public
``rmvpe.pt`` weights load unchanged. Decoding is vectorised.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel
from torch import nn

from .checkpoint import safe_load

N_MELS = 128
N_CLASS = 360
HOP = 160
SAMPLE_RATE = 16000


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x) if self.shortcut is not None else x
        return self.conv(x) + residual


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1):
        super().__init__()
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels)])
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels))
        self.pool = nn.AvgPool2d(kernel_size) if kernel_size is not None else None

    def forward(self, x):
        for block in self.conv:
            x = block(x)
        if self.pool is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, n_encoders, kernel_size, n_blocks, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.01)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks)
            )
            in_channels, out_channels = out_channels, out_channels * 2
        self.out_channel = out_channels

    def forward(self, x):
        skips = []
        x = self.bn(x)
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(skip)
        return x, skips


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResEncoderBlock(in_channels, out_channels, None, n_blocks)]
        )
        for _ in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                3,
                stride,
                padding=1,
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=0.01),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels)])
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels))

    def forward(self, x, skip):
        x = torch.cat((self.conv1(x), skip), dim=1)
        for block in self.conv2:
            x = block(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks)
            )
            in_channels = out_channels

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4):
        super().__init__()
        self.encoder = Encoder(1, en_de_layers, kernel_size, n_blocks, 16)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(self.intermediate(x), skips)


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class E2E(nn.Module):
    def __init__(self, n_blocks=4, n_gru=1, kernel_size=(2, 2)):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks)
        self.cnn = nn.Conv2d(16, 3, 3, padding=1)
        self.fc = nn.Sequential(
            BiGRU(3 * N_MELS, 256, n_gru),
            nn.Linear(512, N_CLASS),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class _MelSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()
        basis = librosa_mel(
            sr=SAMPLE_RATE, n_fft=1024, n_mels=N_MELS, fmin=30, fmax=8000, htk=True
        )
        self.register_buffer("mel_basis", torch.from_numpy(basis).float())
        self.register_buffer("window", torch.hann_window(1024))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        fft = torch.stft(
            audio,
            n_fft=1024,
            hop_length=HOP,
            win_length=1024,
            window=self.window,
            center=True,
            return_complex=True,
        )
        magnitude = fft.abs()
        mel = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))


class RMVPE:
    """Pitch estimator returning F0 in Hz at 100 frames per second."""

    def __init__(self, model_path: str | Path, device: torch.device):
        self.device = device
        model = E2E()
        model.load_state_dict(safe_load(model_path))
        self.model = model.eval().to(device)
        self.mel = _MelSpectrogram().to(device)
        cents = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents, (4, 4))
        self.half = False

    @torch.no_grad()
    def salience(
        self, audio16k: np.ndarray, chunk_frames: int = 6400, context: int = 320
    ) -> np.ndarray:
        """Pitch salience ``[frames, 360]``.

        Long inputs are processed in overlapping chunks (with ``context``
        frames on both sides) to bound memory use without seams.
        """
        audio = torch.from_numpy(np.ascontiguousarray(audio16k, dtype=np.float32))
        mel = self.mel(audio.to(self.device).unsqueeze(0))
        n_frames = mel.shape[-1]
        outputs = []
        for start in range(0, n_frames, chunk_frames):
            stop = min(start + chunk_frames, n_frames)
            left = min(context, start)
            right = min(context, n_frames - stop)
            piece = mel[..., start - left : stop + right]
            length = piece.shape[-1]
            pad = 32 * ((length - 1) // 32 + 1) - length
            if pad:
                mode = "reflect" if pad < length else "constant"
                piece = F.pad(piece, (0, pad), mode=mode)
            hidden = self.model(piece.float())
            outputs.append(hidden[:, left : left + (stop - start)])
        hidden = torch.cat(outputs, dim=1)
        return hidden.squeeze(0).float().cpu().numpy()

    def decode(self, salience: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        center = np.argmax(salience, axis=1)
        padded = np.pad(salience, ((0, 0), (4, 4)))
        index = center[:, None] + np.arange(9)[None, :]
        window = np.take_along_axis(padded, index, axis=1)
        cents = self.cents_mapping[index]
        weight = window.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = (window * cents).sum(axis=1) / weight
        avg[(salience.max(axis=1) <= threshold) | ~np.isfinite(avg)] = 0
        f0 = 10 * (2 ** (avg / 1200))
        f0[avg == 0] = 0
        return f0

    def __call__(self, audio16k: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        return self.decode(self.salience(audio16k), threshold)
