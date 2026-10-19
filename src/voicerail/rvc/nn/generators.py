"""HiFi-GAN vocoders used by RVC (with and without NSF pitch excitation)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .commons import LRELU_SLOPE, init_weights, wn
from .modules import ResBlock


class SineGenerator(torch.nn.Module):
    """Harmonic sine excitation with continuous phase across frames."""

    def __init__(
        self,
        sample_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.dim = harmonic_num + 1
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold

    @torch.no_grad()
    def forward(self, f0: torch.Tensor, upp: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``f0`` is ``[B, T]`` in Hz; returns ``[B, T * upp, dim]`` waves."""
        f0 = f0.float()
        batch, frames = f0.shape
        increment = f0 / self.sample_rate
        # Phase reached at the start of every frame, wrapped to one cycle so
        # float32 stays accurate for arbitrarily long inputs.
        advance = torch.remainder(increment * upp, 1.0)
        start = torch.remainder(torch.cumsum(advance, dim=1) - advance, 1.0)
        steps = torch.arange(1, upp + 1, device=f0.device, dtype=f0.dtype)
        phase = start.unsqueeze(-1) + increment.unsqueeze(-1) * steps
        phase = phase.reshape(batch, frames * upp, 1)

        harmonics = torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype)
        phase = phase * harmonics
        offset = torch.rand(batch, 1, self.dim, device=f0.device)
        offset[..., 0] = 0
        sines = torch.sin(2 * math.pi * (phase + offset)) * self.sine_amp

        uv = (f0 > self.voiced_threshold).float()
        uv = uv.repeat_interleave(upp, dim=1).unsqueeze(-1)
        noise_amp = uv * self.noise_std + (1 - uv) * (self.sine_amp / 3)
        noise = noise_amp * torch.randn_like(sines)
        return sines * uv + noise, uv


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, sample_rate: int, harmonic_num: int = 0):
        super().__init__()
        self.l_sin_gen = SineGenerator(sample_rate, harmonic_num)
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
        sines, _ = self.l_sin_gen(f0, upp)
        sines = sines.to(dtype=self.l_linear.weight.dtype)
        return self.l_tanh(self.l_linear(sines))


class _HiFiGANBase(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        self.ups = torch.nn.ModuleList()
        self.resblocks = torch.nn.ModuleList()
        channels = upsample_initial_channel
        for i, (u, k) in enumerate(
            zip(upsample_rates, upsample_kernel_sizes, strict=True)
        ):
            channels = upsample_initial_channel // (2 ** (i + 1))
            up = torch.nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                channels,
                k,
                u,
                padding=(k - u) // 2,
            )
            init_weights(up)
            self.ups.append(wn(up))
            for kernel, dilations in zip(
                resblock_kernel_sizes, resblock_dilation_sizes, strict=True
            ):
                self.resblocks.append(ResBlock(channels, kernel, dilations))
        self.conv_post = torch.nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False)
        if gin_channels:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def _stage(self, i: int, x: torch.Tensor) -> torch.Tensor:
        blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]
        if self.training and self.checkpointing:
            out = sum(checkpoint(block, x, use_reentrant=False) for block in blocks)
        else:
            out = sum(block(x) for block in blocks)
        return out / self.num_kernels

    def _upsample(self, i: int, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x, LRELU_SLOPE)
        if self.training and self.checkpointing:
            return checkpoint(self.ups[i], x, use_reentrant=False)
        return self.ups[i](x)

    def _post(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x)
        return torch.tanh(self.conv_post(x))


class Generator(_HiFiGANBase):
    """Plain HiFi-GAN generator used by RVC models without pitch guidance."""

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = self._stage(i, self._upsample(i, x))
        return self._post(x)


class NSFGenerator(_HiFiGANBase):
    """HiFi-GAN driven by a harmonic source signal built from F0."""

    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list[int],
        resblock_dilation_sizes: list[list[int]],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: list[int],
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
    ):
        super().__init__(
            initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels,
            checkpointing,
        )
        self.upp = math.prod(upsample_rates)
        self.m_source = SourceModuleHnNSF(sr, harmonic_num=0)
        self.noise_convs = torch.nn.ModuleList()
        for i in range(len(upsample_rates)):
            channels = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride = math.prod(upsample_rates[i + 1 :])
                conv = torch.nn.Conv1d(
                    1,
                    channels,
                    kernel_size=stride * 2,
                    stride=stride,
                    padding=stride // 2,
                )
            else:
                conv = torch.nn.Conv1d(1, channels, kernel_size=1)
            self.noise_convs.append(conv)

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = self._upsample(i, x)
            x = x + self.noise_convs[i](source)
            x = self._stage(i, x)
        return self._post(x)
