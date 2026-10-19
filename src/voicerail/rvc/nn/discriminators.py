"""Multi-period discriminator used for adversarial training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from torch.utils.checkpoint import checkpoint

from .commons import LRELU_SLOPE, get_padding

V2_PERIODS = (2, 3, 5, 7, 11, 17, 23, 37)


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = torch.nn.ModuleList(
            [
                norm(torch.nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm(torch.nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm(torch.nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm(torch.nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm(torch.nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm(torch.nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm(torch.nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm = spectral_norm if use_spectral_norm else weight_norm
        channels = [1, 32, 128, 512, 1024, 1024]
        strides = [stride, stride, stride, stride, 1]
        self.convs = torch.nn.ModuleList(
            norm(
                torch.nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    (kernel_size, 1),
                    (strides[i], 1),
                    padding=(get_padding(kernel_size, 1), 0),
                )
            )
            for i in range(5)
        )
        self.conv_post = norm(torch.nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        fmap = []
        b, c, t = x.shape
        if t % self.period:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(
        self,
        periods: tuple[int, ...] = V2_PERIODS,
        use_spectral_norm: bool = False,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        self.discriminators = torch.nn.ModuleList(
            [DiscriminatorS(use_spectral_norm)]
            + [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in periods]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        real_outs, fake_outs, real_maps, fake_maps = [], [], [], []
        for disc in self.discriminators:
            if self.training and self.checkpointing:
                y_d_r, fmap_r = checkpoint(disc, y, use_reentrant=False)
                y_d_g, fmap_g = checkpoint(disc, y_hat, use_reentrant=False)
            else:
                y_d_r, fmap_r = disc(y)
                y_d_g, fmap_g = disc(y_hat)
            real_outs.append(y_d_r)
            fake_outs.append(y_d_g)
            real_maps.append(fmap_r)
            fake_maps.append(fmap_g)
        return real_outs, fake_outs, real_maps, fake_maps
