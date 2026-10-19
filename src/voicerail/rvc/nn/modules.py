"""WaveNet blocks, normalizing flow and HiFi-GAN residual blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .commons import (
    LRELU_SLOPE,
    fused_add_tanh_sigmoid_multiply,
    get_padding,
    init_weights,
    wn,
)


class WaveNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.drop = torch.nn.Dropout(p_dropout)
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        if gin_channels:
            self.cond_layer = wn(
                torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )
        for i in range(n_layers):
            dilation = dilation_rate**i
            self.in_layers.append(
                wn(
                    torch.nn.Conv1d(
                        hidden_channels,
                        2 * hidden_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=(kernel_size * dilation - dilation) // 2,
                    )
                )
            )
            res_skip = hidden_channels if i == n_layers - 1 else 2 * hidden_channels
            self.res_skip_layers.append(
                wn(torch.nn.Conv1d(hidden_channels, res_skip, 1))
            )

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)
        h = self.hidden_channels
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = g[:, i * 2 * h : (i + 1) * 2 * h] if g is not None else 0.0
            acts = self.drop(fused_add_tanh_sigmoid_multiply(x_in, g_l, h))
            res_skip = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip[:, :h]) * x_mask
                output = output + res_skip[:, h:]
            else:
                output = output + res_skip
        return output * x_mask


class ResidualCouplingLayer(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        assert channels % 2 == 0
        self.half_channels = channels // 2
        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        # Mean-only coupling, as in RVC.
        self.post = torch.nn.Conv1d(hidden_channels, self.half_channels, 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        m = self.post(h) * x_mask
        x1 = (x1 - m) * x_mask if reverse else m + x1 * x_mask
        return torch.cat([x0, x1], 1)


class Flip(torch.nn.Module):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.flip(x, [1])


class ResidualCouplingBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.flows = torch.nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResBlock(torch.nn.Module):
    """HiFi-GAN ``ResBlock1``."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] | list[int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = torch.nn.ModuleList(
            self._conv(channels, kernel_size, d) for d in dilations
        )
        self.convs2 = torch.nn.ModuleList(
            self._conv(channels, kernel_size, 1) for _ in dilations
        )

    @staticmethod
    def _conv(channels: int, kernel_size: int, dilation: int) -> torch.nn.Module:
        conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            1,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
        init_weights(conv)
        return wn(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x
