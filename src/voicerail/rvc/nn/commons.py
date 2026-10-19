"""Small tensor helpers shared by the RVC network modules."""

from __future__ import annotations

import torch
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

LRELU_SLOPE = 0.1


def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01):
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def sequence_mask(lengths: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        max_length = int(lengths.max())
    positions = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def slice_segments(x: torch.Tensor, starts: torch.Tensor, size: int) -> torch.Tensor:
    """Cut ``size`` steps from the last axis of every batch item."""
    index = starts.view(-1, *([1] * (x.dim() - 1))) + torch.arange(
        size, device=x.device
    )
    index = index.expand(*x.shape[:-1], size)
    return torch.gather(x, -1, index)


def rand_slice_segments(
    x: torch.Tensor, lengths: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = (lengths - size + 1).clamp(min=1)
    starts = (torch.rand(x.size(0), device=x.device) * max_start).long()
    return slice_segments(x, starts, size), starts


def fused_add_tanh_sigmoid_multiply(
    a: torch.Tensor, b: torch.Tensor | float, channels: int
) -> torch.Tensor:
    acts = a + b
    return torch.tanh(acts[:, :channels]) * torch.sigmoid(acts[:, channels:])


def wn(module: torch.nn.Module) -> torch.nn.Module:
    return weight_norm(module)


def remove_weight_norm(model: torch.nn.Module) -> torch.nn.Module:
    """Bake every weight-norm parametrization into a plain weight."""
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight")
    return model
