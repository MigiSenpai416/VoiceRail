"""Minimal HuBERT / ContentVec feature extractor.

A self-contained re-implementation of the ``transformers`` HuBERT base
encoder (post layer-norm variant) that loads the same ``pytorch_model.bin``
checkpoints. Only inference is supported. Keeping this in-tree avoids a heavy
dependency whose internals change between releases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import safe_load


@dataclass
class HubertConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-5
    conv_dim: tuple[int, ...] = (512,) * 7
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = False
    feat_extract_norm: str = "group"
    feat_proj_layer_norm: bool = True
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    do_stable_layer_norm: bool = False
    classifier_proj_size: int = 256

    @classmethod
    def from_json(cls, path: Path) -> HubertConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        for key in ("conv_dim", "conv_kernel", "conv_stride"):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)


class _ConvLayer(nn.Module):
    def __init__(self, config: HubertConfig, index: int):
        super().__init__()
        in_dim = config.conv_dim[index - 1] if index > 0 else 1
        out_dim = config.conv_dim[index]
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            config.conv_kernel[index],
            stride=config.conv_stride[index],
            bias=config.conv_bias,
        )
        if config.feat_extract_norm == "group" and index == 0:
            self.layer_norm = nn.GroupNorm(out_dim, out_dim, affine=True)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return F.gelu(x)


class _FeatureExtractor(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            _ConvLayer(config, i) for i in range(len(config.conv_dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, None]
        for layer in self.conv_layers:
            x = layer(x)
        return x


class _FeatureProjection(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            self.layer_norm = nn.LayerNorm(
                config.conv_dim[-1], eps=config.layer_norm_eps
            )
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feat_proj_layer_norm:
            x = self.layer_norm(x)
        return self.projection(x)


class _PositionalConv(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.remove = 1 if config.num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2))
        if self.remove:
            x = x[:, :, : -self.remove]
        return F.gelu(x).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.heads = config.num_attention_heads
        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape

        def split(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, self.heads, d // self.heads).transpose(1, 2)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(b, t, d))


class _FeedForward(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.intermediate_dense = nn.Linear(
            config.hidden_size, config.intermediate_size
        )
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class _EncoderLayer(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.attention = _Attention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = _FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x + self.attention(x))
        return self.final_layer_norm(x + self.feed_forward(x))


class _Encoder(nn.Module):
    def __init__(self, config: HubertConfig):
        super().__init__()
        self.pos_conv_embed = _PositionalConv(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            _EncoderLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, x: torch.Tensor, output_layer: int) -> torch.Tensor:
        x = self.layer_norm(x + self.pos_conv_embed(x))
        for layer in self.layers[:output_layer]:
            x = layer(x)
        return x


class HubertEncoder(nn.Module):
    """HuBERT base encoder; ``forward`` returns features at 50 frames/s."""

    def __init__(self, config: HubertConfig):
        super().__init__()
        if config.do_stable_layer_norm:
            raise ValueError("Pre-layer-norm HuBERT variants are not supported.")
        self.config = config
        self.feature_extractor = _FeatureExtractor(config)
        self.feature_projection = _FeatureProjection(config)
        self.encoder = _Encoder(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    @torch.no_grad()
    def forward(self, wav16k: torch.Tensor, version: str = "v2") -> torch.Tensor:
        """``wav16k`` is ``[B, samples]``; returns ``[B, frames, dim]``.

        RVC v2 models consume the last (12th) layer, v1 models the 9th layer
        passed through the final projection.
        """
        x = self.feature_extractor(wav16k).transpose(1, 2)
        x = self.feature_projection(x)
        if version == "v1":
            return self.final_proj(self.encoder(x, output_layer=9))
        return self.encoder(x, output_layer=self.config.num_hidden_layers)


def load_hubert(model_dir: str | Path, device: torch.device) -> HubertEncoder:
    """Load a HuBERT checkpoint in Hugging Face layout (config + weights)."""
    model_dir = Path(model_dir)
    config = HubertConfig.from_json(model_dir / "config.json")
    model = HubertEncoder(config)
    state = safe_load(model_dir / "pytorch_model.bin")
    state = {k.removeprefix("hubert."): v for k, v in state.items()}
    _bake_positional_weight_norm(state)
    own = model.state_dict()
    missing = [k for k in own if k not in state and not k.startswith("final_proj")]
    if missing:
        raise ValueError(
            f"'{model_dir.name}' is not a supported HuBERT checkpoint "
            f"(missing {', '.join(missing[:3])})."
        )
    model.load_state_dict(
        {k: state[k].float() if k in state else v for k, v in own.items()}
    )
    return model.eval().to(device)


def _bake_positional_weight_norm(state: dict[str, torch.Tensor]) -> None:
    prefix = "encoder.pos_conv_embed.conv."
    for g_key, v_key in (
        ("weight_g", "weight_v"),
        ("parametrizations.weight.original0", "parametrizations.weight.original1"),
    ):
        if prefix + g_key in state:
            g = state.pop(prefix + g_key).float()
            v = state.pop(prefix + v_key).float()
            # Weight norm over every axis except the kernel axis (dim=2).
            norm = v.norm(dim=(0, 1), keepdim=True)
            state[prefix + "weight"] = v * (g / norm)
