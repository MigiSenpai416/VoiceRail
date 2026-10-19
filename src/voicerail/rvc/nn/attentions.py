"""Transformer encoder with relative position attention (from VITS)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class LayerNorm(torch.nn.Module):
    """Channel-first layer norm with VITS parameter names."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (x.size(-1),), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: int | None = None,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.drop = torch.nn.Dropout(p_dropout)

        if window_size:
            std = self.k_channels**-0.5
            shape = (1, 2 * window_size + 1, self.k_channels)
            self.emb_rel_k = torch.nn.Parameter(torch.randn(shape) * std)
            self.emb_rel_v = torch.nn.Parameter(torch.randn(shape) * std)

        for conv in (self.conv_q, self.conv_k, self.conv_v):
            torch.nn.init.xavier_uniform_(conv.weight)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        return self.conv_o(self._attention(q, k, v, attn_mask))

    def _attention(self, query, key, value, mask):
        b, d, t_s = key.size()
        t_t = query.size(2)
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        query = query / math.sqrt(self.k_channels)
        scores = torch.matmul(query, key.transpose(-2, -1))
        if self.window_size:
            rel_k = self._relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = torch.matmul(query, rel_k.unsqueeze(0).transpose(-2, -1))
            scores = scores + self._relative_to_absolute(rel_logits)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        p_attn = self.drop(F.softmax(scores, dim=-1))
        output = torch.matmul(p_attn, value)
        if self.window_size:
            rel_weights = self._absolute_to_relative(p_attn)
            rel_v = self._relative_embeddings(self.emb_rel_v, t_s)
            output = output + torch.matmul(rel_weights, rel_v.unsqueeze(0))
        return output.transpose(2, 3).contiguous().view(b, d, t_t)

    def _relative_embeddings(self, embeddings: torch.Tensor, length: int):
        pad = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        if pad > 0:
            embeddings = F.pad(embeddings, (0, 0, pad, pad))
        return embeddings[:, start : start + 2 * length - 1]

    @staticmethod
    def _relative_to_absolute(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, 1))
        x = x.view(batch, heads, length * 2 * length)
        x = F.pad(x, (0, length - 1))
        x = x.view(batch, heads, length + 1, 2 * length - 1)
        return x[:, :, :length, length - 1 :]

    @staticmethod
    def _absolute_to_relative(x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, _ = x.size()
        x = F.pad(x, (0, length - 1))
        x = x.view(batch, heads, length**2 + length * (length - 1))
        x = F.pad(x, (length, 0))
        return x.view(batch, heads, length, 2 * length)[:, :, :, 1:]


class FFN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = torch.nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = torch.nn.Dropout(p_dropout)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        return F.pad(x, ((self.kernel_size - 1) // 2, self.kernel_size // 2))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self._pad(x * x_mask))
        x = self.drop(torch.relu(x))
        x = self.conv_2(self._pad(x * x_mask))
        return x * x_mask


class Encoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.drop(self.attn_layers[i](x, x, attn_mask))
            x = self.norm_layers_1[i](x + y)
            y = self.drop(self.ffn_layers[i](x, x_mask))
            x = self.norm_layers_2[i](x + y)
        return x * x_mask
