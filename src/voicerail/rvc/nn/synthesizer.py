"""The RVC synthesizer (a VITS variant conditioned on content features)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from .commons import rand_slice_segments, slice_segments
from .encoders import PosteriorEncoder, TextEncoder
from .generators import Generator, NSFGenerator
from .modules import ResidualCouplingBlock


@dataclass
class SynthesizerConfig:
    """Hyper-parameters stored in the ``config`` list of RVC model files."""

    spec_channels: int = 1025
    segment_size: int = 32
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0.0
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: list[list[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    upsample_rates: list[int] = field(default_factory=lambda: [10, 10, 2, 2])
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [16, 16, 4, 4])
    spk_embed_dim: int = 109
    gin_channels: int = 256
    sr: int = 40000

    @classmethod
    def from_list(cls, values: list) -> SynthesizerConfig:
        names = list(cls.__dataclass_fields__)
        if len(values) != len(names):
            raise ValueError(
                f"Expected {len(names)} config values, found {len(values)}."
            )
        config = cls(**dict(zip(names, values, strict=True)))
        config.sr = _parse_sample_rate(config.sr)
        return config

    def to_list(self) -> list:
        return list(asdict(self).values())


def _parse_sample_rate(value: int | str) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        return int(float(value[:-1]) * 1000) if value.endswith("k") else int(value)
    return int(value)


class Synthesizer(torch.nn.Module):
    def __init__(
        self,
        config: SynthesizerConfig,
        use_f0: bool = True,
        embedding_dim: int = 768,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.config = config
        self.use_f0 = use_f0
        self.segment_size = config.segment_size
        self.enc_p = TextEncoder(
            config.inter_channels,
            config.hidden_channels,
            config.filter_channels,
            config.n_heads,
            config.n_layers,
            config.kernel_size,
            config.p_dropout,
            embedding_dim,
            f0=use_f0,
        )
        decoder_args = (
            config.inter_channels,
            config.resblock_kernel_sizes,
            config.resblock_dilation_sizes,
            config.upsample_rates,
            config.upsample_initial_channel,
            config.upsample_kernel_sizes,
            config.gin_channels,
        )
        if use_f0:
            self.dec = NSFGenerator(
                *decoder_args, sr=config.sr, checkpointing=checkpointing
            )
        else:
            self.dec = Generator(*decoder_args, checkpointing=checkpointing)
        self.enc_q = PosteriorEncoder(
            config.spec_channels,
            config.inter_channels,
            config.hidden_channels,
            5,
            1,
            16,
            gin_channels=config.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            config.inter_channels,
            config.hidden_channels,
            5,
            1,
            3,
            gin_channels=config.gin_channels,
        )
        self.emb_g = torch.nn.Embedding(config.spk_embed_dim, config.gin_channels)

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        spec: torch.Tensor,
        spec_lengths: torch.Tensor,
        sid: torch.Tensor,
    ):
        """Training forward pass on a random slice of every item."""
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
        if self.use_f0:
            pitchf = slice_segments(pitchf, ids_slice, self.segment_size)
            o = self.dec(z_slice, pitchf, g=g)
        else:
            o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.no_grad()
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor | None,
        pitchf: torch.Tensor | None,
        sid: torch.Tensor,
        noise_scale: float = 0.66666,
    ) -> torch.Tensor:
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * noise_scale) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        if self.use_f0:
            return self.dec(z * x_mask, pitchf, g=g)
        return self.dec(z * x_mask, g=g)
