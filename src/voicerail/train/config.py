"""Training hyper-parameters per sample rate (RVC v2 defaults)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..assets import DEFAULT_PRETRAINED
from ..rvc.nn.synthesizer import SynthesizerConfig


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    n_mel_channels: int
    segment_samples: int
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]

    @property
    def spec_channels(self) -> int:
        return self.filter_length // 2 + 1

    @property
    def segment_frames(self) -> int:
        return self.segment_samples // self.hop_length

    def synthesizer(self) -> SynthesizerConfig:
        return SynthesizerConfig(
            spec_channels=self.spec_channels,
            segment_size=self.segment_frames,
            upsample_rates=list(self.upsample_rates),
            upsample_kernel_sizes=list(self.upsample_kernel_sizes),
            sr=self.sample_rate,
        )


AUDIO_CONFIGS = {
    32000: AudioConfig(
        32000, 1024, 320, 1024, 80, 12800, (10, 8, 2, 2), (20, 16, 4, 4)
    ),
    40000: AudioConfig(
        40000, 2048, 400, 2048, 125, 12800, (10, 10, 2, 2), (16, 16, 4, 4)
    ),
    48000: AudioConfig(
        48000, 2048, 480, 2048, 128, 17280, (12, 10, 2, 2), (24, 20, 4, 4)
    ),
}
SAMPLE_RATES = tuple(AUDIO_CONFIGS)


@dataclass
class TrainConfig:
    sample_rate: int = 40000
    epochs: int = 0  # 0 = choose automatically from the dataset size
    batch_size: int = 0  # 0 = choose automatically from GPU memory
    pretrained: str = DEFAULT_PRETRAINED
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    save_every: int = 10
    keep_snapshots: bool = False
    early_stopping: bool = True
    patience: int = 40
    learning_rate: float = 1e-4
    lr_decay: float = 0.999875
    betas: tuple[float, float] = (0.8, 0.99)
    eps: float = 1e-9
    c_mel: float = 45.0
    c_kl: float = 1.0
    seed: int = 1234
    gradient_checkpointing: bool = False
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        data = asdict(self)
        data["betas"] = list(self.betas)
        return data

    @classmethod
    def from_json(cls, data: dict) -> TrainConfig:
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "betas" in known:
            known["betas"] = tuple(known["betas"])
        return cls(**known)


def auto_epochs(train_items: int, batch_size: int, target_steps: int = 7000) -> int:
    steps_per_epoch = max(1, -(-train_items // batch_size))
    return int(min(300, max(50, round(target_steps / steps_per_epoch))))


def auto_batch_size(vram_gb: float | None, sample_rate: int) -> int:
    if vram_gb is None:
        return 4
    per_item = 0.75 if sample_rate == 48000 else 0.6
    usable = max(vram_gb - 2.0, 1.0)
    return int(min(16, max(2, usable // per_item)))
