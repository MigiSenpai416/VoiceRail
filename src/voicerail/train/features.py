"""Pitch and content feature extraction for every training segment."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ..rvc import runtime
from .dataset import dataset_dir
from .progress import Reporter

log = logging.getLogger(__name__)

MIN_VOICED_RATIO = 0.1
MUTE_SECONDS = 3.0


@dataclass
class FeatureItem:
    name: str
    wav: str
    feats: str
    f0: str
    voiced: float
    validation: bool = False
    mute: bool = False


@dataclass
class FeatureSummary:
    items: list[FeatureItem]
    f0_median: float | None
    f0_low: float | None
    f0_high: float | None
    skipped_unvoiced: int

    def to_json(self) -> dict:
        return {
            "items": [asdict(i) for i in self.items],
            "f0_median": self.f0_median,
            "f0_low": self.f0_low,
            "f0_high": self.f0_high,
            "skipped_unvoiced": self.skipped_unvoiced,
        }

    @classmethod
    def from_json(cls, data: dict) -> FeatureSummary:
        return cls(
            items=[FeatureItem(**i) for i in data["items"]],
            f0_median=data.get("f0_median"),
            f0_low=data.get("f0_low"),
            f0_high=data.get("f0_high"),
            skipped_unvoiced=data.get("skipped_unvoiced", 0),
        )


def features_dir(exp_dir: Path) -> Path:
    return exp_dir / "features"


def _validation_count(n: int) -> int:
    if n < 40:
        return 0
    return int(min(12, max(2, round(0.03 * n))))


def extract_features(
    exp_dir: Path,
    sample_rate: int,
    reporter: Reporter,
    device: torch.device | None = None,
) -> FeatureSummary:
    ds = dataset_dir(exp_dir)
    wavs16 = sorted((ds / "wavs16k").glob("*.wav"))
    if not wavs16:
        raise ValueError("The dataset is empty; prepare it first.")
    out = features_dir(exp_dir)
    feats_dir, f0_dir = out / "feats", out / "f0"
    feats_dir.mkdir(parents=True, exist_ok=True)
    f0_dir.mkdir(parents=True, exist_ok=True)

    items: list[FeatureItem] = []
    voiced_f0: list[np.ndarray] = []
    skipped = 0
    for i, wav16 in enumerate(wavs16):
        reporter.check_stop()
        if i % 5 == 0:
            reporter.progress(
                i / len(wavs16), f"Analysing segment {i + 1}/{len(wavs16)}"
            )
        name = wav16.stem
        feats_path, f0_path = feats_dir / f"{name}.npy", f0_dir / f"{name}.npy"
        if feats_path.is_file() and f0_path.is_file():
            f0 = np.load(f0_path)
        else:
            audio, sr = sf.read(wav16, dtype="float32")
            assert sr == 16000
            f0 = runtime.extract_f0(audio, device).astype(np.float32)
            feats = runtime.extract_content(audio, device, "v2")[0]
            feats = feats.float().cpu().numpy()
            if not np.isfinite(feats).all():
                log.warning("Skipping %s: invalid features", name)
                continue
            np.save(f0_path, f0)
            np.save(feats_path, feats.astype(np.float16))
        voiced = float(np.mean(f0 > 0))
        if voiced < MIN_VOICED_RATIO:
            skipped += 1
            continue
        voiced_f0.append(f0[f0 > 0])
        items.append(
            FeatureItem(
                name=name,
                wav=str(ds / "wavs" / wav16.name),
                feats=str(feats_path),
                f0=str(f0_path),
                voiced=voiced,
            )
        )
    if not items:
        raise ValueError(
            "None of the segments contain enough voiced speech or singing."
        )

    rng = np.random.default_rng(0)
    candidates = [i for i, item in enumerate(items) if item.voiced > 0.3]
    n_val = min(_validation_count(len(items)), len(candidates) // 4)
    for index in rng.choice(candidates, size=n_val, replace=False):
        items[int(index)].validation = True

    items.extend(_mute_items(exp_dir, sample_rate, device))

    all_f0 = np.concatenate(voiced_f0) if voiced_f0 else np.zeros(0)
    if all_f0.size:
        log_f0 = np.log(all_f0)
        f0_median = float(np.exp(np.median(log_f0)))
        f0_low = float(np.exp(np.percentile(log_f0, 5)))
        f0_high = float(np.exp(np.percentile(log_f0, 95)))
    else:
        f0_median = f0_low = f0_high = None
    summary = FeatureSummary(items, f0_median, f0_low, f0_high, skipped)
    (out / "filelist.json").write_text(json.dumps(summary.to_json(), indent=1))
    reporter.progress(1.0, "Features extracted")
    return summary


def _mute_items(
    exp_dir: Path, sample_rate: int, device: torch.device | None
) -> list[FeatureItem]:
    """Two silent samples teach the model to output silence for silence."""
    mute_dir = features_dir(exp_dir) / "mute"
    mute_dir.mkdir(parents=True, exist_ok=True)
    wav = mute_dir / f"mute_{sample_rate}.wav"
    feats, f0 = mute_dir / "feats.npy", mute_dir / "f0.npy"
    if not wav.is_file():
        sf.write(
            wav, np.zeros(int(sample_rate * MUTE_SECONDS), np.float32), sample_rate
        )
    if not feats.is_file():
        silence = np.zeros(int(16000 * MUTE_SECONDS), dtype=np.float32)
        content = runtime.extract_content(silence, device, "v2")[0].float().cpu()
        np.save(feats, content.numpy().astype(np.float16))
        np.save(f0, np.zeros(int(MUTE_SECONDS * 100) + 1, dtype=np.float32))
    return [
        FeatureItem(f"mute{i}", str(wav), str(feats), str(f0), 0.0, mute=True)
        for i in range(2)
    ]


def load_summary(exp_dir: Path) -> FeatureSummary | None:
    path = features_dir(exp_dir) / "filelist.json"
    if not path.is_file():
        return None
    return FeatureSummary.from_json(json.loads(path.read_text()))


def clear_features(exp_dir: Path) -> None:
    shutil.rmtree(features_dir(exp_dir), ignore_errors=True)
