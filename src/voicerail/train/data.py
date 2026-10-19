"""In-memory training dataset and batching."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..rvc.f0 import coarse_f0
from .config import AudioConfig
from .features import FeatureItem

MAX_FRAMES = 900


class Example:
    __slots__ = ("phone", "pitch", "pitchf", "wav")

    def __init__(self, phone, pitch, pitchf, wav):
        self.phone = phone
        self.pitch = pitch
        self.pitchf = pitchf
        self.wav = wav


def load_example(item: FeatureItem, audio: AudioConfig) -> Example | None:
    wav, sr = sf.read(item.wav, dtype="float32")
    if sr != audio.sample_rate:
        raise ValueError(f"{item.wav}: {sr} Hz but training at {audio.sample_rate} Hz")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    phone = np.repeat(np.load(item.feats), 2, axis=0)
    f0 = np.load(item.f0).astype(np.float32)
    frames = min(len(phone), len(f0), len(wav) // audio.hop_length, MAX_FRAMES)
    if frames < audio.segment_frames:
        return None
    return Example(
        phone=torch.from_numpy(phone[:frames].astype(np.float16)),
        pitch=torch.from_numpy(coarse_f0(f0[:frames])),
        pitchf=torch.from_numpy(f0[:frames]),
        wav=torch.from_numpy(wav[: frames * audio.hop_length].copy()),
    )


class TrainingSet(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Example:
        return self.examples[index]


def collate(batch: list[Example]):
    frames = torch.tensor([len(e.pitch) for e in batch])
    max_frames = int(frames.max())
    hop = len(batch[0].wav) // len(batch[0].pitch)
    phone = torch.zeros(len(batch), max_frames, batch[0].phone.shape[1])
    pitch = torch.zeros(len(batch), max_frames, dtype=torch.long)
    pitchf = torch.zeros(len(batch), max_frames)
    wav = torch.zeros(len(batch), 1, max_frames * hop)
    for i, e in enumerate(batch):
        n = len(e.pitch)
        phone[i, :n] = e.phone.float()
        pitch[i, :n] = e.pitch
        pitchf[i, :n] = e.pitchf
        wav[i, 0, : n * hop] = e.wav
    sid = torch.zeros(len(batch), dtype=torch.long)
    return phone, frames, pitch, pitchf, wav, sid
