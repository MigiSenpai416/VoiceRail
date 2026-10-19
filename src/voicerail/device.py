"""Compute device selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache

import torch


@cache
def default_device() -> torch.device:
    """The best available device, overridable with ``VOICERAIL_DEVICE``."""
    forced = os.environ.get("VOICERAIL_DEVICE", "").strip()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bf16_supported(device: torch.device) -> bool:
    """Native bf16 only (Ampere or newer); emulated bf16 is slower than fp32."""
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 8 and torch.cuda.is_bf16_supported()


def vram_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / 1024**3


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    name: str
    vram_gb: float | None
    torch_version: str
    accelerator: str


def describe(device: torch.device | None = None) -> DeviceInfo:
    device = device or default_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        accelerator = (
            f"ROCm {torch.version.hip}"
            if torch.version.hip
            else f"CUDA {torch.version.cuda}"
        )
    elif device.type == "mps":
        name, accelerator = "Apple Silicon GPU", "MPS"
    else:
        name, accelerator = "CPU", "none"
    return DeviceInfo(
        device=str(device),
        name=name,
        vram_gb=vram_gb(device),
        torch_version=torch.__version__,
        accelerator=accelerator,
    )
