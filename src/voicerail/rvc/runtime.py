"""Process-wide cache of the shared feature models (HuBERT and RMVPE)."""

from __future__ import annotations

import threading

import numpy as np
import torch

from .. import assets
from ..device import default_device, empty_cache
from .f0 import refine_f0
from .hubert import HubertEncoder, load_hubert
from .rmvpe import RMVPE

_lock = threading.Lock()
_hubert: dict[str, HubertEncoder] = {}
_rmvpe: dict[str, RMVPE] = {}


def hubert(device: torch.device | None = None) -> HubertEncoder:
    device = device or default_device()
    with _lock:
        if str(device) not in _hubert:
            assets.ensure(("contentvec/config.json", "contentvec/pytorch_model.bin"))
            _hubert[str(device)] = load_hubert(assets.contentvec_dir(), device)
        return _hubert[str(device)]


def rmvpe(device: torch.device | None = None) -> RMVPE:
    device = device or default_device()
    with _lock:
        if str(device) not in _rmvpe:
            assets.ensure(("rmvpe.pt",))
            _rmvpe[str(device)] = RMVPE(assets.rmvpe_path(), device)
        return _rmvpe[str(device)]


def extract_f0(
    audio16k: np.ndarray,
    device: torch.device | None = None,
    threshold: float = 0.03,
    refine: bool = True,
) -> np.ndarray:
    f0 = rmvpe(device)(audio16k, threshold=threshold)
    return refine_f0(f0) if refine else f0


@torch.no_grad()
def extract_content(
    audio16k: np.ndarray, device: torch.device | None = None, version: str = "v2"
) -> torch.Tensor:
    """HuBERT features ``[1, frames, dim]`` at 50 frames per second."""
    model = hubert(device)
    device = next(model.parameters()).device
    wav = torch.from_numpy(np.ascontiguousarray(audio16k, dtype=np.float32))
    return model(wav.to(device).unsqueeze(0), version=version)


def release() -> None:
    """Free the cached models (e.g. before training on the same GPU)."""
    with _lock:
        devices = {str(m.device) for m in _rmvpe.values()}
        _hubert.clear()
        _rmvpe.clear()
    for device in devices:
        empty_cache(torch.device(device))
