"""Filesystem layout of the VoiceRail data directory.

Everything VoiceRail writes lives under one root so that it is easy to back
up, move or wipe. The root defaults to ``./data`` next to the current working
directory and can be overridden with the ``VOICERAIL_HOME`` environment
variable.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from pathlib import Path


def home() -> Path:
    root = os.environ.get("VOICERAIL_HOME")
    path = Path(root).expanduser() if root else Path.cwd() / "data"
    return path.resolve()


def assets_dir() -> Path:
    """Base models: embedder, pitch estimator, pretrained generators."""
    return _ensure(home() / "assets")


def separator_models_dir() -> Path:
    return _ensure(home() / "assets" / "separation")


def models_dir() -> Path:
    """Voice models ready for conversion, one folder per model."""
    return _ensure(home() / "models")


def training_dir() -> Path:
    """Training workspaces, one folder per experiment."""
    return _ensure(home() / "training")


def outputs_dir() -> Path:
    return _ensure(home() / "outputs")


def cache_dir() -> Path:
    """Intermediate results that can be safely deleted at any time."""
    return _ensure(home() / "cache")


def tmp_dir() -> Path:
    return _ensure(home() / "cache" / "tmp")


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


_UNSAFE = re.compile(r"[^\w\-. ()\[\]]+", flags=re.UNICODE)


def safe_name(name: str, fallback: str = "untitled") -> str:
    """Turn user input into a name that is safe to use as a file name."""
    name = unicodedata.normalize("NFC", name).strip()
    name = _UNSAFE.sub("_", name)
    name = re.sub(r"\s+", " ", name).strip(" ._")
    # Strip again after truncating so the function is idempotent.
    return name[:80].strip(" ._") or fallback


def replace_file(source: Path, target: Path, attempts: int = 20) -> None:
    """``os.replace`` that retries while Windows antivirus/indexers briefly
    lock a freshly written file."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1 * (attempt + 1))
