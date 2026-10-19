"""Base model registry and a resumable, checksummed downloader."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from . import paths

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class Asset:
    key: str
    repo: str
    filename: str
    size: int
    sha256: str | None = None

    @property
    def url(self) -> str:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        return f"{endpoint}/{self.repo}/resolve/main/{self.filename}"

    @property
    def path(self) -> Path:
        return paths.assets_dir() / self.key


_RVC = "lj1995/VoiceConversionWebUI"
_TITAN = "blaise-tk/TITAN"

ASSETS: dict[str, Asset] = {
    a.key: a
    for a in [
        Asset(
            "rmvpe.pt",
            _RVC,
            "rmvpe.pt",
            181184272,
            "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
        ),
        Asset("contentvec/config.json", _RVC, "hubert_base/config.json", 1492),
        Asset(
            "contentvec/pytorch_model.bin",
            _RVC,
            "hubert_base/pytorch_model.bin",
            189206711,
            "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5",
        ),
        Asset(
            "pretrained/rvc_v2/f0G32k.pth",
            _RVC,
            "pretrained_v2/f0G32k.pth",
            73950049,
            "2332611297b8d88c7436de8f17ef5f07a2119353e962cd93cda5806d59a1133d",
        ),
        Asset(
            "pretrained/rvc_v2/f0D32k.pth",
            _RVC,
            "pretrained_v2/f0D32k.pth",
            142875703,
            "bd7134e7793674c85474d5145d2d982e3c5d8124fc7bb6c20f710ed65808fa8a",
        ),
        Asset(
            "pretrained/rvc_v2/f0G40k.pth",
            _RVC,
            "pretrained_v2/f0G40k.pth",
            73106273,
            "3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb",
        ),
        Asset(
            "pretrained/rvc_v2/f0D40k.pth",
            _RVC,
            "pretrained_v2/f0D40k.pth",
            142875703,
            "6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09",
        ),
        Asset(
            "pretrained/rvc_v2/f0G48k.pth",
            _RVC,
            "pretrained_v2/f0G48k.pth",
            75465569,
            "b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4",
        ),
        Asset(
            "pretrained/rvc_v2/f0D48k.pth",
            _RVC,
            "pretrained_v2/f0D48k.pth",
            142875703,
            "2269b73c7a4cf34da09aea99274dabf99b2ddb8a42cbfb065fb3c0aa9a2fc748",
        ),
        Asset(
            "pretrained/titan_medium/G32k.pth",
            _TITAN,
            "models/medium/32k/pretrained/G-f032k-TITAN-Medium.pth",
            443219306,
            "0f445603cc79b77ee1f40bd0f28c31760251ae40073abcb662ba9a544d4780b1",
        ),
        Asset(
            "pretrained/titan_medium/D32k.pth",
            _TITAN,
            "models/medium/32k/pretrained/D-f032k-TITAN-Medium.pth",
            857119946,
            "520dc7236ce87842a6c285e48d88ea0b92344e3654bab1e96c60fda5dcd12a59",
        ),
        Asset(
            "pretrained/titan_medium/G40k.pth",
            _TITAN,
            "models/medium/40k/pretrained/G-f040k-TITAN-Medium.pth",
            438156650,
            "217de2fd349f48856aa6a6d9d1257b07a0addccde7b69fc7b22097f1fee5924d",
        ),
        Asset(
            "pretrained/titan_medium/D40k.pth",
            _TITAN,
            "models/medium/40k/pretrained/D-f040k-TITAN-Medium.pth",
            857119946,
            "bc804da752ba9cb8e3aaa90ad102edc0a6e7f033c90db53bfcce135f100d5ad1",
        ),
        Asset(
            "pretrained/titan_medium/G48k.pth",
            _TITAN,
            "models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth",
            452338845,
            "51a6dc93687f7e1a6051be3dec958289958c9c738d5c520e4fa956e7886ee153",
        ),
        Asset(
            "pretrained/titan_medium/D48k.pth",
            _TITAN,
            "models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth",
            857126469,
            "4999a5b66aec0a9ab7e845063eeb242e47a611ab0e6260fb0b5ca44a7c5bbc44",
        ),
    ]
}

CORE_ASSETS = ("rmvpe.pt", "contentvec/config.json", "contentvec/pytorch_model.bin")

PRETRAINED_SETS: dict[str, dict[int, tuple[str, str]]] = {
    "RVC v2 (standard)": {
        sr: (f"pretrained/rvc_v2/f0G{k}.pth", f"pretrained/rvc_v2/f0D{k}.pth")
        for sr, k in ((32000, "32k"), (40000, "40k"), (48000, "48k"))
    },
    "TITAN Medium (cleaner, larger download)": {
        sr: (f"pretrained/titan_medium/G{k}.pth", f"pretrained/titan_medium/D{k}.pth")
        for sr, k in ((32000, "32k"), (40000, "40k"), (48000, "48k"))
    },
}
DEFAULT_PRETRAINED = "RVC v2 (standard)"


class DownloadError(RuntimeError):
    pass


def is_ready(key: str) -> bool:
    asset = ASSETS[key]
    return asset.path.is_file() and asset.path.stat().st_size == asset.size


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(key: str, progress: ProgressFn | None = None, retries: int = 5) -> Path:
    """Fetch an asset if missing. Interrupted downloads resume."""
    asset = ASSETS[key]
    if is_ready(key):
        return asset.path
    asset.path.parent.mkdir(parents=True, exist_ok=True)
    partial = asset.path.with_name(asset.path.name + ".part")
    label = f"Downloading {asset.key}"

    for attempt in range(1, retries + 1):
        try:
            _fetch(asset, partial, label, progress)
            break
        except (requests.RequestException, OSError) as error:
            if attempt == retries:
                raise DownloadError(
                    f"Failed to download {asset.key} from {asset.url}: {error}"
                ) from error
            wait = 2**attempt
            log.warning("%s failed (%s); retrying in %ss", label, error, wait)
            time.sleep(wait)

    if partial.stat().st_size != asset.size:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"{asset.key}: downloaded size does not match.")
    if asset.sha256 and _sha256(partial) != asset.sha256:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"{asset.key}: checksum mismatch, file discarded.")
    paths.replace_file(partial, asset.path)
    return asset.path


def _fetch(asset: Asset, partial: Path, label: str, progress: ProgressFn | None):
    done = partial.stat().st_size if partial.exists() else 0
    if done >= asset.size:
        return
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(asset.url, headers=headers, stream=True, timeout=30) as resp:
        if resp.status_code == 416:
            partial.unlink(missing_ok=True)
            raise requests.RequestException("server rejected resume range")
        resp.raise_for_status()
        if done and resp.status_code != 206:
            done = 0
        mode = "ab" if done else "wb"
        last = 0.0
        with partial.open(mode) as handle:
            for block in resp.iter_content(chunk_size=1 << 20):
                handle.write(block)
                done += len(block)
                now = time.monotonic()
                if progress and now - last > 0.5:
                    last = now
                    progress(min(done / asset.size, 1.0), label)


def ensure(keys: tuple[str, ...] | list[str], progress: ProgressFn | None = None):
    for key in keys:
        download(key, progress)
    return [ASSETS[k].path for k in keys]


def ensure_core(progress: ProgressFn | None = None) -> None:
    ensure(CORE_ASSETS, progress)


def pretrained_paths(
    name: str, sample_rate: int, progress: ProgressFn | None = None
) -> tuple[Path, Path]:
    try:
        g_key, d_key = PRETRAINED_SETS[name][sample_rate]
    except KeyError as error:
        raise ValueError(
            f"No '{name}' pretrained model for {sample_rate} Hz."
        ) from error
    g, d = ensure((g_key, d_key), progress)
    return g, d


def contentvec_dir() -> Path:
    return paths.assets_dir() / "contentvec"


def rmvpe_path() -> Path:
    return ASSETS["rmvpe.pt"].path
