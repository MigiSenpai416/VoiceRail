"""The voice model library: one folder per model under ``data/models``."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from . import paths
from .rvc.checkpoint import ModelFormatError, read_voice_model

log = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 2 * 1024**3


@dataclass
class ModelInfo:
    name: str
    sample_rate: int
    version: str
    pitch_guided: bool
    epochs: int | None
    has_index: bool
    size_mb: float
    source: str
    created: float
    f0_median: float | None = None
    notes: str = ""

    @property
    def folder(self) -> Path:
        return paths.models_dir() / self.name

    def summary(self) -> str:
        sr = f"{self.sample_rate // 1000}k"
        index = "index" if self.has_index else "no index"
        epochs = f", {self.epochs} epochs" if self.epochs else ""
        return f"RVC {self.version} · {sr} · {index}{epochs}"


def _model_file(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.pth"))
    return files[0] if files else None


def _index_file(folder: Path) -> Path | None:
    files = sorted(folder.glob("*.index"))
    return files[0] if files else None


def _folder(name: str) -> Path:
    """The model folder, refusing names that escape the models directory."""
    root = paths.models_dir().resolve()
    folder = (root / name).resolve()
    if folder.parent != root:
        raise FileNotFoundError(f"Voice model '{name}' was not found.")
    return folder


def model_paths(name: str) -> tuple[Path, Path | None]:
    folder = _folder(name)
    model = _model_file(folder)
    if model is None:
        raise FileNotFoundError(f"Voice model '{name}' was not found.")
    return model, _index_file(folder)


def _read_info(folder: Path) -> ModelInfo | None:
    model = _model_file(folder)
    if model is None:
        return None
    info_path = folder / "info.json"
    if info_path.is_file():
        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
            data["name"] = folder.name
            data["has_index"] = _index_file(folder) is not None
            known = {
                k: v for k, v in data.items() if k in ModelInfo.__dataclass_fields__
            }
            return ModelInfo(**known)
        except (json.JSONDecodeError, TypeError):
            log.warning("Rebuilding damaged info.json in %s", folder)
    return _write_info(folder, source="imported")


def _write_info(folder: Path, source: str, notes: str = "") -> ModelInfo:
    model_path = _model_file(folder)
    model = read_voice_model(model_path)
    epoch = model.metadata.get("epoch")
    if epoch is None:
        match = re.match(r"(\d+)\s*epoch", str(model.metadata.get("info", "")))
        epoch = int(match.group(1)) if match else None
    extra = model.metadata.get("voicerail")
    f0_median = extra.get("f0_median") if isinstance(extra, dict) else None
    info = ModelInfo(
        name=folder.name,
        sample_rate=model.sample_rate,
        version=model.version,
        pitch_guided=model.use_f0,
        epochs=int(epoch) if epoch is not None else None,
        has_index=_index_file(folder) is not None,
        size_mb=round(model_path.stat().st_size / 1024**2, 1),
        source=source,
        created=time.time(),
        f0_median=f0_median,
        notes=notes,
    )
    data = asdict(info)
    (folder / "info.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return info


def list_models() -> list[ModelInfo]:
    models = []
    for folder in sorted(paths.models_dir().iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        try:
            info = _read_info(folder)
        except ModelFormatError as error:
            log.warning("Skipping %s: %s", folder.name, error)
            continue
        if info:
            models.append(info)
    return models


def model_names() -> list[str]:
    return [m.name for m in list_models()]


def get_info(name: str) -> ModelInfo:
    info = _read_info(_folder(name))
    if info is None:
        raise FileNotFoundError(f"Voice model '{name}' was not found.")
    return info


def _unique_name(name: str) -> str:
    base = paths.safe_name(name, "voice")
    candidate, n = base, 2
    while (paths.models_dir() / candidate).exists():
        candidate = f"{base} ({n})"
        n += 1
    return candidate


def _candidate_score(path: Path) -> tuple[int, int]:
    """Prefer exported models over raw G_/D_ training checkpoints."""
    raw = bool(re.match(r"^[GD]_\d+", path.name))
    return (0 if raw else 1, path.stat().st_size if not raw else 0)


def _safe_extract(archive: Path, target: Path) -> list[Path]:
    """Extract only .pth/.index members, flattening paths (no zip-slip)."""
    out = []
    with zipfile.ZipFile(archive) as zf:
        total = 0
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.lower().endswith((".pth", ".index")):
                continue
            total += member.file_size
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("The archive is too large.")
            dest = target / paths.safe_name(name, "file")
            counter = 2
            while dest.exists():
                stem, suffix = Path(name).stem, Path(name).suffix
                dest = target / paths.safe_name(f"{stem} ({counter}){suffix}", "file")
                counter += 1
            with zf.open(member) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(dest)
    return out


def import_files(files: list[Path], name: str | None = None) -> ModelInfo:
    """Import a voice model from .pth / .index / .zip files."""
    with tempfile.TemporaryDirectory(dir=paths.tmp_dir()) as tmp:
        tmp_dir = Path(tmp)
        pths: list[Path] = []
        indexes: list[Path] = []
        for file in files:
            file = Path(file)
            suffix = file.suffix.lower()
            if suffix == ".zip":
                for extracted in _safe_extract(file, tmp_dir):
                    (pths if extracted.suffix.lower() == ".pth" else indexes).append(
                        extracted
                    )
            elif suffix == ".pth":
                pths.append(file)
            elif suffix == ".index":
                indexes.append(file)
            else:
                raise ValueError(f"Unsupported file type: {file.name}")

        model_path, errors = None, []
        for candidate in sorted(pths, key=_candidate_score, reverse=True):
            try:
                read_voice_model(candidate)
                model_path = candidate
                break
            except ModelFormatError as error:
                errors.append(str(error))
        if model_path is None:
            detail = errors[0] if errors else "no .pth model file was provided"
            raise ModelFormatError(f"No usable RVC model found: {detail}")

        index_path = _pick_index(indexes)
        stem = model_path.stem
        final = _unique_name(name or stem)
        folder = paths.models_dir() / final
        folder.mkdir(parents=True)
        try:
            shutil.copy2(model_path, folder / f"{final}.pth")
            if index_path:
                shutil.copy2(index_path, folder / f"{final}.index")
            return _write_info(folder, source="imported")
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise


def _pick_index(indexes: list[Path]) -> Path | None:
    """RVC exports both trained_*.index and added_*.index; use the added one."""
    if not indexes:
        return None
    added = [p for p in indexes if "added" in p.name.lower()]
    pool = added or [p for p in indexes if "trained" not in p.name.lower()] or indexes
    return max(pool, key=lambda p: p.stat().st_size)


def _normalise_url(url: str) -> str:
    url = url.strip()
    if "huggingface.co" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
    return url


def import_url(url: str, name: str | None = None, progress=None) -> ModelInfo:
    url = _normalise_url(url)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http(s) links are supported.")
    filename = Path(unquote(parsed.path)).name or "model.zip"
    if not filename.lower().endswith((".zip", ".pth")):
        filename += ".zip"
    with tempfile.TemporaryDirectory(dir=paths.tmp_dir()) as tmp:
        target = Path(tmp) / paths.safe_name(filename, "model.zip")
        with requests.get(url, stream=True, timeout=30, allow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0)
            if total > MAX_DOWNLOAD_BYTES:
                raise ValueError("The file is too large.")
            done = 0
            with target.open("wb") as handle:
                for block in resp.iter_content(1 << 20):
                    done += len(block)
                    if done > MAX_DOWNLOAD_BYTES:
                        raise ValueError("The file is too large.")
                    handle.write(block)
                    if progress and total:
                        progress(done / total, f"Downloading {filename}")
        if target.suffix.lower() == ".zip" and not zipfile.is_zipfile(target):
            raise ValueError(
                "The link did not return a model file. Use a direct download link."
            )
        return import_files([target], name or Path(filename).stem)


def rename_model(old: str, new: str) -> str:
    source = _folder(old)
    if not source.is_dir():
        raise FileNotFoundError(f"Voice model '{old}' was not found.")
    new = paths.safe_name(new, old)
    if new == old:
        return old
    new = _unique_name(new)
    target = paths.models_dir() / new
    source.rename(target)
    for file in list(target.glob("*.pth")) + list(target.glob("*.index")):
        file.rename(target / f"{new}{file.suffix}")
    info = target / "info.json"
    if info.is_file():
        data = json.loads(info.read_text(encoding="utf-8"))
        data["name"] = new
        info.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return new


def delete_model(name: str) -> None:
    folder = _folder(name)
    if folder.is_dir():
        shutil.rmtree(folder)


def publish_trained_model(
    name: str, model: Path, index: Path | None, exp_dir: Path
) -> Path:
    """Copy a finished training result into the library (replacing it)."""
    final = paths.safe_name(name, "voice")
    folder = paths.models_dir() / final
    staging = paths.models_dir() / f".{final}.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    shutil.copy2(model, staging / f"{final}.pth")
    if index is not None:
        shutil.copy2(index, staging / f"{final}.index")
    shutil.rmtree(folder, ignore_errors=True)
    staging.rename(folder)
    _write_info(folder, source="trained", notes=f"Trained in {exp_dir.name}")
    return folder
