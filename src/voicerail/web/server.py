"""HTTP API and static file server behind the VoiceRail web app.

The React frontend (built into ``static/``) talks to the ``/api/...`` routes
below. Long operations (conversion, downloads) run as background *tasks*
whose progress is streamed with server-sent events; training keeps running
in its own process through :mod:`voicerail.jobs`.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, assets, jobs, library, paths
from ..assets import DEFAULT_PRETRAINED, PRETRAINED_SETS
from ..audio.io import AUDIO_EXTENSIONS, OUTPUT_FORMATS
from ..audio.separation import (
    DENOISE_MODELS,
    DEREVERB_MODELS,
    LEAD_MODELS,
    VOCAL_MODELS,
)
from ..convert import ConvertRequest, convert
from ..device import describe
from ..rvc.pipeline import ConversionSettings
from ..train.config import SAMPLE_RATES, TrainConfig
from ..train.dataset import CleaningOptions, SegmentOptions, load_report
from ..train.job import STAGES, job_file, write_job

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
FULL_PIPELINE = ["prepare", "features", "train", "index", "publish"]
TASK_TTL_SECONDS = 3600

TRAIN_PRESETS = [
    {
        "key": "songs",
        "label": "Songs",
        "description": "Vocals with music. The music, backing vocals and reverb "
        "are removed before training.",
        "cleaning": dict(
            isolate_vocals=True,
            remove_backing_vocals=True,
            dereverb=True,
            denoise=False,
        ),
    },
    {
        "key": "clean",
        "label": "Clean voice",
        "description": "Studio vocals, podcasts, audiobooks. Used as they are.",
        "cleaning": dict(
            isolate_vocals=False,
            remove_backing_vocals=False,
            dereverb=False,
            denoise=False,
        ),
    },
    {
        "key": "noisy",
        "label": "Noisy or echoey voice",
        "description": "Phone, video or stream recordings. Background, echo and "
        "noise are removed first.",
        "cleaning": dict(
            isolate_vocals=True,
            remove_backing_vocals=False,
            dereverb=True,
            denoise=True,
        ),
    },
]

app = FastAPI(title="VoiceRail", version=__version__, docs_url=None, redoc_url=None)


# ------------------------------------------------------------------ helpers
def _fail(error: Exception, status: int = 400) -> HTTPException:
    log.error("%s: %s", error.__class__.__name__, error)
    return HTTPException(status, str(error) or error.__class__.__name__)


def _file_url(path: Path | str | None) -> str | None:
    if not path:
        return None
    return f"/api/files?path={quote(str(path))}"


def _inside_home(path: Path) -> Path:
    home = paths.home()
    resolved = path.resolve()
    if home != resolved and home not in resolved.parents:
        raise HTTPException(403, "Only files inside the data folder can be served.")
    return resolved


def _save_upload(upload: UploadFile, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / paths.safe_name(Path(upload.filename or "audio").name, "audio")
    with target.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return target


def _upload_dir() -> Path:
    return paths.tmp_dir() / "uploads" / uuid.uuid4().hex


def _known(cls, data: dict) -> dict:
    return {k: v for k, v in data.items() if k in cls.__dataclass_fields__}


def _models_payload(models) -> list[dict]:
    return [{**asdict(m), "summary": m.summary()} for m in models]


# -------------------------------------------------------------------- tasks
@dataclass
class Task:
    id: str
    kind: str
    state: str = "running"  # running | finished | failed
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    updated: float = field(default_factory=time.time)


_tasks: dict[str, Task] = {}
_tasks_lock = threading.Lock()


def _start_task(kind: str, work: Callable[[Callable[[float, str], None]], Any]) -> Task:
    task = Task(uuid.uuid4().hex, kind)
    now = time.time()
    with _tasks_lock:
        for old in [
            t
            for t in _tasks.values()
            if t.state != "running" and now - t.updated > TASK_TTL_SECONDS
        ]:
            del _tasks[old.id]
        _tasks[task.id] = task

    def progress(fraction: float, message: str) -> None:
        task.progress = max(0.0, min(1.0, float(fraction)))
        task.message = message
        task.updated = time.time()

    def run() -> None:
        try:
            task.result = work(progress)
            task.state, task.progress = "finished", 1.0
        except Exception as error:
            log.error("Task %s failed: %s", kind, error, exc_info=True)
            task.error = str(error) or error.__class__.__name__
            task.state = "failed"
        task.updated = time.time()

    threading.Thread(target=run, name=f"task-{kind}", daemon=True).start()
    return task


def _task_payload(task: Task) -> dict:
    return asdict(task)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "Unknown task.")
    return _task_payload(task)


@app.get("/api/tasks/{task_id}/events")
def task_events(task_id: str) -> StreamingResponse:
    task = _tasks.get(task_id)
    if task is None:
        raise HTTPException(404, "Unknown task.")

    def stream():
        last = None
        while True:
            payload = json.dumps(_task_payload(task), default=str)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if task.state != "running":
                return
            time.sleep(0.25)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------- system
@app.get("/api/system")
def system() -> dict:
    return {
        "version": __version__,
        "device": asdict(describe()),
        "home": str(paths.home()),
    }


@app.get("/api/options")
def options() -> dict:
    """Defaults and choices the UI builds its forms from."""
    request_defaults = asdict(
        ConvertRequest(input_path=Path(), model_name="", settings=ConversionSettings())
    )
    for key in ("input_path", "model_name", "settings"):
        request_defaults.pop(key)
    return {
        "convert": {
            "settings": asdict(ConversionSettings()),
            "request": request_defaults,
            "vocal_models": [asdict(m) for m in VOCAL_MODELS],
            "lead_models": [asdict(m) for m in LEAD_MODELS],
            "dereverb_models": [asdict(m) for m in DEREVERB_MODELS],
            "denoise_models": [asdict(m) for m in DENOISE_MODELS],
            "formats": list(OUTPUT_FORMATS),
            "audio_extensions": sorted(AUDIO_EXTENSIONS),
        },
        "train": {
            "config": TrainConfig().to_json(),
            "cleaning": asdict(CleaningOptions()),
            "segmenting": asdict(SegmentOptions()),
            "presets": TRAIN_PRESETS,
            "sample_rates": list(SAMPLE_RATES),
            "pretrained": list(PRETRAINED_SETS),
            "default_pretrained": DEFAULT_PRETRAINED,
            "stages": STAGES,
        },
    }


@app.get("/api/files")
def get_file(path: str, download: bool = False) -> FileResponse:
    target = _inside_home(Path(path))
    if not target.is_file():
        raise HTTPException(404, "File not found.")
    return FileResponse(
        target,
        filename=target.name if download else None,
        content_disposition_type="attachment" if download else "inline",
    )


# ------------------------------------------------------------------- models
@app.get("/api/models")
def list_models() -> list[dict]:
    return _models_payload(library.list_models())


@app.post("/api/models/import")
def import_model(
    files: list[UploadFile] = File([]),
    url: str = Form(""),
    name: str = Form(""),
) -> dict:
    name = name.strip() or None
    if files:
        folder = _upload_dir()
        saved = [_save_upload(f, folder) for f in files]

        def work(progress):
            try:
                progress(0.1, "Importing")
                return _models_payload([library.import_files(saved, name)])[0]
            finally:
                shutil.rmtree(folder, ignore_errors=True)

    elif url.strip():

        def work(progress):
            return _models_payload([library.import_url(url, name, progress)])[0]

    else:
        raise HTTPException(400, "Choose model files or paste a download link.")
    return _task_payload(_start_task("import", work))


@app.patch("/api/models/{name}")
def rename_model(name: str, body: dict) -> dict:
    new = str(body.get("name", "")).strip()
    if not new:
        raise HTTPException(400, "Type a new name.")
    try:
        final = library.rename_model(name, new)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise _fail(error) from error
    return {"name": final}


@app.delete("/api/models/{name}")
def delete_model(name: str) -> dict:
    try:
        library.delete_model(name)
    except (FileNotFoundError, OSError) as error:
        raise _fail(error) from error
    return {"ok": True}


@app.get("/api/models/{name}/files")
def model_files(name: str) -> list[dict]:
    try:
        model, index = library.model_paths(name)
    except FileNotFoundError as error:
        raise _fail(error, 404) from error
    files = [model] + ([index] if index else [])
    return [
        {
            "name": f.name,
            "url": _file_url(f) + "&download=1",
            "mb": f.stat().st_size / 1024**2,
        }
        for f in files
    ]


# ------------------------------------------------------------------ convert
@app.post("/api/convert")
def start_convert(audio: UploadFile = File(...), settings: str = Form("{}")) -> dict:
    try:
        data = json.loads(settings or "{}")
    except json.JSONDecodeError as error:
        raise _fail(error) from error
    model_name = str(data.get("model") or "")
    if not model_name:
        raise HTTPException(400, "Select a voice first.")
    folder = _upload_dir()
    source = _save_upload(audio, folder)
    request = ConvertRequest(
        input_path=source,
        model_name=model_name,
        settings=ConversionSettings(
            **_known(ConversionSettings, data.get("settings") or {})
        ),
        **_known(
            ConvertRequest,
            {k: v for k, v in data.items() if k not in ("settings", "model")},
        ),
    )

    def work(progress):
        try:
            result = convert(request, progress)
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        return {
            "output": _file_url(result.output),
            "output_name": result.output.name,
            "converted_vocals": _file_url(result.converted_vocals),
            "instrumental": _file_url(result.instrumental),
            "backing": _file_url(result.backing),
            "source_vocals": _file_url(result.source_vocals),
            "pitch_shift": result.pitch_shift,
            "notes": result.notes,
        }

    return _task_payload(_start_task("convert", work))


# ----------------------------------------------------------------- training
def _session_summary(name: str) -> dict:
    st = jobs.status(name)
    return {"name": name, **asdict(st)}


def _session_detail(name: str) -> dict:
    exp = jobs.experiment_dir(name)
    if not job_file(exp).is_file():
        raise HTTPException(404, "Unknown training session.")
    previews = sorted((exp / "samples").glob("epoch_*.wav"))
    reference = exp / "samples" / "reference.wav"
    clips = sorted((exp / "dataset" / "wavs").glob("*.wav"))
    report = load_report(exp)
    src = exp / "sources"
    sources = [p for p in src.glob("*") if p.is_file()] if src.is_dir() else []
    try:
        spec = json.loads(job_file(exp).read_text())
    except (OSError, json.JSONDecodeError):
        spec = None
    return {
        **_session_summary(name),
        "metrics": jobs.metrics(name),
        "report": asdict(report) if report else None,
        "samples": {
            "preview": _file_url(previews[-1]) if previews else None,
            "preview_epoch": int(previews[-1].stem.split("_")[-1])
            if previews
            else None,
            "reference": _file_url(reference) if reference.is_file() else None,
            "cleaned": _file_url(clips[len(clips) // 2]) if clips else None,
        },
        "log": jobs.tail_log(name),
        "sources": {
            "count": len(sources),
            "mb": sum(p.stat().st_size for p in sources) / 1024**2,
        },
        "job": spec,
    }


@app.get("/api/train/sessions")
def list_sessions() -> list[dict]:
    names = jobs.experiments()
    latest = jobs.latest_experiment()
    return [{**_session_summary(n), "latest": n == latest} for n in names]


@app.get("/api/train/sessions/{name}")
def get_session(name: str) -> dict:
    return _session_detail(name)


@app.post("/api/train")
def start_train(
    name: str = Form(...),
    settings: str = Form("{}"),
    files: list[UploadFile] = File([]),
) -> dict:
    try:
        data = json.loads(settings or "{}")
        jobs.ensure_idle()
    except (json.JSONDecodeError, RuntimeError) as error:
        raise _fail(error) from error
    voice_name = name.strip()
    if not voice_name:
        raise HTTPException(400, "Give the voice a name first.")
    safe = paths.safe_name(voice_name, "voice")
    exp = jobs.experiment_dir(safe)
    src_dir = exp / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    for upload in files:
        _save_upload(upload, src_dir)
    sources = [str(src_dir)]
    folder = str(data.get("folder") or "").strip().strip('"')
    if folder:
        folder_dir = Path(folder).expanduser()
        if not folder_dir.is_dir():
            raise HTTPException(400, f"Folder not found: {folder_dir}")
        sources.append(str(folder_dir))
    stages = ["prepare"] if data.get("stages") == "prepare" else FULL_PIPELINE
    if "prepare" in stages and not any(
        p.is_file() for s in sources for p in Path(s).rglob("*")
    ):
        raise HTTPException(400, "Add some recordings (upload files or give a folder).")
    cleaning = CleaningOptions(**_known(CleaningOptions, data.get("cleaning") or {}))
    segmenting = SegmentOptions(**_known(SegmentOptions, data.get("segmenting") or {}))
    if segmenting.min_seconds >= segmenting.max_seconds:
        raise HTTPException(
            400, "The shortest segment must be shorter than the longest."
        )
    train = TrainConfig.from_json(data.get("train") or {})
    write_job(
        exp,
        safe,
        sources,
        cleaning,
        segmenting,
        train,
        stages,
        fresh=bool(data.get("fresh")) and "train" in stages,
    )
    try:
        jobs.start(safe)
    except RuntimeError as error:
        raise _fail(error) from error
    return _session_detail(safe)


@app.post("/api/train/sessions/{name}/stop")
def stop_session(name: str) -> dict:
    if not jobs.status(name).active:
        raise HTTPException(400, "Nothing is running.")
    jobs.request_stop(name)
    return {"ok": True}


@app.post("/api/train/sessions/{name}/publish")
def publish_session(name: str) -> dict:
    exp = jobs.experiment_dir(name)
    if jobs.status(name).active:
        raise HTTPException(
            400, "Training is running; stop it first (it saves on stop)."
        )
    ckpt = exp / "checkpoints"
    if not (ckpt.is_dir() and any(ckpt.glob("model_*.pth"))):
        raise HTTPException(400, "This session has no trained model yet.")
    try:
        jobs.ensure_idle()
        spec = json.loads(job_file(exp).read_text())
        spec["stages"] = ["index", "publish"]
        spec["fresh"] = False
        job_file(exp).write_text(json.dumps(spec, indent=2))
        jobs.start(name)
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        raise _fail(error) from error
    return _session_detail(name)


@app.delete("/api/train/sessions/{name}")
def delete_session(name: str) -> dict:
    if jobs.status(name).active:
        raise HTTPException(400, "This session is training right now; stop it first.")
    shutil.rmtree(jobs.experiment_dir(name), ignore_errors=True)
    return {"ok": True}


# -------------------------------------------------------- assets and storage
@app.get("/api/assets")
def list_assets() -> list[dict]:
    return [
        {"key": key, "mb": asset.size / 1024**2, "ready": assets.is_ready(key)}
        for key, asset in assets.ASSETS.items()
    ]


@app.post("/api/assets/download")
def download_assets() -> dict:
    keys = [*assets.CORE_ASSETS, *PRETRAINED_SETS[DEFAULT_PRETRAINED][40000]]
    return _task_payload(
        _start_task(
            "assets", lambda progress: [str(p) for p in assets.ensure(keys, progress)]
        )
    )


def _folder_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total / 1024**3


@app.get("/api/storage")
def storage() -> list[dict]:
    parts = {
        "Voices": paths.models_dir(),
        "Training sessions": paths.training_dir(),
        "Converted outputs": paths.outputs_dir(),
        "Cache": paths.cache_dir(),
        "Base & separation models": paths.assets_dir(),
    }
    return [{"label": k, "path": str(v), "gb": _folder_gb(v)} for k, v in parts.items()]


@app.post("/api/cache/clear")
def clear_cache() -> dict:
    shutil.rmtree(paths.cache_dir() / "convert", ignore_errors=True)
    return {"ok": True}


# ------------------------------------------------------------------- static
if (STATIC / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")


@app.get("/{path:path}")
def frontend(path: str) -> FileResponse:
    index = STATIC / "index.html"
    if not index.is_file():
        raise HTTPException(
            503, "The web app has not been built: run `npm run build` in frontend/."
        )
    if path and ".." not in path:
        file = STATIC / path
        if file.is_file():
            return FileResponse(file)
    return FileResponse(index)


def launch(
    host: str = "127.0.0.1", port: int = 9999, open_browser: bool = True
) -> None:
    import uvicorn

    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{port}"
    print(f"VoiceRail is running at {url}", flush=True)
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
