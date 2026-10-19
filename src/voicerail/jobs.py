"""Launch and monitor training jobs running in background processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .train.job import log_file, status_file, stop_file

STALE_AFTER_SECONDS = 90

# name -> (process, launch token) for workers started by this UI process.
_processes: dict[str, tuple[subprocess.Popen, str]] = {}
_lock = threading.Lock()
_start_lock = threading.Lock()


@dataclass
class JobStatus:
    state: str  # idle | running | finished | stopped | failed | crashed
    stages: list[str]
    stage: str | None
    stage_index: int
    stage_count: int
    progress: float
    message: str
    error: str | None
    started: float | None
    updated: float | None
    result: dict | None
    last_metrics: dict | None

    @property
    def active(self) -> bool:
        return self.state == "running"


IDLE = JobStatus("idle", [], None, 0, 0, 0.0, "", None, None, None, None, None)


def experiments() -> list[str]:
    root = paths.training_dir()
    return sorted(
        (p.name for p in root.iterdir() if p.is_dir() and (p / "job.json").is_file()),
        key=str.lower,
    )


def latest_experiment() -> str | None:
    """The running session, else the most recently started one."""
    running = running_job()
    if running:
        return running
    names = experiments()
    if not names:
        return None
    return max(names, key=lambda n: (experiment_dir(n) / "job.json").stat().st_mtime)


def experiment_dir(name: str) -> Path:
    return paths.training_dir() / paths.safe_name(name, "voice")


def _alive(name: str, token: str | None) -> bool | None:
    """Liveness of the worker that wrote ``token``, if we launched it."""
    with _lock:
        tracked = _processes.get(name)
    if tracked is None or token is None or tracked[1] != token:
        return None
    return tracked[0].poll() is None


def status(name: str) -> JobStatus:
    path = status_file(experiment_dir(name))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return IDLE
    state = data.get("state", "idle")
    alive = _alive(name, data.get("token"))
    if state == "running":
        stale = time.time() - float(data.get("updated") or 0) > STALE_AFTER_SECONDS
        if alive is False or (alive is None and stale):
            state = "crashed"
    return JobStatus(
        state=state,
        stages=list(data.get("stages") or []),
        stage=data.get("stage"),
        stage_index=int(data.get("stage_index") or 0),
        stage_count=len(data.get("stages") or []),
        progress=float(data.get("progress") or 0.0),
        message=data.get("message") or "",
        error=data.get("error")
        or (_crash_reason(name) if state == "crashed" else None),
        started=data.get("started"),
        updated=data.get("updated"),
        result=data.get("result"),
        last_metrics=data.get("last_metrics"),
    )


def _crash_reason(name: str) -> str:
    worker_log = experiment_dir(name) / "worker.log"
    try:
        lines = worker_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "The training process stopped unexpectedly."
    tail = [line for line in lines if line.strip()][-6:]
    return "The training process stopped unexpectedly:\n" + "\n".join(tail)


def running_job() -> str | None:
    for name in experiments():
        if status(name).active:
            return name
    return None


def start(name: str) -> None:
    """Launch the job described by the session's ``job.json``.

    Callers must write ``job.json`` only after :func:`ensure_idle`.
    """
    with _start_lock:
        ensure_idle()
        _launch(name)


def ensure_idle() -> None:
    busy = running_job()
    with _lock:
        tracked = [n for n, (proc, _) in _processes.items() if proc.poll() is None]
    busy = busy or (tracked[0] if tracked else None)
    if busy:
        raise RuntimeError(f"Training '{busy}' is still running. Stop it first.")


def _launch(name: str) -> None:
    exp_dir = experiment_dir(name)
    stop_file(exp_dir).unlink(missing_ok=True)
    token = uuid.uuid4().hex
    env = dict(os.environ)
    env.update(
        VOICERAIL_HOME=str(paths.home()),
        VOICERAIL_JOB_TOKEN=token,
        TQDM_DISABLE="1",
        PYTHONUNBUFFERED="1",
        PYTHONIOENCODING="utf-8",
    )
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with (exp_dir / "worker.log").open("w", encoding="utf-8") as worker_log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "voicerail.train.job", str(exp_dir)],
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            **kwargs,
        )
    with _lock:
        _processes[name] = (proc, token)
    # Wait until the worker reports in, so the UI immediately sees it running
    # and a worker that dies on start-up is reported instead of ignored.
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            data = json.loads(status_file(exp_dir).read_text(encoding="utf-8"))
            if data.get("token") == token:
                return
        except (OSError, json.JSONDecodeError):
            pass
        if proc.poll() is not None:
            raise RuntimeError(_crash_reason(name))
        time.sleep(0.2)


def request_stop(name: str) -> None:
    stop_file(experiment_dir(name)).touch()


def tail_log(name: str, lines: int = 200) -> str:
    path = log_file(experiment_dir(name))
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 64 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def metrics(name: str) -> list[dict]:
    path = experiment_dir(name) / "metrics.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
