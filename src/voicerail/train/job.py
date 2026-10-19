"""End-to-end training job, executed in its own process.

``python -m voicerail.train.job <experiment dir>`` runs every stage described
in ``job.json`` and keeps ``status.json`` up to date (with a heartbeat) so the
web UI can follow along, survive page reloads and request a graceful stop by
creating a ``STOP`` file.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from .. import paths
from .config import TrainConfig
from .dataset import (
    CleaningOptions,
    SegmentOptions,
    dataset_dir,
    find_audio_files,
    prepare_dataset,
)
from .progress import Reporter, StopRequested

log = logging.getLogger("voicerail.job")

STAGES = {
    "prepare": "Cleaning and slicing audio",
    "features": "Extracting pitch and voice features",
    "train": "Training",
    "index": "Building the retrieval index",
    "publish": "Publishing the voice model",
}
HEARTBEAT_SECONDS = 5


def job_file(exp_dir: Path) -> Path:
    return exp_dir / "job.json"


def status_file(exp_dir: Path) -> Path:
    return exp_dir / "status.json"


def stop_file(exp_dir: Path) -> Path:
    return exp_dir / "STOP"


def log_file(exp_dir: Path) -> Path:
    return exp_dir / "log.txt"


def write_job(
    exp_dir: Path,
    name: str,
    sources: list[str],
    cleaning: CleaningOptions,
    segmenting: SegmentOptions,
    train: TrainConfig,
    stages: list[str],
    fresh: bool = False,
) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    segmenting.sample_rate = train.sample_rate
    spec = {
        "name": name,
        "sources": sources,
        "cleaning": asdict(cleaning),
        "segmenting": asdict(segmenting),
        "train": train.to_json(),
        "stages": stages,
        "fresh": fresh,
        "created": time.time(),
    }
    job_file(exp_dir).write_text(json.dumps(spec, indent=2))


class StatusWriter:
    def __init__(self, exp_dir: Path, stages: list[str]):
        self.path = status_file(exp_dir)
        self.lock = threading.Lock()
        self.data = {
            "state": "running",
            "pid": os.getpid(),
            "token": os.environ.get("VOICERAIL_JOB_TOKEN"),
            "stages": stages,
            "stage": None,
            "stage_index": 0,
            "progress": 0.0,
            "message": "Starting",
            "started": time.time(),
            "updated": time.time(),
            "error": None,
            "result": None,
            "last_metrics": None,
        }
        self._last_flush = 0.0
        self._stop = threading.Event()
        self.flush()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def update(self, **values) -> None:
        """Record new values; frequent progress-only updates are batched."""
        with self.lock:
            self.data.update(values)
        progress_only = set(values) <= {"progress", "message"}
        if not progress_only or time.time() - self._last_flush > 0.5:
            self.flush()

    def flush(self) -> None:
        # One lock around serialise + write + replace so the heartbeat thread
        # and the main thread never interleave (or reorder) their writes.
        with self.lock:
            self.data["updated"] = time.time()
            payload = json.dumps(self.data, indent=1, default=str)
            tmp = self.path.with_suffix(".tmp")
            for _ in range(20):
                try:
                    tmp.write_text(payload, encoding="utf-8")
                    tmp.replace(self.path)
                    break
                except OSError:
                    # Windows refuses the replace while a reader has it open.
                    time.sleep(0.05)
            self._last_flush = time.time()

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            self.flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self.flush()


def _setup_logging(exp_dir: Path) -> None:
    handler = logging.FileHandler(log_file(exp_dir), encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler, console]
    root.setLevel(logging.INFO)
    for noisy in (
        "audio_separator",
        "httpx",
        "urllib3",
        "numba",
        "matplotlib",
        "faiss",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _options_key(spec: dict) -> str:
    """Identity of a prepared dataset: options plus the exact input files."""
    files = find_audio_files([Path(p) for p in spec["sources"]])
    inputs = [(str(p), p.stat().st_size, int(p.stat().st_mtime)) for p in files]
    return json.dumps(
        {
            "cleaning": spec["cleaning"],
            "segmenting": spec["segmenting"],
            "inputs": inputs,
        },
        sort_keys=True,
    )


def run_job(exp_dir: Path) -> int:
    exp_dir = exp_dir.resolve()
    spec = json.loads(job_file(exp_dir).read_text())
    stop_file(exp_dir).unlink(missing_ok=True)
    _setup_logging(exp_dir)
    stages = [s for s in spec["stages"] if s in STAGES]
    status = StatusWriter(exp_dir, stages)
    name = spec["name"]
    train_config = TrainConfig.from_json(spec["train"])
    cleaning = CleaningOptions(**spec["cleaning"])
    segmenting = SegmentOptions(**spec["segmenting"])
    segmenting.sample_rate = train_config.sample_rate

    def on_progress(fraction: float, message: str) -> None:
        status.update(progress=fraction, message=message)

    reporter = Reporter(
        on_progress=on_progress,
        stop_requested=lambda: stop_file(exp_dir).exists(),
        on_metrics=lambda m: status.update(last_metrics=m),
    )
    log.info("Job '%s' started: %s", name, ", ".join(stages))
    result: dict = {}
    try:
        for index, stage in enumerate(stages):
            reporter.check_stop()
            status.update(
                stage=stage, stage_index=index, progress=0.0, message=STAGES[stage]
            )
            log.info("== %s ==", STAGES[stage])
            result.update(
                _run_stage(
                    stage,
                    exp_dir,
                    spec,
                    name,
                    train_config,
                    cleaning,
                    segmenting,
                    reporter,
                )
                or {}
            )
            status.update(result=result)
        status.update(state="finished", progress=1.0, message="Done", result=result)
        log.info("Job finished")
        return 0
    except (StopRequested, KeyboardInterrupt):
        log.info("Job stopped by user")
        status.update(state="stopped", message="Stopped", result=result)
        return 0
    except Exception as error:
        log.error("Job failed: %s\n%s", error, traceback.format_exc())
        status.update(state="failed", error=str(error), message="Failed", result=result)
        return 1
    finally:
        stop_file(exp_dir).unlink(missing_ok=True)
        status.close()


def _run_stage(
    stage, exp_dir, spec, name, train_config, cleaning, segmenting, reporter
):
    from ..audio.separation import shared as separator
    from ..device import default_device
    from ..rvc import runtime
    from .features import clear_features, extract_features, load_summary

    device = default_device()
    if stage == "prepare":
        marker = dataset_dir(exp_dir) / "options.json"
        report_path = dataset_dir(exp_dir) / "report.json"
        key = _options_key(spec)
        if report_path.is_file() and marker.is_file() and marker.read_text() == key:
            log.info("Dataset already prepared with these options; reusing it.")
            return {}
        # Invalidate first: an interrupted run must never leave markers that
        # make stale features look valid next to half-written audio.
        marker.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        clear_features(exp_dir)
        sources = [Path(p) for p in spec["sources"]]
        report = prepare_dataset(sources, exp_dir, cleaning, segmenting, reporter)
        marker.write_text(key)
        clear_features(exp_dir)
        separator().unload()
        if report.total_segments == 0:
            raise ValueError(
                "No usable voice was found in the uploaded audio. Check the files "
                "or disable some cleaning steps."
            )
        for warning in report.warnings:
            log.warning(warning)
        return {
            "dataset_minutes": round(report.total_seconds / 60, 2),
            "segments": report.total_segments,
        }

    if stage == "features":
        summary = load_summary(exp_dir)
        report_path = dataset_dir(exp_dir) / "report.json"
        filelist = exp_dir / "features" / "filelist.json"
        up_to_date = (
            summary is not None
            and report_path.is_file()
            and filelist.stat().st_mtime >= report_path.stat().st_mtime
        )
        if up_to_date:
            log.info("Features are up to date.")
        else:
            summary = extract_features(
                exp_dir, train_config.sample_rate, reporter, device
            )
        runtime.release()
        return {"f0_median": summary.f0_median}

    if stage == "train":
        from .trainer import Trainer, checkpoints_dir

        summary = load_summary(exp_dir)
        if summary is None:
            raise ValueError("Features are missing; run feature extraction first.")
        if spec.get("fresh"):
            shutil.rmtree(checkpoints_dir(exp_dir), ignore_errors=True)
            (exp_dir / "metrics.jsonl").unlink(missing_ok=True)
            shutil.rmtree(exp_dir / "samples", ignore_errors=True)
            spec["fresh"] = False
            job_file(exp_dir).write_text(json.dumps(spec, indent=2))
        trainer = Trainer(exp_dir, name, train_config, summary, reporter, device)
        result = trainer.run()
        if result.interrupted:
            raise StopRequested
        return {
            "epochs": result.epochs_done,
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
        }

    if stage == "index":
        import numpy as np

        from ..rvc.index import build_index

        summary = load_summary(exp_dir)
        feats = [np.load(i.feats) for i in summary.items if not i.mute]
        reporter.progress(0.2, "Training the FAISS index")
        build_index(np.concatenate(feats).astype("float32"), exp_dir / "added.index")
        reporter.progress(1.0, "Index built")
        return {}

    if stage == "publish":
        from ..library import publish_trained_model
        from .trainer import checkpoints_dir

        best = checkpoints_dir(exp_dir) / "model_best.pth"
        if not best.is_file():
            best = checkpoints_dir(exp_dir) / "model_latest.pth"
        if not best.is_file():
            raise ValueError("No trained model found to publish.")
        index = exp_dir / "added.index"
        folder = publish_trained_model(
            name, best, index if index.is_file() else None, exp_dir
        )
        log.info("Voice model saved to %s", folder)
        return {"model_dir": str(folder)}
    raise ValueError(f"Unknown stage {stage}")


def main() -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    if len(sys.argv) != 2:
        print("usage: python -m voicerail.train.job <experiment-dir>")
        raise SystemExit(2)
    exp_dir = Path(sys.argv[1])
    if not exp_dir.is_absolute():
        exp_dir = paths.training_dir() / exp_dir
    raise SystemExit(run_job(exp_dir))


if __name__ == "__main__":
    main()
