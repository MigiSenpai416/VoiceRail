"""Command line interface: ``voicerail <command>``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PRESETS = {
    "songs": dict(isolate_vocals=True, remove_backing_vocals=True, dereverb=True),
    "clean": dict(isolate_vocals=False, remove_backing_vocals=False, dereverb=False),
    "noisy": dict(
        isolate_vocals=True, remove_backing_vocals=False, dereverb=True, denoise=True
    ),
}


def _progress(fraction: float, message: str) -> None:
    sys.stdout.write(f"\r[{fraction * 100:5.1f}%] {message[:70]:<70}")
    sys.stdout.flush()


def cmd_web(args) -> None:
    from .web.app import launch

    launch(args.host, args.port, args.share, not args.no_browser)


def cmd_convert(args) -> None:
    from .convert import ConvertRequest, convert
    from .rvc.pipeline import ConversionSettings

    request = ConvertRequest(
        input_path=Path(args.input),
        model_name=args.model,
        mode="voice" if args.voice else "song",
        settings=ConversionSettings(
            pitch_shift=args.pitch,
            index_rate=args.index_rate,
            protect=args.protect,
            rms_mix_rate=args.volume_envelope,
        ),
        auto_pitch=args.auto_pitch,
        dereverb=not args.no_dereverb,
        backing_vocals=args.backing,
        output_format=args.format,
    )
    result = convert(request, _progress)
    print(f"\nSaved: {result.output}")
    for note in result.notes:
        print(note)


def cmd_train(args) -> None:
    from . import jobs, paths
    from .train.config import TrainConfig
    from .train.dataset import CleaningOptions, SegmentOptions
    from .train.job import run_job, write_job

    busy = jobs.running_job()
    if busy:
        raise SystemExit(f"Training '{busy}' is still running. Stop it first.")
    name = paths.safe_name(args.name, "voice")
    exp = paths.training_dir() / name
    stages = (
        ["prepare"]
        if args.prepare_only
        else [
            "prepare",
            "features",
            "train",
            "index",
            "publish",
        ]
    )
    write_job(
        exp,
        name,
        [str(Path(p).resolve()) for p in args.data],
        CleaningOptions(**PRESETS[args.preset]),
        SegmentOptions(),
        TrainConfig(
            sample_rate=args.sample_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            precision=args.precision,
        ),
        stages,
        fresh=args.fresh,
    )
    raise SystemExit(run_job(exp))


def cmd_import(args) -> None:
    from . import library

    source = args.source
    if source.startswith(("http://", "https://")):
        info = library.import_url(source, args.name)
    else:
        info = library.import_files([Path(p) for p in source.split(",")], args.name)
    print(f"Imported {info.name}: {info.summary()}")


def cmd_models(_args) -> None:
    from . import library

    models = library.list_models()
    if not models:
        print("No voice models yet.")
    for m in models:
        print(f"{m.name:30} {m.summary()}")


def cmd_download(_args) -> None:
    from . import assets

    keys = [
        *assets.CORE_ASSETS,
        *assets.PRETRAINED_SETS[assets.DEFAULT_PRETRAINED][40000],
    ]
    assets.ensure(keys, _progress)
    print("\nBase models are ready.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voicerail", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    web = sub.add_parser("web", help="start the web UI (default)")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=None)
    web.add_argument("--share", action="store_true", help="create a public link")
    web.add_argument("--no-browser", action="store_true")
    web.set_defaults(func=cmd_web)

    conv = sub.add_parser("convert", help="convert a song or recording")
    conv.add_argument("input")
    conv.add_argument("-m", "--model", required=True)
    conv.add_argument("--voice", action="store_true", help="input has no music")
    conv.add_argument("-p", "--pitch", type=float, default=0.0)
    conv.add_argument("--auto-pitch", action="store_true")
    conv.add_argument("--index-rate", type=float, default=0.5)
    conv.add_argument("--protect", type=float, default=0.33)
    conv.add_argument("--volume-envelope", type=float, default=0.25)
    conv.add_argument("--no-dereverb", action="store_true")
    conv.add_argument(
        "--backing", choices=["keep", "convert", "remove"], default="keep"
    )
    conv.add_argument("-f", "--format", choices=["wav", "flac", "mp3"], default="wav")
    conv.set_defaults(func=cmd_convert)

    train = sub.add_parser("train", help="train a voice model")
    train.add_argument("name")
    train.add_argument("--data", nargs="+", required=True, help="files or folders")
    train.add_argument("--preset", choices=list(PRESETS), default="songs")
    train.add_argument(
        "--sample-rate", type=int, choices=[32000, 40000, 48000], default=40000
    )
    train.add_argument("--epochs", type=int, default=0, help="0 = automatic")
    train.add_argument("--batch-size", type=int, default=0, help="0 = automatic")
    train.add_argument(
        "--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto"
    )
    train.add_argument("--fresh", action="store_true", help="discard old checkpoints")
    train.add_argument("--prepare-only", action="store_true")
    train.set_defaults(func=cmd_train)

    imp = sub.add_parser("import", help="import a model (.pth/.zip/URL)")
    imp.add_argument("source", help="path(s) separated by commas, or a URL")
    imp.add_argument("--name", default=None)
    imp.set_defaults(func=cmd_import)

    models = sub.add_parser("models", help="list voice models")
    models.set_defaults(func=cmd_models)

    dl = sub.add_parser("download", help="download the essential base models")
    dl.set_defaults(func=cmd_download)
    return parser


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "faiss", "numba", "audio_separator"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args(["web", *(argv or [])])
    args.func(args)


if __name__ == "__main__":
    main()
