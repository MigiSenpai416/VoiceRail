"""RVC v2 training loop.

Differences from the classic RVC / Applio trainers, all aimed at reliability
and better final models:

* single process, no ``torch.distributed`` (no port or socket issues),
* the whole dataset is held in memory (no worker processes),
* a held-out validation set scored through the real inference path picks the
  best epoch and drives optional early stopping,
* non-finite losses are skipped instead of corrupting the weights,
* a preview clip is rendered at every save so progress can be heard,
* stopping at any time saves a resumable checkpoint.
"""

from __future__ import annotations

import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from ..assets import pretrained_paths
from ..device import bf16_supported, default_device, empty_cache, vram_gb
from ..paths import replace_file
from ..rvc.checkpoint import export_voice_model, load_module_state, safe_load
from ..rvc.nn import MultiPeriodDiscriminator, Synthesizer
from ..rvc.nn.commons import slice_segments
from .config import AUDIO_CONFIGS, TrainConfig, auto_batch_size, auto_epochs
from .data import TrainingSet, collate, load_example
from .features import FeatureSummary
from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .mel import MelFrontEnd
from .progress import Reporter, StopRequested

log = logging.getLogger(__name__)

MAX_BAD_STEPS = 25
# Safety net only: typical RVC fine-tuning norms are in the hundreds to ~1k.
MAX_GRAD_NORM = 1e4
EMA_ALPHA = 0.3
MIN_IMPROVEMENT = 0.002


@dataclass
class TrainResult:
    best_model: Path
    latest_model: Path
    epochs_done: int
    stopped_early: bool
    interrupted: bool
    best_epoch: int | None
    best_val: float | None


def checkpoints_dir(exp_dir: Path) -> Path:
    return exp_dir / "checkpoints"


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    replace_file(tmp, path)


class Trainer:
    def __init__(
        self,
        exp_dir: Path,
        name: str,
        config: TrainConfig,
        features: FeatureSummary,
        reporter: Reporter,
        device: torch.device | None = None,
    ):
        self.exp_dir = exp_dir
        self.name = name
        self.config = config
        self.features = features
        self.reporter = reporter
        self.device = device or default_device()
        self.audio = AUDIO_CONFIGS[config.sample_rate]
        self.ckpt_dir = checkpoints_dir(exp_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = exp_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        self.metrics_path = exp_dir / "metrics.jsonl"

    # ------------------------------------------------------------------ setup
    def _precision(self) -> tuple[torch.dtype | None, bool]:
        """(autocast dtype or None, use GradScaler)."""
        wanted = self.config.precision
        if self.device.type != "cuda" or wanted == "fp32":
            return None, False
        if wanted in ("auto", "bf16") and bf16_supported(self.device):
            return torch.bfloat16, False
        if wanted == "fp16":
            return torch.float16, True
        return None, False

    def _load_data(self):
        train, val = [], []
        for item in self.features.items:
            example = load_example(item, self.audio)
            if example is None:
                continue
            (val if item.validation else train).append(example)
        if len(train) < 2:
            raise ValueError("Not enough training segments (need at least 2).")
        return train, val

    def _build(self):
        synth_config = self.audio.synthesizer()
        net_g = Synthesizer(
            synth_config,
            use_f0=True,
            embedding_dim=768,
            checkpointing=self.config.gradient_checkpointing,
        )
        net_d = MultiPeriodDiscriminator(
            checkpointing=self.config.gradient_checkpointing
        )
        return net_g.to(self.device), net_d.to(self.device), synth_config

    def _load_pretrained(self, net_g, net_d) -> None:
        g_path, d_path = pretrained_paths(
            self.config.pretrained,
            self.config.sample_rate,
            progress=lambda f, m: self.reporter.progress(f, m),
        )
        self.reporter.progress(0.0, "Loading pretrained weights")
        load_module_state(net_g, safe_load(g_path)["model"])
        load_module_state(net_d, safe_load(d_path)["model"])

    # --------------------------------------------------------------- training
    def run(self) -> TrainResult:
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if self.device.type == "cuda":
            # Segment lengths vary per batch: autotuning would re-run constantly.
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        train_set, val_set = self._load_data()
        batch_size = cfg.batch_size or auto_batch_size(
            vram_gb(self.device), cfg.sample_rate
        )
        batch_size = max(1, min(batch_size, len(train_set)))
        epochs = cfg.epochs or auto_epochs(len(train_set), batch_size)
        log.info(
            "Training on %d segments (%d validation), batch %d, %d epochs, %s",
            len(train_set),
            len(val_set),
            batch_size,
            epochs,
            self.device,
        )
        if self.device.type == "cpu":
            log.warning("No GPU found: training on the CPU will be very slow.")

        net_g, net_d, synth_config = self._build()
        front = MelFrontEnd(self.audio).to(self.device)
        optim_g = torch.optim.AdamW(
            net_g.parameters(), cfg.learning_rate, betas=cfg.betas, eps=cfg.eps
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(), cfg.learning_rate, betas=cfg.betas, eps=cfg.eps
        )
        amp_dtype, use_scaler = self._precision()
        scaler = torch.amp.GradScaler(self.device.type, enabled=use_scaler)

        state = self._resume(net_g, net_d, optim_g, optim_d, scaler)
        if state is None:
            self._load_pretrained(net_g, net_d)
            state = {
                "epoch": 0,
                "step": 0,
                "best_val": None,
                "best_epoch": None,
                "ema_val": None,
                "best_ema": None,
                "since_improvement": 0,
            }
        start_epoch = state["epoch"] + 1
        for optim in (optim_g, optim_d):
            for group in optim.param_groups:
                group["lr"] = cfg.learning_rate * cfg.lr_decay ** state["epoch"]

        loader = DataLoader(
            TrainingSet(train_set),
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_set) >= 2 * batch_size,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(cfg.seed + state["epoch"]),
        )
        steps_per_epoch = len(loader)
        bad_steps = 0
        interrupted = stopped_early = False
        epoch = state["epoch"]
        autocast = (
            (lambda: torch.autocast("cuda", dtype=amp_dtype))
            if amp_dtype is not None
            else nullcontext
        )

        try:
            for epoch in range(start_epoch, epochs + 1):
                net_g.train()
                net_d.train()
                started = time.time()
                totals: dict[str, float] = {}
                counts: dict[str, int] = {}
                for batch_index, batch in enumerate(loader):
                    self.reporter.check_stop()
                    losses = self._step(
                        batch,
                        net_g,
                        net_d,
                        optim_g,
                        optim_d,
                        scaler,
                        front,
                        autocast,
                    )
                    if losses is None:
                        bad_steps += 1
                        if bad_steps > MAX_BAD_STEPS:
                            raise RuntimeError(
                                "Training became numerically unstable (repeated "
                                "NaN/inf losses). Try precision 'fp32' or a lower "
                                "batch size."
                            )
                        continue
                    bad_steps = 0
                    state["step"] += 1
                    for key, value in losses.items():
                        # fp16 overflow steps report inf gradient norms.
                        if math.isfinite(value):
                            totals[key] = totals.get(key, 0.0) + value
                            counts[key] = counts.get(key, 0) + 1
                    done = (epoch - 1 + (batch_index + 1) / steps_per_epoch) / epochs
                    self.reporter.progress(
                        done,
                        f"Epoch {epoch}/{epochs} - step {batch_index + 1}/"
                        f"{steps_per_epoch}",
                    )

                for optim in (optim_g, optim_d):
                    for group in optim.param_groups:
                        group["lr"] *= cfg.lr_decay

                averages = {k: v / counts[k] for k, v in totals.items()}
                val = self._validate(net_g, front, val_set) if val_set else None
                state["epoch"] = epoch
                improved = self._track(state, val)
                record = {
                    "epoch": epoch,
                    "step": state["step"],
                    "seconds": round(time.time() - started, 2),
                    "lr": optim_g.param_groups[0]["lr"],
                    **{k: round(v, 5) for k, v in averages.items()},
                    "val_mel": None if val is None else round(val, 5),
                    "best_epoch": state["best_epoch"],
                }
                with self.metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                self.reporter.metrics(record)
                log.info(
                    "epoch %d | mel %.3f | kl %.3f | fm %.3f | gen %.3f | disc %.3f"
                    " | val %s | %.1fs",
                    epoch,
                    averages.get("mel", math.nan),
                    averages.get("kl", math.nan),
                    averages.get("fm", math.nan),
                    averages.get("gen", math.nan),
                    averages.get("disc", math.nan),
                    "-" if val is None else f"{val:.4f}",
                    record["seconds"],
                )

                if improved:
                    self._export(net_g, synth_config, "best", epoch, state["step"])
                    self._write_best(state)
                final = epoch == epochs
                patience_out = (
                    cfg.early_stopping
                    and val_set
                    and state["since_improvement"] >= cfg.patience
                )
                if epoch % cfg.save_every == 0 or final or patience_out:
                    self._save(net_g, net_d, optim_g, optim_d, scaler, state)
                    self._preview(net_g, val_set or train_set, epoch)
                    if cfg.keep_snapshots:
                        self._export(
                            net_g, synth_config, f"e{epoch}", epoch, state["step"]
                        )
                if patience_out:
                    stopped_early = True
                    log.info(
                        "Validation has not improved for %d epochs; stopping early.",
                        cfg.patience,
                    )
                    break
        except StopRequested:
            interrupted = True
            log.info("Stop requested: saving a checkpoint at epoch %d", epoch)
            state["epoch"] = max(epoch - 1, state["epoch"]) if epoch else 0
            self._save(net_g, net_d, optim_g, optim_d, scaler, state)

        latest = self._export(
            net_g, synth_config, "latest", state["epoch"], state["step"]
        )
        best = self.ckpt_dir / "model_best.pth"
        if not best.is_file():
            best = latest
        empty_cache(self.device)
        return TrainResult(
            best_model=best,
            latest_model=latest,
            epochs_done=state["epoch"],
            stopped_early=stopped_early,
            interrupted=interrupted,
            best_epoch=state["best_epoch"],
            best_val=state["best_val"],
        )

    def _step(self, batch, net_g, net_d, optim_g, optim_d, scaler, front, autocast):
        phone, lengths, pitch, pitchf, wav, sid = (t.to(self.device) for t in batch)
        hop = self.audio.hop_length
        seg = self.audio.segment_samples
        with torch.no_grad():
            spec = front.spectrogram(wav.squeeze(1))[..., : phone.shape[1]]

        with autocast():
            y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = net_g(
                phone, lengths, pitch, pitchf, spec, lengths, sid
            )
            wav_slice = slice_segments(wav, ids_slice * hop, seg)
            real, fake, _, _ = net_d(wav_slice, y_hat.detach())
        loss_disc = discriminator_loss(real, fake)
        if not torch.isfinite(loss_disc):
            optim_d.zero_grad(set_to_none=True)
            return None
        optim_d.zero_grad(set_to_none=True)
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_d = torch.nn.utils.clip_grad_norm_(net_d.parameters(), MAX_GRAD_NORM)
        if not scaler.is_enabled() and not torch.isfinite(grad_d):
            # Without a GradScaler nothing else would stop inf/NaN gradients
            # from permanently corrupting the optimizer state.
            optim_d.zero_grad(set_to_none=True)
            return None
        scaler.step(optim_d)

        with autocast():
            _, fake, fmap_r, fmap_g = net_d(wav_slice, y_hat)
        y_mel = front.mel(wav_slice.squeeze(1))
        y_hat_mel = front.mel(y_hat.float().squeeze(1))
        loss_mel = torch.nn.functional.l1_loss(y_hat_mel, y_mel) * self.config.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.config.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen = generator_loss(fake)
        total = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            scaler.update()
            return None
        scaler.scale(total).backward()
        scaler.unscale_(optim_g)
        grad_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), MAX_GRAD_NORM)
        if not scaler.is_enabled() and not torch.isfinite(grad_g):
            optim_g.zero_grad(set_to_none=True)
            return None
        scaler.step(optim_g)
        scaler.update()
        return {
            "total": total.item(),
            "mel": loss_mel.item(),
            "kl": loss_kl.item(),
            "fm": loss_fm.item(),
            "gen": loss_gen.item(),
            "disc": loss_disc.item(),
            "grad_g": grad_g.item(),
            "grad_d": grad_d.item(),
        }

    # ------------------------------------------------------------- validation
    @torch.no_grad()
    def _render(self, net_g, example) -> torch.Tensor:
        phone = example.phone.float().unsqueeze(0).to(self.device)
        lengths = torch.tensor([phone.shape[1]], device=self.device)
        pitch = example.pitch.unsqueeze(0).to(self.device)
        pitchf = example.pitchf.unsqueeze(0).to(self.device)
        sid = torch.zeros(1, dtype=torch.long, device=self.device)
        return net_g.infer(phone, lengths, pitch, pitchf, sid)[0, 0].float()

    @torch.no_grad()
    def _validate(self, net_g, front, val_set) -> float:
        net_g.eval()
        errors = []
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            for example in val_set:
                generated = self._render(net_g, example)
                target = example.wav.to(self.device)
                n = min(len(generated), len(target))
                mel_g = front.mel(generated[:n].unsqueeze(0))
                mel_t = front.mel(target[:n].unsqueeze(0))
                errors.append(float(torch.mean(torch.abs(mel_g - mel_t))))
        net_g.train()
        return float(np.mean(errors))

    def _track(self, state: dict, val: float | None) -> bool:
        """Update best/EMA bookkeeping; True when this epoch is the new best."""
        if val is None:
            return False
        ema = (
            val
            if state["ema_val"] is None
            else (EMA_ALPHA * val + (1 - EMA_ALPHA) * state["ema_val"])
        )
        state["ema_val"] = ema
        if state["best_ema"] is None or ema < state["best_ema"] * (1 - MIN_IMPROVEMENT):
            state["best_ema"] = ema
            state["since_improvement"] = 0
        else:
            state["since_improvement"] += 1
        if state["best_val"] is None or val < state["best_val"]:
            state["best_val"] = val
            state["best_epoch"] = state["epoch"]
            return True
        return False

    @torch.no_grad()
    def _preview(self, net_g, examples, epoch: int) -> None:
        net_g.eval()
        audio = self._render(net_g, examples[0]).cpu().numpy()
        net_g.train()
        sf.write(
            self.samples_dir / f"epoch_{epoch:04d}.wav", audio, self.audio.sample_rate
        )
        reference = self.samples_dir / "reference.wav"
        if not reference.exists():
            sf.write(reference, examples[0].wav.numpy(), self.audio.sample_rate)

    # ------------------------------------------------------------ persistence
    def _save(self, net_g, net_d, optim_g, optim_d, scaler, state) -> None:
        _atomic_save(
            {"model": net_g.state_dict(), "optimizer": optim_g.state_dict()},
            self.ckpt_dir / "G_latest.pth",
        )
        _atomic_save(
            {"model": net_d.state_dict(), "optimizer": optim_d.state_dict()},
            self.ckpt_dir / "D_latest.pth",
        )
        meta = {
            **state,
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "sample_rate": self.config.sample_rate,
        }
        tmp = self.ckpt_dir / "state.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        replace_file(tmp, self.ckpt_dir / "state.json")

    def _resume(self, net_g, net_d, optim_g, optim_d, scaler) -> dict | None:
        meta_path = self.ckpt_dir / "state.json"
        g_path, d_path = self.ckpt_dir / "G_latest.pth", self.ckpt_dir / "D_latest.pth"
        if not (meta_path.is_file() and g_path.is_file() and d_path.is_file()):
            return None
        meta = json.loads(meta_path.read_text())
        if meta.get("sample_rate") != self.config.sample_rate:
            log.warning("Existing checkpoint has another sample rate; starting over.")
            return None
        g, d = safe_load(g_path), safe_load(d_path)
        load_module_state(net_g, g["model"])
        load_module_state(net_d, d["model"])
        optim_g.load_state_dict(g["optimizer"])
        optim_d.load_state_dict(d["optimizer"])
        if meta.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(meta["scaler"])
        log.info("Resuming from epoch %d", meta["epoch"])
        state = {k: v for k, v in meta.items() if k not in ("scaler", "sample_rate")}
        self._trim_metrics(state["epoch"])
        # model_best.pth may be newer than the last full checkpoint (crash);
        # keep its score so a worse epoch cannot overwrite it.
        best_path = self.ckpt_dir / "best.json"
        if best_path.is_file():
            best = json.loads(best_path.read_text())
            if state.get("best_val") is None or best["best_val"] < state["best_val"]:
                state["best_val"] = best["best_val"]
                state["best_epoch"] = best["best_epoch"]
        return state

    def _trim_metrics(self, last_epoch: int) -> None:
        """Drop chart rows of epochs that will be trained again after resume."""
        if not self.metrics_path.is_file():
            return
        kept = []
        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line)["epoch"] <= last_epoch:
                    kept.append(line)
            except (json.JSONDecodeError, KeyError):
                continue
        self.metrics_path.write_text(
            "".join(f"{line}\n" for line in kept), encoding="utf-8"
        )

    def _write_best(self, state: dict) -> None:
        tmp = self.ckpt_dir / "best.json.tmp"
        tmp.write_text(
            json.dumps(
                {"best_val": state["best_val"], "best_epoch": state["best_epoch"]}
            )
        )
        replace_file(tmp, self.ckpt_dir / "best.json")

    def _export(self, net_g, synth_config, tag: str, epoch: int, step: int) -> Path:
        extra = {
            "f0_median": self.features.f0_median,
            "f0_low": self.features.f0_low,
            "f0_high": self.features.f0_high,
            "trained_with": "VoiceRail",
            "pretrained": self.config.pretrained,
        }
        return export_voice_model(
            net_g.state_dict(),
            synth_config,
            self.ckpt_dir / f"model_{tag}.pth",
            name=self.name,
            epoch=epoch,
            step=step,
            extra=extra,
        )
