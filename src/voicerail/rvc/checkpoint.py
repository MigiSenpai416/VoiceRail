"""Reading and writing RVC model files.

Voice models are stored in the format popularised by RVC WebUI and Applio::

    {"weight": state_dict (fp16, no posterior encoder), "config": [...],
     "sr": "40k", "f0": 1, "version": "v2", "info": "...", ...}

Weight-normalised layers use the legacy ``weight_g`` / ``weight_v`` names in
these files so that models trained here work in other RVC tools and vice
versa. Internally the modules use ``torch.nn.utils.parametrizations`` and keys
are translated on the way in and out.

Files are always opened with ``weights_only=True`` so that loading a model
downloaded from the internet cannot execute arbitrary code.
"""

from __future__ import annotations

import datetime as dt
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..paths import replace_file
from .nn.commons import remove_weight_norm
from .nn.synthesizer import Synthesizer, SynthesizerConfig

_LEGACY_G = ".weight_g"
_LEGACY_V = ".weight_v"
_PARAM_G = ".parametrizations.weight.original0"
_PARAM_V = ".parametrizations.weight.original1"

SUPPORTED_VOCODERS = {"hifi-gan", "hifigan", ""}
# Embedders producing ContentVec/HuBERT-base features (what VoiceRail extracts).
SUPPORTED_EMBEDDERS = {"", "contentvec", "contentvec_base", "hubert", "hubert_base"}


class ModelFormatError(ValueError):
    """The file is not a usable RVC voice model."""


def to_parametrized(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = OrderedDict()
    for key, value in state.items():
        if key.endswith(_LEGACY_G):
            key = key[: -len(_LEGACY_G)] + _PARAM_G
        elif key.endswith(_LEGACY_V):
            key = key[: -len(_LEGACY_V)] + _PARAM_V
        out[key] = value
    return out


def to_legacy(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = OrderedDict()
    for key, value in state.items():
        if key.endswith(_PARAM_G):
            key = key[: -len(_PARAM_G)] + _LEGACY_G
        elif key.endswith(_PARAM_V):
            key = key[: -len(_PARAM_V)] + _LEGACY_V
        out[key] = value
    return out


def safe_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelFormatError(
            f"Could not read '{Path(path).name}'. It is either corrupted or not a "
            f"PyTorch model file ({error.__class__.__name__}: {error})."
        ) from error


def load_module_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    ignore_prefixes: tuple[str, ...] = (),
) -> None:
    """Load a (legacy or parametrized) state dict, reporting mismatches."""
    state = to_parametrized(state)
    own = module.state_dict()
    missing = [k for k in own if k not in state and not k.startswith(ignore_prefixes)]
    unexpected = [k for k in state if k not in own]
    mismatched = [
        f"{k}: file {tuple(state[k].shape)} vs model {tuple(own[k].shape)}"
        for k in own
        if k in state and state[k].shape != own[k].shape
    ]
    if missing or unexpected or mismatched:
        details = []
        if mismatched:
            details.append("shape mismatch: " + "; ".join(mismatched[:3]))
        if missing:
            details.append("missing: " + ", ".join(missing[:3]))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected[:3]))
        raise ModelFormatError(
            "The weights do not match the expected architecture ("
            + " | ".join(details)
            + ")."
        )
    merged = {k: state.get(k, v) for k, v in own.items()}
    module.load_state_dict(
        {k: v.to(own[k].dtype) for k, v in merged.items()}, strict=True
    )


@dataclass
class VoiceModel:
    """A parsed RVC voice model file."""

    path: Path
    config: SynthesizerConfig
    weights: dict[str, torch.Tensor]
    version: str
    use_f0: bool
    embedding_dim: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_rate(self) -> int:
        return self.config.sr

    @property
    def embedder(self) -> str:
        return str(self.metadata.get("embedder_model") or "contentvec")

    def build(self, device: torch.device, half: bool = False) -> Synthesizer:
        """Instantiate the synthesizer ready for inference."""
        net = Synthesizer(
            self.config, use_f0=self.use_f0, embedding_dim=self.embedding_dim
        )
        del net.enc_q
        load_module_state(net, self.weights, ignore_prefixes=("enc_q.",))
        remove_weight_norm(net)
        net.eval().to(device)
        return net.half() if half else net.float()


def read_voice_model(path: str | Path) -> VoiceModel:
    path = Path(path)
    data = safe_load(path)
    if not isinstance(data, dict):
        raise ModelFormatError(f"'{path.name}' is not an RVC voice model.")
    if "weight" not in data:
        if "model" in data:
            raise ModelFormatError(
                f"'{path.name}' is a raw training checkpoint (G_*.pth/D_*.pth), "
                "not an exported voice model. Use the exported model file instead."
            )
        raise ModelFormatError(f"'{path.name}' is not an RVC voice model.")

    weights = data["weight"]
    if "config" not in data:
        raise ModelFormatError(f"'{path.name}' has no model configuration.")
    try:
        config = SynthesizerConfig.from_list(list(data["config"]))
    except (TypeError, ValueError) as error:
        raise ModelFormatError(
            f"'{path.name}' has an unsupported configuration: {error}"
        ) from error

    vocoder = str(data.get("vocoder") or "").lower().replace(" ", "")
    if vocoder not in SUPPORTED_VOCODERS:
        raise ModelFormatError(
            f"'{path.name}' uses the '{data['vocoder']}' vocoder, which VoiceRail "
            "does not support yet. Only standard HiFi-GAN RVC models are supported."
        )

    embedder = str(data.get("embedder_model") or "").lower().replace("-", "_")
    if embedder not in SUPPORTED_EMBEDDERS:
        raise ModelFormatError(
            f"'{path.name}' was trained with the '{data['embedder_model']}' feature "
            "embedder; VoiceRail only supports ContentVec/HuBERT-base models, and "
            "converting with a different embedder would sound garbled."
        )

    emb_g = weights.get("emb_g.weight")
    emb_phone = weights.get("enc_p.emb_phone.weight")
    if emb_g is None or emb_phone is None:
        raise ModelFormatError(f"'{path.name}' is missing core RVC weights.")
    # Older exports sometimes record a stale speaker count in the config.
    config.spk_embed_dim = int(emb_g.shape[0])
    embedding_dim = int(emb_phone.shape[1])
    use_f0 = "enc_p.emb_pitch.weight" in weights
    version = str(data.get("version") or ("v1" if embedding_dim == 256 else "v2"))

    metadata = {
        k: v
        for k, v in data.items()
        if k not in {"weight", "config"} and isinstance(v, (str, int, float, dict))
    }
    return VoiceModel(
        path=path,
        config=config,
        weights=weights,
        version=version,
        use_f0=use_f0,
        embedding_dim=embedding_dim,
        metadata=metadata,
    )


def export_voice_model(
    net_g_state: dict[str, torch.Tensor],
    config: SynthesizerConfig,
    path: str | Path,
    *,
    name: str,
    epoch: int,
    step: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write an inference-only model file compatible with other RVC tools."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = OrderedDict(
        (k, v.detach().to("cpu", torch.float16))
        for k, v in to_legacy(net_g_state).items()
        if not k.startswith("enc_q.")
    )
    exported_config = SynthesizerConfig(**vars(config))
    data: dict[str, Any] = OrderedDict(
        weight=weights,
        config=exported_config.to_list(),
        info=f"{epoch}epoch",
        sr=f"{config.sr // 1000}k",
        f0=1,
        version="v2",
        epoch=epoch,
        step=step,
        model_name=name,
        creation_date=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        embedder_model="contentvec",
        vocoder="HiFi-GAN",
    )
    if extra:
        data["voicerail"] = extra
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    replace_file(tmp, path)
    return path
