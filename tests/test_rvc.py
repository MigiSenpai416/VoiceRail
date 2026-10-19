import numpy as np
import pytest
import torch

from voicerail.rvc import f0 as f0u
from voicerail.rvc.checkpoint import (
    ModelFormatError,
    export_voice_model,
    read_voice_model,
    to_legacy,
    to_parametrized,
)
from voicerail.rvc.index import FeatureIndex, build_index
from voicerail.rvc.nn import MultiPeriodDiscriminator, Synthesizer
from voicerail.rvc.nn.commons import rand_slice_segments, slice_segments
from voicerail.train.config import AUDIO_CONFIGS


def test_coarse_f0_matches_rvc_range():
    f0 = np.array([0.0, 30.0, 50.0, 440.0, 1100.0, 5000.0])
    coarse = f0u.coarse_f0(f0)
    assert coarse[0] == 1 and coarse[1] == 1
    assert coarse[2] == 1
    assert 1 < coarse[3] < 255
    assert coarse[4] == 255 and coarse[5] == 255


def test_refine_f0_fixes_octave_glitch_and_gaps():
    f0 = np.full(200, 220.0)
    f0[100:103] = 440.0  # short octave error
    f0[150:152] = 0.0  # tiny dropout inside a phrase
    out = f0u.refine_f0(f0)
    assert np.allclose(out[95:110], 220.0, rtol=0.01)
    assert (out[150:152] > 0).all()


def test_refine_f0_keeps_real_leaps():
    f0 = np.concatenate([np.full(100, 220.0), np.full(100, 440.0)])
    out = f0u.refine_f0(f0)
    assert np.allclose(out[120:], 440.0, rtol=0.01)


def test_refine_f0_drops_short_blips():
    f0 = np.zeros(100)
    f0[50] = 300.0
    assert (f0u.refine_f0(f0) == 0).all()


def test_suggest_shift():
    source = np.full(100, 110.0)
    assert f0u.suggest_shift(source, 220.0, octaves_only=True) == 12
    assert f0u.suggest_shift(source, 165.0, octaves_only=False) == 7
    assert f0u.suggest_shift(source, None, octaves_only=False) == 0


def test_weight_norm_key_mapping_roundtrip():
    state = {"a.weight_g": torch.ones(1), "a.weight_v": torch.zeros(1), "b": 1}
    param = to_parametrized(state)
    assert "a.parametrizations.weight.original0" in param
    assert to_legacy(param) == state


def test_slice_segments():
    x = torch.arange(20).view(2, 1, 10).float()
    out = slice_segments(x, torch.tensor([2, 5]), 3)
    assert out[0, 0].tolist() == [2, 3, 4]
    assert out[1, 0].tolist() == [15, 16, 17]
    sliced, starts = rand_slice_segments(x, torch.tensor([10, 4]), 4)
    assert sliced.shape == (2, 1, 4)
    assert starts[1].item() == 0


@pytest.mark.parametrize("sr", [32000, 40000, 48000])
def test_synthesizer_shapes(sr):
    config = AUDIO_CONFIGS[sr]
    net = Synthesizer(config.synthesizer()).eval()
    frames = 50
    phone = torch.randn(1, frames, 768)
    lengths = torch.tensor([frames])
    pitch = torch.randint(1, 255, (1, frames))
    pitchf = torch.rand(1, frames) * 300 + 80
    audio = net.infer(phone, lengths, pitch, pitchf, torch.tensor([0]))
    assert audio.shape == (1, 1, frames * config.hop_length)


def test_training_forward_and_discriminator():
    config = AUDIO_CONFIGS[40000]
    net = Synthesizer(config.synthesizer())
    disc = MultiPeriodDiscriminator()
    frames = 40
    batch = 2
    out = net(
        torch.randn(batch, frames, 768),
        torch.tensor([frames, frames - 5]),
        torch.randint(1, 255, (batch, frames)),
        torch.rand(batch, frames) * 300 + 80,
        torch.rand(batch, config.spec_channels, frames),
        torch.tensor([frames, frames - 5]),
        torch.zeros(batch, dtype=torch.long),
    )
    y_hat = out[0]
    assert y_hat.shape == (batch, 1, config.segment_samples)
    real, fake, _, _ = disc(y_hat.detach(), y_hat.detach())
    assert len(real) == len(fake) == 9


def test_export_and_read_model(tmp_path):
    config = AUDIO_CONFIGS[40000].synthesizer()
    net = Synthesizer(config)
    path = export_voice_model(
        net.state_dict(),
        config,
        tmp_path / "m.pth",
        name="T",
        epoch=3,
        step=10,
        extra={"f0_median": 200.0},
    )
    model = read_voice_model(path)
    assert model.sample_rate == 40000 and model.use_f0 and model.version == "v2"
    assert model.metadata["voicerail"]["f0_median"] == 200.0
    assert not any(k.startswith("enc_q") for k in model.weights)
    assert any(k.endswith("weight_g") for k in model.weights)
    built = model.build(torch.device("cpu"))
    assert not hasattr(built, "enc_q")


def test_read_rejects_training_checkpoint(tmp_path):
    torch.save({"model": {}, "iteration": 1}, tmp_path / "G_100.pth")
    with pytest.raises(ModelFormatError, match="training checkpoint"):
        read_voice_model(tmp_path / "G_100.pth")


def test_read_rejects_garbage(tmp_path):
    (tmp_path / "x.pth").write_bytes(b"not a model")
    with pytest.raises(ModelFormatError):
        read_voice_model(tmp_path / "x.pth")


def test_index_blend_handles_missing_neighbours(tmp_path):
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(500, 16)).astype(np.float32)
    path = build_index(feats, tmp_path / "f.index")
    index = FeatureIndex(path)
    query = rng.normal(size=(20, 16)).astype(np.float32)
    out = index.blend(query, 0.75)
    assert out.shape == query.shape and np.isfinite(out).all()
    assert np.allclose(index.blend(query, 0.0), query)
    exact = index.blend(feats[:5], 1.0)
    assert np.allclose(exact, feats[:5], atol=1e-3)
