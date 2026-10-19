"""End-to-end tests with the real base models.

Set ``VOICERAIL_TEST_ASSETS`` to a data directory whose ``assets`` folder
holds rmvpe.pt, contentvec/ and pretrained/rvc_v2/f0{G,D}40k.pth.
"""

import shutil

import numpy as np
import pytest

from tests.conftest import assets_home

pytestmark = [pytest.mark.assets, pytest.mark.slow]


@pytest.fixture
def with_assets(isolated_home):
    source = assets_home()
    if not source or not (source / "assets" / "rmvpe.pt").exists():
        pytest.skip("VOICERAIL_TEST_ASSETS not configured")
    isolated_home.mkdir(parents=True, exist_ok=True)
    try:
        # Link instead of copying ~600 MB of base models on every run.
        (isolated_home / "assets").symlink_to(
            (source / "assets").resolve(), target_is_directory=True
        )
    except OSError:
        shutil.copytree(source / "assets", isolated_home / "assets", symlinks=True)
    return isolated_home


def _speech(seconds: float, sr: int = 16000) -> np.ndarray:
    """A crude voiced signal: harmonic stack with a moving pitch."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    wave = sum(np.sin(k * phase) / k for k in range(1, 12))
    envelope = (np.sin(2 * np.pi * 2 * t) > -0.3).astype(float)
    return (0.2 * wave * envelope).astype(np.float32)


def test_train_then_convert(with_assets, tmp_path):
    import soundfile as sf

    from voicerail import library, paths
    from voicerail.convert import ConvertRequest, convert
    from voicerail.train.config import TrainConfig
    from voicerail.train.dataset import CleaningOptions, SegmentOptions
    from voicerail.train.job import run_job, write_job

    data = tmp_path / "data"
    data.mkdir()
    for i in range(3):
        sf.write(data / f"take{i}.wav", _speech(8), 16000)

    exp = paths.training_dir() / "It"
    write_job(
        exp,
        "It",
        [str(data)],
        CleaningOptions(isolate_vocals=False, dereverb=False),
        SegmentOptions(),
        TrainConfig(epochs=1, batch_size=2, save_every=1),
        ["prepare", "features", "train", "index", "publish"],
    )
    assert run_job(exp) == 0
    info = library.get_info("It")
    assert info.has_index and info.f0_median and 100 < info.f0_median < 200

    result = convert(
        ConvertRequest(
            data / "take0.wav",
            "It",
            mode="voice",
            dereverb=False,
            auto_pitch=True,
            output_format="flac",
        )
    )
    audio, sr = sf.read(result.output)
    assert sr == 44100 and abs(len(audio) / sr - 8) < 0.05
    assert np.isfinite(audio).all() and np.abs(audio).max() > 0.01
