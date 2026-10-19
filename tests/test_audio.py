import numpy as np
import pytest

from voicerail.audio import dsp
from voicerail.audio.io import load_audio, save_audio
from voicerail.audio.separation import _is_wanted
from voicerail.audio.slicer import Slicer, split_long

SR = 16000


def tone(seconds: float, sr: int = SR, freq: float = 220.0, amp: float = 0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_slicer_cuts_at_silence():
    audio = np.concatenate([tone(2), np.zeros(SR), tone(2)])
    chunks = Slicer(SR).slice(audio)
    assert len(chunks) == 2
    for chunk in chunks:
        # Each clip keeps the tone and at most ~0.5 s of silence.
        assert 1.9 * SR < len(chunk) < 2.7 * SR


def test_slicer_keeps_short_audio_whole():
    audio = tone(1.0)
    assert len(Slicer(SR).slice(audio)) == 1


def test_split_long_windows_overlap():
    audio = tone(10)
    pieces = split_long(audio, SR, max_seconds=3.7, overlap=0.3)
    assert all(len(p) <= int(3.7 * SR) + int(0.3 * SR) for p in pieces)
    assert sum(len(p) for p in pieces) >= len(audio)


def test_crossfade_concat_preserves_constant_signal():
    a = np.ones(1000, np.float32)
    b = np.ones(1000, np.float32)
    out = dsp.crossfade_concat([a, b], 100)
    assert len(out) == 1900
    assert np.allclose(out, 1.0, atol=1e-6)


def test_normalize_loudness_respects_peak():
    audio = tone(3, amp=0.01)
    out = dsp.normalize_loudness(audio, SR, target_lufs=-5.0, peak_db=-1.0)
    assert dsp.peak(out) <= 10 ** (-1 / 20) + 1e-6


def test_highpass_removes_dc():
    audio = tone(1) + 0.3
    out = dsp.highpass(audio, SR, 48)
    assert abs(out.mean()) < 0.01


@pytest.mark.parametrize(
    "stem,wanted",
    [
        ("Vocals", True),
        ("vocals", True),
        ("Instrumental", False),
        ("other", False),
        ("noreverb", True),
        ("reverb", False),
        ("dry", True),
        ("No dry", False),
        ("No Reverb", True),
        ("No Vocals", False),
    ],
)
def test_separation_stem_choice(stem, wanted):
    assert _is_wanted(stem) is wanted


def test_save_and_load_roundtrip(tmp_path):
    audio = np.stack([tone(1, 44100), tone(1, 44100, 330)])
    path = save_audio(tmp_path / "x.wav", audio, 44100)
    back, sr = load_audio(path, mono=False)
    assert sr == 44100 and back.shape == audio.shape
    assert np.allclose(back, audio, atol=1e-6)
    mono, sr16 = load_audio(path, sr=16000)
    assert sr16 == 16000 and mono.ndim == 1
    assert abs(len(mono) - 16000) <= 1


def test_mp3_encode_decode(tmp_path):
    audio = tone(2, 44100)
    path = save_audio(tmp_path / "x", audio, 44100, "mp3")
    back, sr = load_audio(path, sr=44100)
    assert path.suffix == ".mp3" and sr == 44100
    assert abs(len(back) - len(audio)) < 44100 * 0.1


def test_save_audio_keeps_dotted_names(tmp_path):
    audio = tone(0.5, 44100)
    a = save_audio(tmp_path / "01. Intro (Voice v1.5) 2026", audio, 44100, "wav")
    b = save_audio(
        tmp_path / "01. Intro (Voice v1.5) 2026 [vocals]", audio, 44100, "mp3"
    )
    assert a.name == "01. Intro (Voice v1.5) 2026.wav"
    assert b.name == "01. Intro (Voice v1.5) 2026 [vocals].mp3"
    assert save_audio(tmp_path / "x.flac", audio, 44100).suffix == ".flac"


def test_frame_rms_matches_direct_computation():
    rng = np.random.default_rng(0)
    audio = rng.normal(size=5000).astype(np.float32)
    frame, hop = 400, 160
    padded = np.pad(audio, (frame // 2, frame // 2))
    n = 1 + (len(padded) - frame) // hop
    direct = [
        np.sqrt(np.mean(padded[i * hop : i * hop + frame] ** 2)) for i in range(n)
    ]
    assert np.allclose(dsp.frame_rms(audio, frame, hop), direct, atol=1e-5)
