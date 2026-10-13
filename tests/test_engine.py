"""Audio-engine callback logic, driven directly (no audio device is opened)."""
import numpy as np
import pytest

from voxshift.presets import build_processor, load_spec

SR = 48000


@pytest.fixture
def engine():
    try:
        from voxshift.audio.engine import AudioEngine
        return AudioEngine(samplerate=SR, blocksize=480)
    except Exception as e:  # no PortAudio devices on this machine
        pytest.skip(f"no audio devices: {e}")


def call(engine, x, out_ch=2):
    out = np.zeros((len(x), out_ch), np.float32)
    engine._callback(x[:, None].astype(np.float32), out, len(x), None, None)
    return out


def test_bypass_is_transparent_below_limiter(engine):
    engine.voice_enabled = False
    x = 0.1 * np.sin(2 * np.pi * 300 * np.arange(480) / SR)
    out = call(engine, x)
    np.testing.assert_allclose(out[:, 0], out[:, 1])
    assert np.max(np.abs(out[:, 0] - x)) < 1e-3  # tanh limiter is ~linear at -20 dBFS


def test_limiter_caps_hot_signal(engine):
    engine.set_processor(build_processor(load_spec("megaphone"), SR))
    engine.output_gain_db = 24
    out = call(engine, np.random.default_rng(0).standard_normal(480))
    assert np.max(np.abs(out)) <= 10 ** (-0.5 / 20) + 1e-6


def test_soundboard_mixes_and_drains(engine):
    engine.voice_enabled = False
    engine.play_sound(np.full(24000, 0.25), 24000)   # 1 s at 24 kHz -> 48000 samples
    blocks = [call(engine, np.zeros(480))[:, 0] for _ in range(101)]
    assert np.mean(np.abs(blocks[0])) > 0.2
    assert np.max(np.abs(blocks[-1])) == 0.0
    assert engine._sounds == []
