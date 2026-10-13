import numpy as np
import pytest

from voxshift.dsp.graph import Voice
from voxshift.dsp.nodes import NODE_TYPES, Biquad, Delay, FormantPitchShifter, PitchShift
from voxshift.dsp.stft import StreamingSTFT
from voxshift.presets import is_ai, list_presets

SR = 48000


def sine(f, secs=1.0, sr=SR):
    return 0.5 * np.sin(2 * np.pi * f * np.arange(int(sr * secs)) / sr)


def run(node, x, block):
    return np.concatenate([node.process(x[i:i + block]) for i in range(0, len(x), block)])


def dominant_hz(y, sr=SR):
    y = y[len(y) // 3:]  # skip warm-up
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.argmax(spec) * sr / len(y)


@pytest.mark.parametrize("block", [64, 300, 512, 2048])
def test_stft_identity_has_exact_latency(block):
    st = StreamingSTFT(2048, 512)
    x = np.random.default_rng(0).standard_normal(SR // 2)
    y = run(st, x, block)
    lat = st.latency
    assert len(y) == len(x)
    np.testing.assert_allclose(y[lat + 2048:], x[2048: len(x) - lat], atol=1e-9)


def test_formant_pitch_shifter_octave_up():
    node = FormantPitchShifter(SR, semitones=12, preserve_formants=False)
    assert abs(dominant_hz(run(node, sine(220), 512)) - 440) < 15


def test_granular_pitch_shift_fifth_down():
    node = PitchShift(SR, semitones=-7)
    target = 330 * 2 ** (-7 / 12)
    assert abs(dominant_hz(run(node, sine(330), 480)) - target) < 10


def test_autotune_snaps_to_semitone():
    f_in = 440 * 2 ** (0.4 / 12)  # 40 cents sharp of A4
    node = FormantPitchShifter(SR, autotune=1.0, autotune_speed=1.0, preserve_formants=False)
    assert abs(dominant_hz(run(node, sine(f_in, 1.5), 512)) - 440) < 8


@pytest.mark.parametrize("block_a,block_b", [(512, 173), (64, 1000)])
def test_time_domain_nodes_are_block_size_invariant(block_a, block_b):
    x = np.random.default_rng(1).standard_normal(SR // 4)
    for make in (lambda: Biquad(SR, kind="peaking", freq=1000, q=2, gain_db=6),
                 lambda: Delay(SR, time_ms=7, feedback=0.6, mix=0.5)):
        np.testing.assert_allclose(run(make(), x, block_a), run(make(), x, block_b), atol=1e-9)


DSP_PRESETS = {k: v for k, v in list_presets().items() if not is_ai(v)}


@pytest.mark.parametrize("key", sorted(DSP_PRESETS))
def test_every_dsp_preset_is_stable(key):
    v = Voice(DSP_PRESETS[key], SR)
    rng = np.random.default_rng(2)
    x = np.concatenate([sine(180, 0.5), 0.1 * rng.standard_normal(SR // 2)])
    y = run(v, x, 480)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 8.0


def test_node_registry_covers_presets():
    used = set()

    def walk(chain):
        for item in chain:
            if item["type"] == "Parallel":
                for b in item["branches"]:
                    walk(b)
            else:
                used.add(item["type"])
    for spec in DSP_PRESETS.values():
        walk(spec["chain"])
    assert used <= set(NODE_TYPES)


def test_parallel_branches_are_latency_aligned():
    spec = {"chain": [{"type": "Parallel", "branches": [
        [{"type": "FormantPitchShifter", "params": {"semitones": 0, "preserve_formants": False}}],
        [{"type": "Gain", "params": {"gain_db": 0}}]]}]}
    v = Voice(spec, SR)
    assert v.latency == 2048


def test_partitioned_reverb_matches_direct_convolution():
    from scipy.signal import fftconvolve

    from voxshift.dsp.nodes import ConvolutionReverb
    node = ConvolutionReverb(SR, decay_s=0.3, predelay_ms=30, mix=1.0)
    x = np.random.default_rng(3).standard_normal(SR // 2)
    y = run(node, x, 300)
    # rebuild the effective IR: partition delay P + shortened pre-delay
    pre = int(SR * 0.03) - node.P
    parts = np.fft.irfft(node.H, axis=1)[:, : node.P].ravel()
    ref = fftconvolve(x, np.concatenate([np.zeros(node.P), parts]))[: len(x)]
    assert pre > 0
    np.testing.assert_allclose(y, ref, atol=1e-8)
