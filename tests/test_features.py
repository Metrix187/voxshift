"""numpy front-ends vs the torch / transformers implementations they replace."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from voxshift.ai import features as F  # noqa: E402
from voxshift.ai.seedvc import _seedvc_imports  # noqa: E402


@pytest.fixture(scope="module")
def consts():
    fe = transformers.WhisperFeatureExtractor()  # whisper-small front-end defaults
    return fe, F.build_constants(fe.mel_filters, 22050, 1024, 80, 0, None)


def audio(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(n)).astype(np.float32)


def test_whisper_log_mel(consts):
    fe, c = consts
    x = audio(16000 * 4)
    ref = fe([x], return_tensors="np", sampling_rate=16000).input_features
    got = F.whisper_log_mel(x, c["whisper_mel_filters"])
    assert got.shape == ref.shape == (1, 80, 3000)
    np.testing.assert_allclose(got, ref, atol=2e-5)


def test_seedvc_mel(consts):
    _, c = consts
    with _seedvc_imports():
        from modules.audio import mel_spectrogram
    x = audio(22050 * 2, 1)
    ref = mel_spectrogram(torch.from_numpy(x)[None], n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256,
                          win_size=1024, fmin=0, fmax=None, center=False).numpy()
    got = F.mel_spectrogram(x, c["mel_basis"])
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_kaldi_fbank(consts):
    import torchaudio
    _, c = consts
    x = audio(16000 * 3, 2)
    ref = torchaudio.compliance.kaldi.fbank(torch.from_numpy(x)[None], num_mel_bins=80, dither=0,
                                            sample_frequency=16000).numpy()
    got = F.kaldi_fbank(x, c["kaldi_mel_banks"])
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-3)
