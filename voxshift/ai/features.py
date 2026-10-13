"""Torch-free audio front-ends for the ONNX runtime path.

The standalone build computes these in numpy so it needs only numpy,
scipy and ONNX Runtime (no PyTorch / transformers / librosa). Each function reproduces its reference
implementation (tested against them in tests/test_features.py):

* :func:`whisper_log_mel` — Hugging Face ``WhisperFeatureExtractor``
* :func:`mel_spectrogram` — Seed-VC ``modules.audio.mel_spectrogram``
* :func:`kaldi_fbank`     — ``torchaudio.compliance.kaldi.fbank`` (CAMPPlus input)

Filter matrices are produced once at export time (``features.npz``).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample_poly


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    return sliding_window_view(x, size)[::hop]


def whisper_log_mel(w16: np.ndarray, mel_filters: np.ndarray, n_samples: int = 480000,
                    n_fft: int = 400, hop: int = 160) -> np.ndarray:
    """[N] float audio @16 kHz -> [1, 80, 3000] Whisper input features."""
    x = np.zeros(n_samples, dtype=np.float64)
    x[: min(len(w16), n_samples)] = w16[:n_samples]
    x = np.pad(x, n_fft // 2, mode="reflect")
    win = np.hanning(n_fft + 1)[:-1]
    spec = np.abs(np.fft.rfft(_frames(x, n_fft, hop) * win, axis=-1)) ** 2   # [frames, 201]
    mel = np.maximum(spec @ mel_filters, 1e-10)                                # [frames, 80]
    log_spec = np.log10(mel).T[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0)[None].astype(np.float32)


def mel_spectrogram(wav: np.ndarray, mel_basis: np.ndarray, n_fft: int = 1024, hop: int = 256,
                    win_size: int = 1024) -> np.ndarray:
    """[N] float audio @ model rate -> [1, n_mels, frames] log-mel (Seed-VC, center=False)."""
    pad = (n_fft - hop) // 2
    x = np.pad(wav.astype(np.float64), pad, mode="reflect")
    win = np.hanning(win_size + 1)[:-1]
    if win_size < n_fft:
        lpad = (n_fft - win_size) // 2
        win = np.pad(win, (lpad, n_fft - win_size - lpad))
    spec = np.fft.rfft(_frames(x, n_fft, hop) * win, axis=-1)
    mag = np.sqrt(spec.real**2 + spec.imag**2 + 1e-9)                         # [frames, bins]
    mel = mel_basis @ mag.T                                                    # [n_mels, frames]
    return np.log(np.maximum(mel, 1e-5))[None].astype(np.float32)


def build_constants(whisper_mel_filters: np.ndarray, sr: int, n_fft: int, n_mels: int, fmin: float,
                    fmax: float | None) -> dict[str, np.ndarray]:
    """Dev/export-time only (needs librosa + torchaudio): the filter matrices
    the runtime functions above consume."""
    import librosa
    from torchaudio.compliance.kaldi import get_mel_banks
    banks, _ = get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    return {
        "whisper_mel_filters": np.asarray(whisper_mel_filters, np.float64),                  # [201, 80]
        "mel_basis": librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax),
        "kaldi_mel_banks": np.pad(banks.numpy().astype(np.float64), ((0, 0), (0, 1))),       # [80, 257]
    }


def kaldi_fbank(w16: np.ndarray, mel_banks: np.ndarray, frame_len: int = 400, shift: int = 160,
                preemph: float = 0.97) -> np.ndarray:
    """[N] float audio @16 kHz -> [frames, n_mels] log mel filterbank (Kaldi defaults,
    povey window, dither 0, snip_edges, power spectrum)."""
    fr = _frames(w16.astype(np.float64), frame_len, shift).copy()
    fr -= fr.mean(axis=1, keepdims=True)                                       # remove DC offset
    prev = np.concatenate([fr[:, :1], fr[:, :-1]], axis=1)                     # replicate-pad preemphasis
    fr = fr - preemph * prev
    fr *= np.hanning(frame_len) ** 0.85                                        # povey window
    n_fft = 1 << (frame_len - 1).bit_length()
    power = np.abs(np.fft.rfft(fr, n=n_fft, axis=-1)) ** 2                     # [frames, n_fft/2+1]
    eps = np.finfo(np.float32).eps
    return np.log(np.maximum(power @ mel_banks.T, eps)).astype(np.float32)
