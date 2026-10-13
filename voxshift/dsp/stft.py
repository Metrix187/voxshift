"""Streaming STFT framework shared by the spectral nodes.

Used by the spectral nodes (pitch/formant shifting, vocoder, whisper,
autotune): audio arrives in arbitrary block sizes, is framed
with a fixed hop, transformed, edited per frame, and overlap-added back.
Output latency is exactly ``n_fft`` samples regardless of block size.
"""
from __future__ import annotations

import numpy as np


class StreamingSTFT:
    def __init__(self, n_fft: int = 2048, hop: int = 512):
        if n_fft % hop:
            raise ValueError("n_fft must be a multiple of hop")
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = n_fft // 2 + 1
        self.window = np.hanning(n_fft + 1)[:-1]
        # Hann analysis * Hann synthesis overlap-adds to sum(w^2)/hop.
        self.ola_gain = hop / np.sum(self.window**2)
        self._buf = np.zeros(n_fft - hop)
        self._ola = np.zeros(n_fft)
        # One hop of pre-roll guarantees the output FIFO never underruns.
        self._fifo = np.zeros(hop)

    @property
    def latency(self) -> int:
        return self.n_fft

    def process_frame(self, spec: np.ndarray) -> np.ndarray:
        return spec

    def process(self, x: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, x])
        outs = []
        while len(self._buf) >= self.n_fft:
            spec = np.fft.rfft(self._buf[: self.n_fft] * self.window)
            y = np.fft.irfft(self.process_frame(spec), self.n_fft)
            self._ola += y * self.window * self.ola_gain
            outs.append(self._ola[: self.hop].copy())
            self._ola = np.concatenate([self._ola[self.hop:], np.zeros(self.hop)])
            self._buf = self._buf[self.hop:]
        if outs:
            self._fifo = np.concatenate([self._fifo, *outs])
        out, self._fifo = self._fifo[: len(x)], self._fifo[len(x):]
        return out


def cepstral_envelope(mag: np.ndarray, n_coeffs: int) -> np.ndarray:
    """Smooth spectral envelope via cepstral liftering (used for formants)."""
    log_mag = np.log(np.maximum(mag, 1e-9))
    cep = np.fft.irfft(log_mag)
    cep[n_coeffs: len(cep) - n_coeffs + 1] = 0.0
    return np.exp(np.fft.rfft(cep).real)


def wrap_phase(p: np.ndarray) -> np.ndarray:
    return (p + np.pi) % (2.0 * np.pi) - np.pi
