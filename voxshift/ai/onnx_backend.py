"""ONNX Runtime backend for the Seed-VC pipeline — numpy only.

Runs the networks exported by ``onnx_export.py`` through ONNX Runtime with
DirectML first (any DirectX 12 GPU), then CUDA, then CPU. The Euler/CFG
loop, resampling and audio front-ends run in numpy (``features.py``), so a
standalone build needs no PyTorch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import features as F
from ..paths import install_dir

PROVIDER_PREFERENCE = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
NETWORKS = ("whisper_encoder", "length_regulator", "dit_step", "campplus", "vocoder")


def make_session(path: Path, providers: list[str] | None = None):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    chosen = [p for p in (providers or PROVIDER_PREFERENCE) if p in avail] or ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if "DmlExecutionProvider" in chosen:  # DML requires these two settings
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=so, providers=chosen)


class OnnxSeedVCModel:
    _cache: dict[tuple, "OnnxSeedVCModel"] = {}

    @classmethod
    def get(cls, onnx_dir: str, providers: list[str] | None = None) -> "OnnxSeedVCModel":
        key = (str(onnx_dir), tuple(providers or ()))
        if key not in cls._cache:
            cls._cache[key] = cls(onnx_dir, providers)
        return cls._cache[key]

    def __init__(self, onnx_dir: str, providers: list[str] | None = None):
        d = Path(onnx_dir)
        if not d.is_absolute():  # "models/onnx/<name>": bundled, or the downloaded voice pack
            from .model_pack import find_model
            d = find_model(d.name) or install_dir() / d
        if not (d / "config.json").is_file():
            raise FileNotFoundError(f"{d} has no exported model; run `python -m voxshift export-onnx` first")
        self.dir = d
        self.cfg = json.loads((d / "config.json").read_text())
        self.sr, self.hop = self.cfg["sr"], self.cfg["hop_length"]
        self.sess = {n: make_session(d / f"{n}.onnx", providers) for n in NETWORKS}
        self.providers = self.sess["dit_step"].get_providers()
        c = np.load(d / "features.npz")
        self.whisper_filters, self.mel_basis, self.kaldi_banks = (
            c["whisper_mel_filters"], c["mel_basis"], c["kaldi_mel_banks"])
        self.rng = np.random.default_rng()

    def _run(self, name, **feeds):
        feeds = {k: np.ascontiguousarray(v, dtype=np.int64 if v.dtype.kind in "iu" else np.float32)
                 for k, v in feeds.items()}
        return self.sess[name].run(None, feeds)[0]

    # -- building blocks (all numpy) ------------------------------------
    def content(self, w16: np.ndarray) -> np.ndarray:
        """16 kHz audio -> Whisper encoder states [1, frames, 768] (50 frames/s)."""
        hidden = self._run("whisper_encoder", input_features=F.whisper_log_mel(w16, self.whisper_filters))
        return hidden[:, : len(w16) // 320 + 1]

    def length_regulator(self, content: np.ndarray, frames: int) -> np.ndarray:
        return self._run("length_regulator", content=content, ylens=np.array([frames]))

    def style(self, w16: np.ndarray) -> np.ndarray:
        fb = F.kaldi_fbank(w16, self.kaldi_banks)
        return self._run("campplus", fbank=(fb - fb.mean(axis=0, keepdims=True))[None])

    def mel(self, wav: np.ndarray) -> np.ndarray:
        return F.mel_spectrogram(wav, self.mel_basis, self.cfg["n_fft"], self.hop, self.cfg["win_length"])

    def vocoder(self, mel: np.ndarray) -> np.ndarray:
        return self._run("vocoder", mel=mel).reshape(-1)

    def cfm(self, mu, prompt, style, n_timesteps: int, cfg_rate: float) -> np.ndarray:
        """Seed-VC ``BASECFM.solve_euler`` with the DiT estimator in ORT."""
        B, T = mu.shape[0], mu.shape[1]
        x = self.rng.standard_normal((B, 80, T)).astype(np.float32)
        P = prompt.shape[-1]
        prompt_x = np.zeros_like(x)
        prompt_x[..., :P] = prompt[..., :P]
        x[..., :P] = 0
        t_span = np.linspace(0, 1, n_timesteps + 1, dtype=np.float32)
        if cfg_rate > 0:
            pz, sz, mz = (np.concatenate([a, np.zeros_like(a)]) for a in (prompt_x, style, mu))
        for step in range(1, len(t_span)):
            t, dt = t_span[step - 1], t_span[step] - t_span[step - 1]
            if cfg_rate > 0:
                d = self._run("dit_step", x=np.concatenate([x, x]), prompt_x=pz, t=np.array([t, t]), style=sz, mu=mz)
                d = (1 + cfg_rate) * d[:B] - cfg_rate * d[B:]
            else:
                d = self._run("dit_step", x=x, prompt_x=prompt_x, t=np.array([t]), style=style, mu=mu)
            x = x + dt * d
            x[..., :P] = 0
        return x


def load_audio(path: str, sr: int) -> np.ndarray:
    import soundfile as sf
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    return F.resample(data.mean(axis=1), file_sr, sr)


class NumpyStreamer:
    """numpy port of :class:`voxshift.ai.seedvc.SeedVCStreamer` (same context
    windows, skip head/tail and SOLA crossfade) driving :class:`OnnxSeedVCModel`."""

    def __init__(self, model: OnnxSeedVCModel, device_sr: int, cfg):
        if cfg.extra_time_ce < cfg.extra_time:
            raise ValueError("extra_time_ce must be >= extra_time")
        self.m, self.cfg, self.sr = model, cfg, device_sr
        zc = self.zc = device_sr // 50
        q = lambda s: int(round(s * device_sr / zc)) * zc  # noqa: E731
        self.block_frame = q(cfg.block_time)
        self.block_frame_16k = 320 * self.block_frame // zc
        crossfade = q(cfg.crossfade_time)
        self.sola_buffer_frame = min(crossfade, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = q(cfg.extra_time_ce)
        self.extra_frame_right = max(zc, q(cfg.extra_time_right))
        total = self.extra_frame + crossfade + self.sola_search_frame + self.block_frame + self.extra_frame_right
        self.input_wav = np.zeros(total, np.float32)
        self.input_wav_res = np.zeros(320 * total // zc, np.float32)
        self.sola_buffer = np.zeros(self.sola_buffer_frame, np.float32)
        self.skip_head = self.extra_frame // zc
        self.skip_tail = self.extra_frame_right // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc
        self.fade_in = (np.sin(0.5 * np.pi * np.linspace(0, 1, self.sola_buffer_frame)) ** 2).astype(np.float32)
        self.fade_out = 1 - self.fade_in
        self.prompt = None
        self._prev_loud = False
        self.last_infer_ms = 0.0
        if cfg.reference:
            self.set_reference(cfg.reference)

    @property
    def latency(self) -> int:
        return self.extra_frame_right + self.sola_buffer_frame + self.sola_search_frame // 2

    def set_reference(self, path: str) -> None:
        m = self.m
        wav = load_audio(path, m.sr)[: int(m.sr * self.cfg.max_prompt_length)]
        w16 = F.resample(wav, m.sr, 16000)
        mel = m.mel(wav)
        cond = m.length_regulator(m.content(w16), mel.shape[2])
        self.prompt = (cond, mel, m.style(w16))
        self.cfg.reference = path

    def _infer(self) -> np.ndarray:
        m, cfg = self.m, self.cfg
        cond_p, mel_p, style = self.prompt
        ce_dit = int((cfg.extra_time_ce - cfg.extra_time) * 50)        # content-encoder vs DiT context gap
        s_alt = m.content(self.input_wav_res)[:, ce_dit:]
        frames = int((self.skip_head + self.return_length + self.skip_tail - ce_dit) / 50 * m.sr // m.hop)
        cat = np.concatenate([cond_p, m.length_regulator(s_alt, frames)], axis=1)
        mel = m.cfm(cat, mel_p, style, cfg.diffusion_steps, cfg.inference_cfg_rate)[:, :, mel_p.shape[-1]:]
        wave = m.vocoder(mel)
        out_len, tail = self.return_length * m.sr // 50, self.skip_tail * m.sr // 50
        wave = F.resample(wave[-out_len - tail: -tail], m.sr, self.sr)
        need = self.return_length * self.zc
        return np.pad(wave, (0, max(0, need - len(wave))))[:need]

    def process_block(self, block: np.ndarray) -> np.ndarray:
        import time
        if self.prompt is None:
            raise RuntimeError("no reference voice set; call set_reference(path)")
        n = self.block_frame
        assert len(block) == n, (len(block), n)
        self.input_wav = np.roll(self.input_wav, -n)
        self.input_wav[-n:] = block
        self.input_wav_res = np.roll(self.input_wav_res, -self.block_frame_16k)
        self.input_wav_res[-320 * (n // self.zc + 1):] = F.resample(
            self.input_wav[-n - 2 * self.zc:], self.sr, 16000)[320:]
        loud = 20 * np.log10(float(np.sqrt(np.mean(block**2))) + 1e-9) > self.cfg.silence_threshold_db
        t0 = time.perf_counter()
        infer = self._infer() if (loud or self._prev_loud) else np.zeros(self.return_length * self.zc, np.float32)
        self._prev_loud = loud
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        # SOLA: best-aligned offset by normalised cross-correlation, then crossfade
        sb, ss = self.sola_buffer_frame, self.sola_search_frame
        conv_in = infer[: sb + ss]
        num = np.correlate(conv_in, self.sola_buffer, "valid")
        den = np.sqrt(np.convolve(conv_in**2, np.ones(sb), "valid") + 1e-8)
        off = int(np.argmax(num / den))
        infer = infer[off:].copy()
        infer[:sb] = infer[:sb] * self.fade_in + self.sola_buffer * self.fade_out
        self.sola_buffer[:] = infer[n: n + sb]
        return infer[:n].astype(np.float64)
