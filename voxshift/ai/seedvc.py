"""AI voice conversion: Seed-VC v1 real-time pipeline, driven headlessly.

    mic @ device rate ──► 16 kHz ──► content encoder (Whisper-small | XLSR)
         ──► length regulator ──► DiT flow matching (N Euler steps,
             conditioned on the reference prompt mel + CAMPPlus style)
         ──► vocoder (BigVGAN | HiFT) ──► device rate ──► SOLA crossfade

The streaming arithmetic (context windows, skip head/tail, SOLA) is a port
of Seed-VC's ``real-time-gui.py`` with the GUI, FunASR VAD and globals
removed. Seed-VC is GPL-3.0; it is used as a git submodule.
"""
from __future__ import annotations

import contextlib
import io
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDVC_DIR = Path(os.environ.get("VOXSHIFT_SEEDVC", REPO_ROOT / "third_party" / "seed-vc"))

MODEL_PRESETS = {
    # name: (HF checkpoint, Seed-VC config)
    "whisper-small-bigvgan": ("DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                              "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
    "xlsr-tiny-hifigan": ("DiT_uvit_tat_xlsr_ema.pth", "config_dit_mel_seed_uvit_xlsr_tiny.yml"),
}


def _stub_dac():
    """Seed-VC imports ``dac.nn.quantize.VectorQuantize`` unconditionally but
    only instantiates it when ``vector_quantize: true`` (no shipped preset
    does). Installing descript-audio-codec drags in audiotools and a pinned
    protobuf, so register a stub when the real package is absent."""
    import importlib.util
    import types
    if "dac" in sys.modules or importlib.util.find_spec("dac") is not None:
        return

    class VectorQuantize:  # noqa: D401 - placeholder
        def __init__(self, *a, **k):
            raise RuntimeError("this checkpoint needs `pip install descript-audio-codec`")

    pkg, nn, quant = (types.ModuleType(n) for n in ("dac", "dac.nn", "dac.nn.quantize"))
    quant.VectorQuantize = VectorQuantize
    pkg.nn, nn.quantize = nn, quant
    sys.modules.update({"dac": pkg, "dac.nn": nn, "dac.nn.quantize": quant})


@contextlib.contextmanager
def _seedvc_imports():
    """Seed-VC resolves some files relative to its repo root."""
    if not (SEEDVC_DIR / "modules").is_dir():
        raise RuntimeError(
            f"Seed-VC not found at {SEEDVC_DIR}. Run `git submodule update --init` "
            "or set VOXSHIFT_SEEDVC.")
    if str(SEEDVC_DIR) not in sys.path:
        sys.path.insert(0, str(SEEDVC_DIR))
    _stub_dac()
    cwd = os.getcwd()
    os.chdir(SEEDVC_DIR)
    try:
        yield
    finally:
        os.chdir(cwd)


class SeedVCModel:
    """Loads every network once; shareable between streamers."""

    _cache: dict[tuple, "SeedVCModel"] = {}

    @classmethod
    def get(cls, preset: str = "whisper-small-bigvgan", device: str | None = None,
            fp16: bool = True, checkpoint: str | None = None, config: str | None = None) -> "SeedVCModel":
        key = (preset, device, fp16, checkpoint, config)
        if key not in cls._cache:
            cls._cache[key] = cls(preset, device, fp16, checkpoint, config)
        return cls._cache[key]

    def __init__(self, preset, device, fp16, checkpoint, config):
        import torch
        import yaml
        from huggingface_hub import hf_hub_download

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16 and self.device.type == "cuda"
        if checkpoint is None:
            ckpt_name, cfg_name = MODEL_PRESETS[preset]
            checkpoint = hf_hub_download("Plachta/Seed-VC", ckpt_name)
            config = hf_hub_download("Plachta/Seed-VC", cfg_name)
        cfg = yaml.safe_load(open(config, "r"))
        self.config = cfg

        with _seedvc_imports():
            import modules.flow_matching as fm
            from modules.commons import build_model, load_checkpoint, recursive_munch
            from modules.campplus.DTDNN import CAMPPlus
            from modules.audio import mel_spectrogram

            fm.tqdm = lambda it, *a, **k: it  # no progress bar per streamed block

            mp = recursive_munch(cfg["model_params"])
            mp.dit_type = "DiT"
            model = build_model(mp, stage="DiT")
            model, *_ = load_checkpoint(model, None, checkpoint, load_only_params=True,
                                        ignore_modules=[], is_distributed=False)
            for k in model:
                model[k].eval().to(self.device)
            model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            self.model = model

            campplus_path = hf_hub_download("funasr/campplus", "campplus_cn_common.bin")
            self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
            self.campplus.load_state_dict(torch.load(campplus_path, map_location="cpu"))
            self.campplus.eval().to(self.device)

            self.vocoder_type = mp.vocoder.type
            self.vocoder = self._load_vocoder(mp, torch, yaml)
            self.content_type = mp.speech_tokenizer.type
            self.semantic_fn = self._load_content_encoder(mp, cfg, torch)

            sp = cfg["preprocess_params"]["spect_params"]
            self.sr = cfg["preprocess_params"]["sr"]
            self.hop = sp["hop_length"]
            mel_args = dict(n_fft=sp["n_fft"], win_size=sp["win_length"], hop_size=sp["hop_length"],
                            num_mels=sp["n_mels"], sampling_rate=self.sr, fmin=sp.get("fmin", 0),
                            fmax=None if sp.get("fmax", "None") == "None" else 8000, center=False)
            self.to_mel = lambda x: mel_spectrogram(x, **mel_args)

    def _load_vocoder(self, mp, torch, yaml):
        if mp.vocoder.type == "bigvgan":
            from modules.bigvgan import bigvgan
            voc = bigvgan.BigVGAN.from_pretrained(mp.vocoder.name, use_cuda_kernel=False)
            voc.remove_weight_norm()
            return voc.eval().to(self.device)
        if mp.vocoder.type == "hifigan":
            from huggingface_hub import hf_hub_download
            from modules.hifigan.f0_predictor import ConvRNNF0Predictor
            from modules.hifigan.generator import HiFTGenerator
            hcfg = yaml.safe_load(open(SEEDVC_DIR / "configs" / "hifigan.yml", "r"))
            gen = HiFTGenerator(**hcfg["hift"], f0_predictor=ConvRNNF0Predictor(**hcfg["f0_predictor"]))
            gen.load_state_dict(torch.load(hf_hub_download("FunAudioLLM/CosyVoice-300M", "hift.pt"),
                                           map_location="cpu"))
            return gen.eval().to(self.device)
        raise ValueError(f"unsupported vocoder {mp.vocoder.type!r} (bigvgan | hifigan)")

    def _load_content_encoder(self, mp, cfg, torch):
        dev = self.device
        if mp.speech_tokenizer.type == "whisper":
            from transformers import AutoFeatureExtractor, WhisperModel
            name = mp.speech_tokenizer.name
            wm = WhisperModel.from_pretrained(name, torch_dtype=torch.float16 if self.fp16 else torch.float32).to(dev)
            del wm.decoder
            fe = AutoFeatureExtractor.from_pretrained(name)
            self.whisper, self.whisper_fe = wm, fe

            def semantic_fn(w16):
                inp = fe([w16.squeeze(0).cpu().numpy()], return_tensors="pt", return_attention_mask=True,
                         sampling_rate=16000)
                feats = wm._mask_input_features(inp.input_features, attention_mask=inp.attention_mask).to(dev)
                out = wm.encoder(feats.to(wm.encoder.dtype), head_mask=None, output_attentions=False,
                                 output_hidden_states=False, return_dict=True)
                return out.last_hidden_state.float()[:, : w16.size(-1) // 320 + 1]
            return semantic_fn
        if mp.speech_tokenizer.type == "xlsr":
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            name = cfg["model_params"]["speech_tokenizer"]["name"]
            layer = cfg["model_params"]["speech_tokenizer"]["output_layer"]
            fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
            wm = Wav2Vec2Model.from_pretrained(name)
            wm.encoder.layers = wm.encoder.layers[:layer]
            wm = wm.to(dev).eval()
            if self.fp16:
                wm = wm.half()

            def semantic_fn(w16):
                inp = fe([w16[0].cpu().numpy()], return_tensors="pt", return_attention_mask=True,
                         padding=True, sampling_rate=16000).to(dev)
                x = inp.input_values.half() if self.fp16 else inp.input_values
                return wm(x).last_hidden_state.float()
            return semantic_fn
        raise ValueError(f"unsupported content encoder {mp.speech_tokenizer.type!r}")


class _GraphedSolver:
    """Captures the whole N-step Euler flow-matching solve as one CUDA graph.

    In streaming every DiT input has a fixed shape, so the ~30 small kernels
    per transformer layer per step can be replayed without Python or launch
    overhead. The only host sync in the DiT (``sequence_mask`` building an
    ``arange`` from a tensor length) is replaced by a static all-ones mask —
    valid because streaming never pads. Fresh noise is still drawn on every
    replay (PyTorch's CUDA RNG is graph-safe).
    """

    def __init__(self, model: SeedVCModel, mu, x_lens, prompt, style, steps: int, cfg_rate: float):
        import torch
        with _seedvc_imports():
            import modules.diffusion_transformer as dtm
        est = model.model.cfm.estimator
        seq_len = int(x_lens[0]) + int(est.style_as_token) + int(est.time_as_token)
        self.key = (tuple(mu.shape), tuple(prompt.shape), steps, cfg_rate)
        self.mu, self.prompt, self.style, self.x_lens = mu.clone(), prompt.clone(), style.clone(), x_lens.clone()
        fp16 = model.fp16

        def run():
            with torch.autocast("cuda", dtype=torch.float16 if fp16 else torch.float32, cache_enabled=False):
                return model.model.cfm.inference(self.mu, self.x_lens, self.prompt, self.style, None,
                                                 n_timesteps=steps, inference_cfg_rate=cfg_rate)

        orig_mask = dtm.sequence_mask
        dtm.sequence_mask = lambda length, max_length=None: torch.ones(
            (length.shape[0], seq_len), dtype=torch.bool, device=length.device)
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = run()
        finally:
            dtm.sequence_mask = orig_mask

    def __call__(self, mu, prompt, style):
        self.mu.copy_(mu)
        self.prompt.copy_(prompt)
        self.style.copy_(style)
        self.graph.replay()
        return self.out.clone()


@dataclass
class StreamConfig:
    reference: str = ""
    diffusion_steps: int = 8
    inference_cfg_rate: float = 0.7
    block_time: float = 0.30
    crossfade_time: float = 0.04
    extra_time_ce: float = 2.5       # content-encoder left context (s)
    extra_time: float = 0.5          # DiT left context (s); ce_dit gap = extra_time_ce - extra_time
    extra_time_right: float = 0.02   # right look-ahead (s)
    max_prompt_length: float = 3.0   # seconds of reference used as DiT prompt
    silence_threshold_db: float = -55.0
    cuda_graph: bool = True          # replay the DiT solve as a CUDA graph

    @classmethod
    def from_spec(cls, spec: dict) -> "StreamConfig":
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in spec.items() if k in names})


class SeedVCStreamer:
    """Fixed-block streaming converter. Call :meth:`process_block` with exactly
    ``block_frame`` samples at ``device_sr``; it returns the same amount."""

    def __init__(self, model: SeedVCModel, device_sr: int, cfg: StreamConfig):
        import torch
        import torch.nn.functional as F
        import torchaudio.transforms as tat

        if cfg.extra_time_ce < cfg.extra_time:
            raise ValueError("extra_time_ce must be >= extra_time")
        self.torch, self.F = torch, F
        self.m, self.cfg, self.sr = model, cfg, device_sr
        dev = model.device
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
        self.input_wav = torch.zeros(total, device=dev)
        self.input_wav_res = torch.zeros(320 * total // zc, device=dev)
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=dev)
        self.skip_head = self.extra_frame // zc
        self.skip_tail = self.extra_frame_right // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc
        self.fade_in = torch.sin(0.5 * np.pi * torch.linspace(0, 1, self.sola_buffer_frame, device=dev)) ** 2
        self.fade_out = 1 - self.fade_in
        self.to16k = tat.Resample(device_sr, 16000).to(dev)
        self.from_model = tat.Resample(model.sr, device_sr).to(dev) if model.sr != device_sr else None
        self.prompt = None
        self._solver = None
        self._prev_loud = False
        self.last_infer_ms = 0.0
        if cfg.reference:
            self.set_reference(cfg.reference)

    @property
    def latency(self) -> int:
        return self.extra_frame_right + self.sola_buffer_frame + self.sola_search_frame // 2

    def set_reference(self, path: str) -> None:
        import librosa
        import torchaudio
        torch, m = self.torch, self.m
        wav, _ = librosa.load(path, sr=m.sr)
        wav = torch.from_numpy(wav[: int(m.sr * self.cfg.max_prompt_length)]).to(m.device)
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):  # Seed-VC prints mel stats
            w16 = torchaudio.functional.resample(wav, m.sr, 16000)
            s_ori = m.semantic_fn(w16.unsqueeze(0))
            fb = torchaudio.compliance.kaldi.fbank(w16.unsqueeze(0), num_mel_bins=80, dither=0,
                                                   sample_frequency=16000)
            style = m.campplus((fb - fb.mean(dim=0, keepdim=True)).unsqueeze(0))
            mel = m.to_mel(wav.unsqueeze(0))
            cond = m.model.length_regulator(s_ori, ylens=torch.LongTensor([mel.size(2)]).to(m.device),
                                            n_quantizers=3, f0=None)[0]
        self.prompt = (cond, mel, style)
        self.cfg.reference = path

    def _infer(self):
        torch, m, cfg = self.torch, self.m, self.cfg
        cond_p, mel_p, style = self.prompt
        s_alt = m.semantic_fn(self.input_wav_res.unsqueeze(0))
        ce_dit_frames = int((cfg.extra_time_ce - cfg.extra_time) * 50)   # content-encoder vs DiT context gap
        s_alt = s_alt[:, ce_dit_frames:]
        tlen = torch.LongTensor([int((self.skip_head + self.return_length + self.skip_tail - ce_dit_frames)
                                     / 50 * m.sr // m.hop)]).to(m.device)
        cond = m.model.length_regulator(s_alt, ylens=tlen, n_quantizers=3, f0=None)[0]
        cat = torch.cat([cond_p, cond], dim=1)
        x_lens = torch.LongTensor([cat.size(1)]).to(m.device)
        use_graph = cfg.cuda_graph and m.device.type == "cuda"
        with torch.autocast(device_type=m.device.type, dtype=torch.float16 if m.fp16 else torch.float32,
                            enabled=m.device.type == "cuda"):
            if use_graph:
                key = (tuple(cat.shape), tuple(mel_p.shape), cfg.diffusion_steps, cfg.inference_cfg_rate)
                if self._solver is None or self._solver.key != key:
                    self._solver = None  # release the old graph's memory pool first
                    self._solver = _GraphedSolver(m, cat, x_lens, mel_p, style, cfg.diffusion_steps,
                                                  cfg.inference_cfg_rate)
                mel = self._solver(cat, mel_p, style)
            else:
                mel = m.model.cfm.inference(cat, x_lens, mel_p, style, None, n_timesteps=cfg.diffusion_steps,
                                            inference_cfg_rate=cfg.inference_cfg_rate)
            mel = mel[:, :, mel_p.size(-1):]
            wave = m.vocoder(mel).squeeze().float()
        out_len = self.return_length * m.sr // 50
        tail = self.skip_tail * m.sr // 50
        wave = wave[-out_len - tail: -tail]
        return self.from_model(wave) if self.from_model is not None else wave

    def process_block(self, block: np.ndarray) -> np.ndarray:
        torch, F = self.torch, self.F
        if self.prompt is None:
            raise RuntimeError("no reference voice set; call set_reference(path)")
        n = self.block_frame
        assert len(block) == n, (len(block), n)
        x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).to(self.m.device)
        self.input_wav = torch.roll(self.input_wav, -n)
        self.input_wav[-n:] = x
        self.input_wav_res = torch.roll(self.input_wav_res, -self.block_frame_16k)
        seg = self.input_wav[-n - 2 * self.zc:]
        self.input_wav_res[-320 * (n // self.zc + 1):] = self.to16k(seg)[320:]

        rms_db = 20 * np.log10(float(np.sqrt(np.mean(block**2))) + 1e-9)
        loud = rms_db > self.cfg.silence_threshold_db
        t0 = time.perf_counter()
        if loud or self._prev_loud:
            with torch.no_grad():
                infer = self._infer()
        else:  # silence: skip the GPU entirely
            infer = torch.zeros(self.return_length * self.zc, device=self.m.device)
        self._prev_loud = loud
        self.last_infer_ms = (time.perf_counter() - t0) * 1000

        # SOLA (from DDSP-SVC via Seed-VC): find best-aligned offset, then crossfade.
        conv_in = infer[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        num = F.conv1d(conv_in, self.sola_buffer[None, None, :])
        den = torch.sqrt(F.conv1d(conv_in**2, torch.ones(1, 1, self.sola_buffer_frame, device=self.m.device)) + 1e-8)
        off = int(torch.argmax(num[0, 0] / den[0, 0]).item())
        infer = infer[off:].clone()
        infer[: self.sola_buffer_frame] = infer[: self.sola_buffer_frame] * self.fade_in + self.sola_buffer * self.fade_out
        self.sola_buffer[:] = infer[n: n + self.sola_buffer_frame]
        return infer[:n].cpu().numpy().astype(np.float64)


class SeedVCProcessor:
    """Adapts the fixed-block streamer to arbitrary host block sizes.

    ``mode="inline"``   — inference runs inline (host should use
                              ``preferred_blocksize``); lowest latency.
    ``mode="threaded"`` — a worker thread converts while the audio
                              callback only moves samples; adds one block of
                              latency but never blocks the callback.
    """

    def __init__(self, spec: dict, sr: int, mode: str = "inline", device: str | None = None):
        self.spec = spec
        self.name = spec.get("name", "AI voice")
        self.cfg = StreamConfig.from_spec(spec)
        if spec.get("backend") == "onnx":  # torch-free: ONNX Runtime + numpy (standalone build)
            from .onnx_backend import NumpyStreamer, OnnxSeedVCModel
            self.model = OnnxSeedVCModel.get(spec.get("onnx_dir", f"models/onnx/{spec.get('model')}"),
                                           spec.get("providers"))
            self.streamer = NumpyStreamer(self.model, sr, self.cfg)
        else:
            self.model = SeedVCModel.get(spec.get("model", "whisper-small-bigvgan"), device=device,
                                       checkpoint=spec.get("checkpoint"), config=spec.get("config"))
            self.streamer = SeedVCStreamer(self.model, sr, self.cfg)
        self.mode = mode
        self.sr = sr
        # inline: the host must call us with whole blocks; threaded
        # decouples, so the host keeps small low-latency buffers.
        self.preferred_blocksize = self.streamer.block_frame if mode == "inline" else None
        self.headroom = float(spec.get("rtf_headroom", 0.9))  # inference time as a fraction of a block
        self.error: str | None = None  # set by the worker if conversion fails
        self._in = np.zeros(0)
        self._out = np.zeros(self._prefill())
        self._q_in: queue.Queue = queue.Queue()
        self._q_out: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        if mode == "threaded":
            self._worker = threading.Thread(target=self._run, name="seedvc-worker", daemon=True)
            self._worker.start()

    def _prefill(self) -> int:
        """Output delay that guarantees a block is converted before it is due:
        one block to accumulate + the worst-case inference time."""
        n = self.streamer.block_frame
        return n if self.mode == "inline" else n + int(self.headroom * n)

    @property
    def latency(self) -> int:
        return self.streamer.latency + self._prefill()

    def set_reference(self, path: str) -> None:
        with self._lock:
            self.streamer.set_reference(path)

    def warmup(self) -> None:
        if self.streamer.prompt is None:
            return
        block = np.random.default_rng(0).standard_normal(self.streamer.block_frame) * 0.05
        with self._lock:
            for _ in range(2):  # 1st call compiles kernels for these shapes; time the 2nd
                self.streamer.process_block(block)
            self.streamer._prev_loud = False
        if self.mode == "threaded":  # size the prefill from measured speed (+35 % margin)
            n = self.streamer.block_frame
            measured = self.streamer.last_infer_ms / 1000 * self.sr / n
            self.headroom = float(np.clip(1.35 * measured, 0.3, 1.0))
            self._out = np.zeros(self._prefill())

    def _run(self):
        while not self._stop.is_set():
            try:
                blk = self._q_in.get(timeout=0.1)
            except queue.Empty:
                continue
            # if conversion falls behind (hot / throttled GPU), skip stale audio instead of
            # queueing it without bound: memory stays flat and latency recovers
            while self._q_in.qsize() > 1:
                blk = self._q_in.get_nowait()
            try:
                with self._lock:
                    self._q_out.put(self.streamer.process_block(blk))
            except Exception as e:  # e.g. GPU device removed: report, stop converting
                import logging
                logging.getLogger(__name__).exception("AI voice conversion failed")
                self.error = f"{type(e).__name__}: {e}"
                return

    def close(self):
        self._stop.set()

    def process(self, x: np.ndarray) -> np.ndarray:
        n = self.streamer.block_frame
        self._in = np.concatenate([self._in, x])
        while len(self._in) >= n:
            blk, self._in = self._in[:n], self._in[n:]
            if self.mode == "inline":
                with self._lock:
                    self._out = np.concatenate([self._out, self.streamer.process_block(blk)])
            else:
                self._q_in.put(blk)
        while not self._q_out.empty():
            self._out = np.concatenate([self._out, self._q_out.get_nowait()])
        if len(self._out) < len(x):  # worker fell behind: emit silence rather than glitching
            self._out = np.concatenate([np.zeros(len(x) - len(self._out)), self._out])
        y, self._out = self._out[: len(x)], self._out[len(x):]
        return y
