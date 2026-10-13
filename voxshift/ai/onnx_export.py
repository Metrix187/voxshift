"""Export the Seed-VC networks to ONNX (``voxshift export-onnx``).

Splits the open Seed-VC Whisper-small + BigVGAN checkpoint into networks
that ONNX Runtime can run without PyTorch:

    whisper_encoder.onnx    log-mel [1,80,3000]            -> hidden [1,1500,768]
    length_regulator.onnx   hidden [1,S,768], ylens [1]     -> cond [1,T,512]
    dit_step.onnx           x, prompt_x [B,80,T], t [B],
                            style [B,192], mu [B,T,512]     -> dphi/dt [B,80,T]
    campplus.onnx           fbank [1,F,80]                  -> style [1,192]
    vocoder.onnx            mel [1,80,T]                    -> wave [1,1,T*hop]
    config.json             sample rates, hop, Whisper front-end settings

The Euler loop, CFG mix, resampling and feature extraction stay in the host
(``onnx_backend.py``).
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .seedvc import SeedVCModel


class _WhisperEncoder(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feats):
        return self.enc(feats, return_dict=True).last_hidden_state


class _LengthRegulator(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def forward(self, x, ylens):
        return self.lr(x, ylens=ylens, n_quantizers=3, f0=None)[0]


class _DiTStep(nn.Module):
    def __init__(self, est):
        super().__init__()
        self.est = est

    def forward(self, x, prompt_x, t, style, mu):
        # `sequence_mask` builds `arange(length.max())`, which the tracer freezes
        # at the export length. Streaming never pads, so derive an all-ones mask
        # from x's (dynamic) shape instead.
        # Likewise SConv1d's `get_extra_padding_for_conv1d` runs math.ceil() on
        # the traced length (freezing `extra = export_len - T`); for stride-1
        # convs (all of the WaveNet head) the extra padding is exactly 0.
        import torch.nn.functional as F
        from modules import diffusion_transformer as dtm
        from modules import encodec
        extra = int(self.est.style_as_token) + int(self.est.time_as_token)
        row = F.pad(torch.ones_like(x[:1, 0, :]), (extra, 0), value=1.0) > 0
        saved_mask, saved_pad = dtm.sequence_mask, encodec.get_extra_padding_for_conv1d
        dtm.sequence_mask = lambda length, max_length=None: row
        encodec.get_extra_padding_for_conv1d = (
            lambda xx, k, s, p=0: 0 if s == 1 else saved_pad(xx, k, s, p))
        try:
            return self.est(x, prompt_x, torch.zeros(1, dtype=torch.long), t, style, mu)
        finally:
            dtm.sequence_mask, encodec.get_extra_padding_for_conv1d = saved_mask, saved_pad


class _Vocoder(nn.Module):
    def __init__(self, voc):
        super().__init__()
        self.voc = voc

    def forward(self, mel):
        return self.voc(mel)


@contextlib.contextmanager
def _exportable_bigvgan():
    """BigVGAN's anti-aliased resamplers apply one filter to every channel via
    ``filter.expand(C, -1, -1)`` with C read from the traced input shape, so
    the exporter sees a depthwise kernel of unknown shape. The channel count
    is fixed by the architecture; pinning it as a Python int yields a
    constant ``[C, 1, K]`` kernel and an ordinary grouped (depthwise) conv,
    which every execution provider runs efficiently."""
    import torch.nn.functional as F
    from modules.bigvgan.alias_free_activation.torch import filter as filt
    from modules.bigvgan.alias_free_activation.torch import resample

    def up(self, x):
        C = int(x.shape[1])
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        w = self.filter.expand(C, -1, -1).contiguous()
        y = self.ratio * F.conv_transpose1d(x, w, stride=self.stride, groups=C)
        return y[..., self.pad_left: -self.pad_right]

    def lowpass(self, x):
        C = int(x.shape[1])
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(x, self.filter.expand(C, -1, -1).contiguous(), stride=self.stride, groups=C)

    saved = resample.UpSample1d.forward, filt.LowPassFilter1d.forward
    resample.UpSample1d.forward, filt.LowPassFilter1d.forward = up, lowpass
    try:
        yield
    finally:
        resample.UpSample1d.forward, filt.LowPassFilter1d.forward = saved


@contextlib.contextmanager
def _exportable_campplus():
    """CAMPPlus ``seg_pooling`` uses ``avg_pool1d(ceil_mode=True)``; PyTorch
    averages a trailing partial window over its valid samples only, which the
    exported graph does not reproduce for lengths that are not multiples of
    the segment. Re-express it as pad + reshape + sum / valid-count."""
    import torch.nn.functional as F
    from modules.campplus import layers

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        B, C, T = x.shape
        n = (T + seg_len - 1) // seg_len
        xp = F.pad(x, (0, n * seg_len - T))
        blocks = xp.reshape(B, C, n, seg_len)
        if stype == "avg":
            counts = torch.clamp(T - torch.arange(n, device=x.device) * seg_len, max=seg_len).to(x.dtype)
            seg = blocks.sum(-1) / counts
        else:
            seg = blocks.amax(-1)
        return seg.unsqueeze(-1).expand(B, C, n, seg_len).reshape(B, C, n * seg_len)[..., :T]

    saved = layers.CAMLayer.seg_pooling
    layers.CAMLayer.seg_pooling = seg_pooling
    try:
        yield
    finally:
        layers.CAMLayer.seg_pooling = saved


def _rel(a, b) -> float:
    """Relative L2 error; the rewrites only differ by float rounding (~1e-6)."""
    return float((a - b).norm() / (b.norm() + 1e-12))


def _export(module, args, path, inputs, outputs, dynamic, opset):
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs,
                      dynamic_axes=dynamic, opset_version=opset, do_constant_folding=True, dynamo=False)
    print(f"  wrote {path.name} ({path.stat().st_size / 2**20:.1f} MB)")


def _to_fp16(path: Path) -> None:
    """Store weights/compute in fp16 but keep fp32 graph I/O, so the host
    code is precision-agnostic."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    model = onnx.load(str(path))
    model.graph.ClearField("value_info")  # stale fp32 annotations break the converter's Cast handling
    model = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model, str(path))
    print(f"  converted {path.name} to fp16 ({path.stat().st_size / 2**20:.1f} MB)")


def export(preset: str = "whisper-small-bigvgan", out_dir: str = "models/onnx", opset: int = 17,
           fp16: bool = True) -> Path:
    m = SeedVCModel.get(preset, device="cpu", fp16=False)
    if m.content_type != "whisper" or m.vocoder_type != "bigvgan":
        raise SystemExit("ONNX export supports the Whisper + BigVGAN model; "
                         "HiFT's complex STFT/ISTFT has no ONNX lowering.")
    out = Path(out_dir).resolve() / preset
    out.mkdir(parents=True, exist_ok=True)
    print(f"exporting {preset} -> {out}")
    fe = m.whisper_fe
    T, S = 300, 150
    with torch.no_grad():
        enc = m.whisper.encoder.float().eval()
        _export(_WhisperEncoder(enc), (torch.randn(1, 80, fe.nb_max_frames),), out / "whisper_encoder.onnx",
                ["input_features"], ["hidden"], None, opset)
        _export(_LengthRegulator(m.model.length_regulator), (torch.randn(1, S, 768), torch.LongTensor([T])),
                out / "length_regulator.onnx", ["content", "ylens"], ["cond"],
                {"content": {1: "S"}, "cond": {1: "T"}}, opset)
        est = m.model.cfm.estimator
        dit_args = (torch.randn(2, 80, T), torch.randn(2, 80, T), torch.rand(2), torch.randn(2, 192),
                    torch.randn(2, T, 512))
        # The DiT adds python bools to a tensor (`x_lens + self.style_as_token`),
        # which the TorchScript tracer cannot type; ints behave identically.
        flags = {k: getattr(est, k) for k in ("style_as_token", "time_as_token")}
        for k, v in flags.items():
            setattr(est, k, int(v))
        try:
            _export(_DiTStep(est), dit_args, out / "dit_step.onnx", ["x", "prompt_x", "t", "style", "mu"],
                    ["dphi"], {"x": {0: "B", 2: "T"}, "prompt_x": {0: "B", 2: "T"}, "t": {0: "B"},
                               "style": {0: "B"}, "mu": {0: "B", 1: "T"}, "dphi": {0: "B", 2: "T"}}, opset)
        finally:
            for k, v in flags.items():
                setattr(est, k, v)
        with _exportable_campplus():
            fb = torch.randn(1, 257, 80)
            ref = m.campplus(fb)
            _export(m.campplus, (fb,), out / "campplus.onnx", ["fbank"], ["style"], {"fbank": {1: "F"}}, opset)
        assert _rel(ref, m.campplus(fb)) < 1e-4, "CAMPPlus rewrite changed the output"
        with _exportable_bigvgan():
            voc = _Vocoder(m.vocoder)
            mel = torch.randn(1, 80, 100)
            ref = voc(mel)
            _export(voc, (mel,), out / "vocoder.onnx", ["mel"], ["wave"], {"mel": {2: "T"}, "wave": {2: "N"}}, opset)
        assert _rel(ref, m.vocoder(mel)) < 1e-4, "BigVGAN rewrite changed the output"
    if fp16:  # the transformer stacks dominate runtime; BigVGAN stays fp32 for its snake activations
        for name in ("whisper_encoder", "dit_step"):
            _to_fp16(out / f"{name}.onnx")
    sp = m.config["preprocess_params"]["spect_params"]
    cfg = {
        "model": preset,
        "sr": m.sr, "hop_length": m.hop, "n_fft": sp["n_fft"], "win_length": sp["win_length"],
        "n_mels": sp["n_mels"], "fmin": sp.get("fmin", 0), "fmax": sp.get("fmax", "None"),
        "content_encoder": "whisper", "whisper_name": m.config["model_params"]["speech_tokenizer"]["name"],
        "whisper_sr": 16000, "whisper_hop_length": fe.hop_length, "whisper_n_fft": fe.n_fft,
        "whisper_n_frames": fe.nb_max_frames, "vocoder": "bigvgan",
        "dit_extra_tokens": int(est.style_as_token) + int(est.time_as_token),
        "precision": {"whisper_encoder": "fp16" if fp16 else "fp32", "dit_step": "fp16" if fp16 else "fp32",
                      "length_regulator": "fp32", "campplus": "fp32", "vocoder": "fp32"},
    }
    from .features import build_constants
    np.savez(out / "features.npz", **build_constants(fe.mel_filters, m.sr, sp["n_fft"], sp["n_mels"],
                                                       sp.get("fmin", 0), None if sp.get("fmax", "None") == "None"
                                                       else 8000))
    print("  wrote features.npz (Whisper / mel / Kaldi filter banks for the numpy front-ends)")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print("  wrote config.json")
    return out
