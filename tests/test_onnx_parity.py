"""ONNX Runtime vs PyTorch parity for every exported Seed-VC network.

Integration test: needs the Seed-VC weights and `python -m voxshift export-onnx`.
"""
import numpy as np
import pytest

from voxshift.ai.seedvc import REPO_ROOT

ONNX_DIR = REPO_ROOT / "models" / "onnx" / "whisper-small-bigvgan"
pytestmark = pytest.mark.skipif(not (ONNX_DIR / "config.json").is_file(), reason="run export-onnx first")


@pytest.fixture(scope="module", params=["CPUExecutionProvider", "DmlExecutionProvider"])
def pair(request):
    import onnxruntime as ort
    if request.param not in ort.get_available_providers():
        pytest.skip(f"{request.param} unavailable")
    from voxshift.ai.onnx_backend import OnnxSeedVCModel
    from voxshift.ai.seedvc import SeedVCModel
    return SeedVCModel.get("whisper-small-bigvgan", device="cpu", fp16=False), \
        OnnxSeedVCModel(str(ONNX_DIR), [request.param])


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def test_networks_match(pair):
    import torch
    tm, om = pair
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        w16 = torch.randn(1, 16000 * 3, generator=g) * 0.1
        fp16 = om.cfg.get("precision", {})
        tol = lambda net: 3e-2 if fp16.get(net) == "fp16" else 1e-2  # noqa: E731
        assert rel_err(om.content(w16[0].numpy()), tm.semantic_fn(w16)) < tol("whisper_encoder")

        s = torch.randn(1, 120, 768, generator=g)
        yl = torch.LongTensor([240])
        assert rel_err(om.length_regulator(s.numpy(), 240),
                       tm.model.length_regulator(s, ylens=yl, n_quantizers=3, f0=None)[0]) < 1e-3

        for frames in (120, 257, 300, 451):  # include partial pooling segments
            fb = torch.randn(1, frames, 80, generator=g)
            assert rel_err(om._run("campplus", fbank=fb.numpy()), tm.campplus(fb)) < 1e-3, frames

        mel = torch.randn(1, 80, 64, generator=g)
        assert rel_err(om.vocoder(mel.numpy()), tm.vocoder(mel).reshape(-1)) < tol("vocoder")

        T = 200
        est = tm.model.cfm.estimator
        args = (torch.randn(2, 80, T, generator=g), torch.randn(2, 80, T, generator=g), torch.rand(2, generator=g),
                torch.randn(2, 192, generator=g), torch.randn(2, T, 512, generator=g))
        ref = est(args[0], args[1], torch.LongTensor([T]), args[2], args[3], args[4])
        got = om._run("dit_step", x=args[0].numpy(), prompt_x=args[1].numpy(), t=args[2].numpy(),
                      style=args[3].numpy(), mu=args[4].numpy())
        assert rel_err(got, ref) < tol("dit_step")
