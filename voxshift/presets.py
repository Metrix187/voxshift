"""Voice preset discovery and processor construction (DSP graph or AI)."""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

from .dsp.graph import Voice
from .paths import models_dir, references_dir, user_voices_dir

BUILTIN_DIR = Path(__file__).resolve().parent / "voices"
ONNX_MODEL = "whisper-small-bigvgan"


def preset_dirs() -> list[Path]:
    return [BUILTIN_DIR, user_voices_dir()]


def is_ai(spec: dict) -> bool:
    return spec.get("engine") == "seedvc"


def unavailable_reason(spec: dict) -> str | None:
    """Why a voice cannot run in this installation (None = it can)."""
    if not is_ai(spec):
        return None
    if spec.get("backend") == "onnx":
        d = models_dir() / "onnx" / spec.get("model", ONNX_MODEL)
        return None if (d / "config.json").is_file() else "AI model files are not installed"
    if importlib.util.find_spec("torch") is None:
        return "needs the PyTorch developer setup"
    return None


def list_presets(include_unavailable: bool = False) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for d in preset_dirs():
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                spec = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            spec["_path"] = str(p)
            spec["_user"] = d != BUILTIN_DIR
            if include_unavailable or unavailable_reason(spec) is None:
                found[p.stem] = spec
    return found


def load_spec(name_or_path: str) -> dict:
    p = Path(name_or_path)
    if p.suffix == ".json" and p.is_file():
        spec = json.loads(p.read_text(encoding="utf-8"))
        spec["_path"] = str(p)
        return spec
    presets = list_presets(include_unavailable=True)
    if name_or_path not in presets:
        raise KeyError(f"voice {name_or_path!r} not found; available: {', '.join(presets)}")
    return presets[name_or_path]


def create_ai_voice(name: str, reference: str, base: str = "ai-whisper-small-onnx") -> str:
    """Save a personal AI voice ("Voicelab"): copy the reference recording into
    the user data folder and write a preset pointing at it. Returns its key."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "voice"
    key = f"my-{slug}"
    ref = Path(reference)
    dest = references_dir() / f"{key}{ref.suffix.lower() or '.wav'}"
    if ref.resolve() != dest.resolve():
        shutil.copyfile(ref, dest)
    spec = {k: v for k, v in load_spec(base).items() if not k.startswith("_") and k != "template"}
    spec.update({"name": name, "category": "my voices", "reference": str(dest),
                 "description": f"Your AI voice from {ref.name}."})
    (user_voices_dir() / f"{key}.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return key


def delete_user_voice(key: str) -> None:
    spec = load_spec(key)
    if not spec.get("_user"):
        raise ValueError("built-in voices cannot be deleted")
    ref = spec.get("reference")
    if ref and Path(ref).parent == references_dir():
        Path(ref).unlink(missing_ok=True)
    Path(spec["_path"]).unlink(missing_ok=True)


def build_processor(spec: dict, sr: int, *, reference: str | None = None,
                    ai_mode: str = "inline", device: str | None = None, **overrides):
    """Return an object with ``process(np.ndarray) -> np.ndarray`` and ``latency``."""
    if is_ai(spec):
        reason = unavailable_reason(spec)
        if reason:
            raise RuntimeError(f"{spec.get('name')}: {reason}")
        from .ai.seedvc import SeedVCProcessor  # heavy import, only when needed
        spec = {**spec, **{k: v for k, v in overrides.items() if v is not None}}
        if reference:
            spec["reference"] = reference
        if not spec.get("reference"):
            raise ValueError("AI voices need a reference recording of the target voice (--reference file.wav)")
        return SeedVCProcessor(spec, sr, mode=ai_mode, device=device)
    return Voice(spec, sr)
