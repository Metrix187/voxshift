"""Voice graphs: JSON presets → node chains with modulated parameters.

A voice is a graph of effect nodes whose parameters can be driven by
generators (oscillators, random walks, sequences). The JSON schema::

    {
      "name": "Robot",
      "chain": [
        {"type": "Vocoder", "params": {"carrier_hz": 110}},
        {"type": "RingModulation",
         "params": {"freq": {"lfo": {"rate": 0.3, "min": 30, "max": 60}}}},
        {"type": "Parallel", "branches": [[...], [...]], "gains_db": [0, -6]}
      ]
    }

Modulated values (``lfo`` / ``random`` / ``sequence``) are re-evaluated once
per processed block.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .nodes import NODE_TYPES, LatencyDelay, Node, db_to_lin


class Modulator:
    def value(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class LFO(Modulator):
    """Periodic modulator (sine / triangle / square / saw)."""

    def __init__(self, rate=1.0, min=0.0, max=1.0, shape="sine", phase=0.0):
        self.rate, self.lo, self.hi, self.shape, self.phase = rate, min, max, shape, phase

    def value(self, t):
        ph = (t * self.rate + self.phase) % 1.0
        if self.shape == "sine":
            u = 0.5 * (1 - math.cos(2 * math.pi * ph))
        elif self.shape == "triangle":
            u = 1 - abs(2 * ph - 1)
        elif self.shape == "square":
            u = 1.0 if ph < 0.5 else 0.0
        else:  # saw
            u = ph
        return self.lo + (self.hi - self.lo) * u


class RandomWalk(Modulator):
    """Random target values with optional smoothing."""

    def __init__(self, min=0.0, max=1.0, interval_s=0.5, smooth=0.5, seed=0):
        self.lo, self.hi, self.interval, self.smooth = min, max, interval_s, smooth
        self.rng = np.random.default_rng(seed)
        self.cur = self.target = (min + max) / 2
        self.next_t = 0.0

    def value(self, t):
        if t >= self.next_t:
            self.target = float(self.rng.uniform(self.lo, self.hi))
            self.next_t = t + self.interval
        self.cur = self.smooth * self.cur + (1 - self.smooth) * self.target
        return self.cur


class Sequence(Modulator):
    """Steps through a list of values."""

    def __init__(self, values, interval_s=0.25):
        self.values, self.interval = list(values), interval_s

    def value(self, t):
        return self.values[int(t / self.interval) % len(self.values)]


MODULATORS = {"lfo": LFO, "random": RandomWalk, "sequence": Sequence}


def _split_params(params: dict):
    static, mods = {}, {}
    for k, v in params.items():
        if isinstance(v, dict) and len(v) == 1 and next(iter(v)) in MODULATORS:
            kind, cfg = next(iter(v.items()))
            mods[k] = MODULATORS[kind](**cfg)
        else:
            static[k] = v
    return static, mods


@dataclass
class _Slot:
    node: "Node | Parallel"
    mods: dict = field(default_factory=dict)
    bypass: bool = False


class Parallel:
    """Runs branches side by side, aligns their latency and sums them."""

    def __init__(self, sr, branches, gains_db=None):
        self.chains = [Chain(sr, b) for b in branches]
        self.gains = [db_to_lin(g) for g in (gains_db or [0.0] * len(branches))]
        lat = max(c.latency for c in self.chains)
        for c in self.chains:
            if c.latency < lat:
                c.slots.append(_Slot(LatencyDelay(sr, samples=lat - c.latency)))
        self.latency = lat

    def process(self, x, t):
        return sum(g * c.process(x, t) for g, c in zip(self.gains, self.chains))


class Chain:
    def __init__(self, sr: int, spec: list[dict]):
        self.sr = sr
        self.slots: list[_Slot] = []
        for item in spec:
            kind = item["type"]
            if kind == "Parallel":
                self.slots.append(_Slot(Parallel(sr, item["branches"], item.get("gains_db"))))
                continue
            if kind not in NODE_TYPES:
                raise ValueError(f"unknown node type {kind!r}; known: {sorted(NODE_TYPES)}")
            static, mods = _split_params(item.get("params", {}))
            node = NODE_TYPES[kind](sr, **static)
            if mods:  # initialise modulated params at t=0
                node.set(**{k: m.value(0.0) for k, m in mods.items()})
            self.slots.append(_Slot(node, mods, item.get("bypass", False)))

    @property
    def latency(self) -> int:
        return sum(s.node.latency for s in self.slots if not s.bypass)

    def process(self, x: np.ndarray, t: float) -> np.ndarray:
        for s in self.slots:
            if s.bypass:
                continue
            if isinstance(s.node, Parallel):
                x = s.node.process(x, t)
                continue
            if s.mods:
                s.node.set(**{k: m.value(t) for k, m in s.mods.items()})
            x = s.node.process(x)
        return x


class Voice:
    """A loaded voice preset, ready to process a mono stream."""

    def __init__(self, spec: dict, sr: int):
        self.spec = spec
        self.name = spec.get("name", "Unnamed")
        self.sr = sr
        self.chain = Chain(sr, spec.get("chain", []))
        self.t = 0.0

    @property
    def latency(self) -> int:
        return self.chain.latency

    def process(self, x: np.ndarray) -> np.ndarray:
        y = self.chain.process(np.asarray(x, dtype=np.float64), self.t)
        self.t += len(x) / self.sr
        return y


VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"


def list_voices(extra_dirs: list[Path] | None = None) -> dict[str, Path]:
    found = {}
    for d in [VOICES_DIR, *(extra_dirs or [])]:
        for p in sorted(Path(d).glob("*.json")):
            found[p.stem] = p
    return found


def load_voice(name_or_path: str, sr: int, extra_dirs: list[Path] | None = None) -> Voice:
    p = Path(name_or_path)
    if not p.is_file():
        voices = list_voices(extra_dirs)
        if name_or_path not in voices:
            raise KeyError(f"voice {name_or_path!r} not found; available: {', '.join(voices)}")
        p = voices[name_or_path]
    return Voice(json.loads(p.read_text(encoding="utf-8")), sr)
