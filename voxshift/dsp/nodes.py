"""DSP node library.

Every node processes mono float64 blocks of any length and keeps its own
state, so a voice is just an ordered graph of nodes driven block-by-block.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from .stft import StreamingSTFT, cepstral_envelope, wrap_phase

NODE_TYPES: dict[str, type["Node"]] = {}


def register(name: str):
    def deco(cls):
        NODE_TYPES[name] = cls
        cls.type_name = name
        return cls
    return deco


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


class Node:
    type_name = "Node"
    defaults: dict = {}

    def __init__(self, sr: int, **params):
        self.sr = sr
        self.params = dict(self.defaults)
        self.params.update(params)
        self.setup()

    def setup(self) -> None:
        pass

    def set(self, **params) -> None:
        changed = {k: v for k, v in params.items() if self.params.get(k) != v}
        if changed:
            self.params.update(changed)
            self.on_change(changed)

    def on_change(self, changed: dict) -> None:
        pass

    @property
    def latency(self) -> int:
        return 0

    def process(self, x: np.ndarray) -> np.ndarray:
        return x


# --------------------------------------------------------------------------- basic

@register("Gain")
class Gain(Node):
    defaults = {"gain_db": 0.0}

    def process(self, x):
        return x * db_to_lin(self.params["gain_db"])


@register("Biquad")
class Biquad(Node):
    """RBJ cookbook biquad filter."""
    defaults = {"kind": "lowpass", "freq": 1000.0, "q": 0.707, "gain_db": 0.0}

    def setup(self):
        self._design()
        self._zi = np.zeros(2)

    def on_change(self, changed):
        self._design()

    def _design(self):
        p = self.params
        f = min(max(float(p["freq"]), 10.0), self.sr * 0.49)
        w0 = 2 * math.pi * f / self.sr
        cw, sw = math.cos(w0), math.sin(w0)
        alpha = sw / (2 * float(p["q"]))
        A = 10 ** (float(p["gain_db"]) / 40)
        k = p["kind"]
        if k == "lowpass":
            b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
        elif k == "highpass":
            b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
        elif k == "bandpass":
            b = [alpha, 0, -alpha]; a = [1 + alpha, -2 * cw, 1 - alpha]
        elif k == "notch":
            b = [1, -2 * cw, 1]; a = [1 + alpha, -2 * cw, 1 - alpha]
        elif k == "peaking":
            b = [1 + alpha * A, -2 * cw, 1 - alpha * A]; a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
        elif k in ("lowshelf", "highshelf"):
            sq = 2 * math.sqrt(A) * alpha
            s = 1 if k == "lowshelf" else -1
            b = [A * ((A + 1) - s * (A - 1) * cw + sq),
                 s * 2 * A * ((A - 1) - s * (A + 1) * cw),
                 A * ((A + 1) - s * (A - 1) * cw - sq)]
            a = [(A + 1) + s * (A - 1) * cw + sq,
                 -s * 2 * ((A - 1) + s * (A + 1) * cw),
                 (A + 1) + s * (A - 1) * cw - sq]
        else:
            raise ValueError(f"unknown biquad kind {k!r}")
        self.b = np.array(b) / a[0]
        self.a = np.array(a) / a[0]

    def process(self, x):
        y, self._zi = lfilter(self.b, self.a, x, zi=self._zi)
        return y


@register("ParametricEQ")
class ParametricEQ(Node):
    """Cascade of biquads."""
    defaults = {"bands": []}

    def setup(self):
        self.filters = [Biquad(self.sr, **b) for b in self.params["bands"]]

    def on_change(self, changed):
        self.setup()

    def process(self, x):
        for f in self.filters:
            x = f.process(x)
        return x


# --------------------------------------------------------------------------- time-domain FX

@register("Delay")
class Delay(Node):
    """Feedback echo."""
    defaults = {"time_ms": 250.0, "feedback": 0.35, "mix": 0.4, "max_ms": 2000.0}

    def setup(self):
        self.size = int(self.sr * self.params["max_ms"] / 1000) + 1
        self.ring = np.zeros(self.size)
        self.w = 0

    def process(self, x):
        d = int(np.clip(self.params["time_ms"] * self.sr / 1000, 1, self.size - 1))
        fb, mix = float(self.params["feedback"]), float(self.params["mix"])
        out = np.empty_like(x)
        pos = 0
        while pos < len(x):
            n = min(d, len(x) - pos)
            idx_r = (self.w - d + np.arange(n)) % self.size
            idx_w = (self.w + np.arange(n)) % self.size
            delayed = self.ring[idx_r]
            chunk = x[pos:pos + n]
            out[pos:pos + n] = chunk + mix * delayed
            self.ring[idx_w] = chunk + fb * delayed
            self.w = (self.w + n) % self.size
            pos += n
        return out


@register("ModDelay")
class ModDelay(Node):
    """LFO-modulated delay taps: chorus / flanger / vibrato."""
    defaults = {"base_ms": 15.0, "depth_ms": 4.0, "rate_hz": 0.8, "voices": 2, "mix": 0.5}

    def setup(self):
        self.hist = np.zeros(int(self.sr * 0.1) + 4)
        self.phase = 0.0

    def process(self, x):
        p = self.params
        n = len(x)
        H = len(self.hist)
        ext = np.concatenate([self.hist, x])
        t = self.phase + np.arange(n) * p["rate_hz"] / self.sr
        wet = np.zeros(n)
        voices = max(1, int(p["voices"]))
        for v in range(voices):
            lfo = np.sin(2 * np.pi * (t + v / voices))
            d = (p["base_ms"] + p["depth_ms"] * lfo) * self.sr / 1000
            d = np.clip(d, 1, H - 2)
            pos = H + np.arange(n) - d
            i0 = np.floor(pos).astype(int)
            fr = pos - i0
            wet += ext[i0] * (1 - fr) + ext[i0 + 1] * fr
        wet /= voices
        self.phase = (self.phase + n * p["rate_hz"] / self.sr) % 1.0
        self.hist = ext[-H:]
        mix = float(p["mix"])
        return x * (1 - mix) + wet * mix


@register("Distortion")
class Distortion(Node):
    """Waveshaping distortion / overdrive."""
    defaults = {"drive_db": 12.0, "shape": "tanh", "mix": 1.0, "output_db": -6.0}

    def process(self, x):
        p = self.params
        g = db_to_lin(p["drive_db"])
        s = p["shape"]
        if s == "tanh":
            y = np.tanh(g * x)
        elif s == "hard":
            y = np.clip(g * x, -1, 1)
        elif s == "fold":
            y = np.sin(g * x * math.pi / 2)
        elif s == "fuzz":
            y = np.sign(x) * (1 - np.exp(-np.abs(g * x)))
        else:
            raise ValueError(s)
        y *= db_to_lin(p["output_db"])
        return x * (1 - p["mix"]) + y * p["mix"]


@register("Bitcrusher")
class Bitcrusher(Node):
    defaults = {"bits": 8, "downsample": 4}

    def setup(self):
        self.hold = 0.0
        self.count = 0

    def process(self, x):
        ds = max(1, int(self.params["downsample"]))
        levels = 2 ** (int(self.params["bits"]) - 1)
        n = len(x)
        latch = (self.count + np.arange(n)) % ds == 0
        # sample-and-hold: each output holds the most recent latched input
        idx = np.maximum.accumulate(np.where(latch, np.arange(n), -1))
        held = np.where(idx >= 0, x[np.maximum(idx, 0)], self.hold)
        if latch.any():
            self.hold = x[idx[-1]]
        self.count += n
        return np.round(held * levels) / levels


@register("RingModulation")
class RingModulation(Node):
    """Carrier multiplication — daleks, robots."""
    defaults = {"freq": 50.0, "mix": 1.0, "waveform": "sine"}

    def setup(self):
        self.phase = 0.0

    def process(self, x):
        n = len(x)
        ph = self.phase + np.arange(n) * self.params["freq"] / self.sr
        self.phase = (self.phase + n * self.params["freq"] / self.sr) % 1.0
        w = self.params["waveform"]
        car = np.sin(2 * np.pi * ph) if w == "sine" else np.sign(np.sin(2 * np.pi * ph))
        m = self.params["mix"]
        return x * (1 - m) + x * car * m


@register("Tremolo")
class Tremolo(Node):
    defaults = {"rate_hz": 6.0, "depth": 0.5}

    def setup(self):
        self.phase = 0.0

    def process(self, x):
        n = len(x)
        ph = self.phase + np.arange(n) * self.params["rate_hz"] / self.sr
        self.phase = (self.phase + n * self.params["rate_hz"] / self.sr) % 1.0
        d = self.params["depth"]
        return x * (1 - d * 0.5 * (1 + np.sin(2 * np.pi * ph)))


@register("Oscillator")
class Oscillator(Node):
    """Adds a generated signal — hum, static, beeps."""
    defaults = {"waveform": "noise", "freq": 60.0, "level_db": -40.0, "follow_input": False}

    def setup(self):
        self.phase = 0.0
        self.rng = np.random.default_rng(1234)

    def process(self, x):
        n = len(x)
        w = self.params["waveform"]
        if w == "noise":
            s = self.rng.standard_normal(n) * 0.3
        else:
            ph = self.phase + np.arange(n) * self.params["freq"] / self.sr
            self.phase = (self.phase + n * self.params["freq"] / self.sr) % 1.0
            s = np.sin(2 * np.pi * ph) if w == "sine" else 2 * (ph % 1.0) - 1
        s *= db_to_lin(self.params["level_db"])
        if self.params["follow_input"]:
            s *= min(1.0, float(np.sqrt(np.mean(x**2)) * 10))
        return x + s


class _Envelope:
    """Sub-block RMS envelope follower shared by gate/compressor."""

    def __init__(self, sr, sub=32):
        self.sr, self.sub, self.env, self.gain = sr, sub, 0.0, 1.0

    def gains(self, x, target_fn, attack_ms, release_ms):
        n = len(x)
        nb = max(1, n // self.sub)
        a = math.exp(-self.sub / (self.sr * max(attack_ms, 0.1) / 1000))
        r = math.exp(-self.sub / (self.sr * max(release_ms, 0.1) / 1000))
        pts = np.empty(nb)
        for i, seg in enumerate(np.array_split(x, nb)):
            lvl = float(np.sqrt(np.mean(seg**2)) + 1e-12)
            tgt = target_fn(20 * math.log10(lvl))
            coef = a if tgt < self.gain else r
            self.gain = coef * self.gain + (1 - coef) * tgt
            pts[i] = self.gain
        return np.interp(np.arange(n), np.linspace(0, n - 1, nb), pts)


@register("NoiseGate")
class NoiseGate(Node):
    """Downward noise gate."""
    defaults = {"threshold_db": -50.0, "attack_ms": 2.0, "release_ms": 120.0, "floor_db": -80.0}

    def setup(self):
        self.envf = _Envelope(self.sr)

    def process(self, x):
        p = self.params
        floor = db_to_lin(p["floor_db"])
        g = self.envf.gains(x, lambda lv: 1.0 if lv > p["threshold_db"] else floor,
                            p["release_ms"], p["attack_ms"])  # open fast, close slow
        return x * g


@register("Compressor")
class Compressor(Node):
    """Feed-forward compressor."""
    defaults = {"threshold_db": -20.0, "ratio": 4.0, "attack_ms": 5.0, "release_ms": 80.0, "makeup_db": 6.0}

    def setup(self):
        self.envf = _Envelope(self.sr)

    def process(self, x):
        p = self.params

        def target(lv):
            over = lv - p["threshold_db"]
            return db_to_lin(-over * (1 - 1 / p["ratio"])) if over > 0 else 1.0

        return x * self.envf.gains(x, target, p["attack_ms"], p["release_ms"]) * db_to_lin(p["makeup_db"])


@register("PitchShift")
class PitchShift(Node):
    """Low-latency granular delay-line pitch shifter (~window/2 latency)."""
    defaults = {"semitones": 0.0, "window_ms": 40.0, "mix": 1.0}

    def setup(self):
        self.W = int(self.sr * self.params["window_ms"] / 1000)
        self.hist = np.zeros(self.W + 4)
        self.phase = 0.0

    def on_change(self, changed):
        if "window_ms" in changed:
            self.setup()

    @property
    def latency(self):
        return self.W // 2

    def process(self, x):
        r = 2 ** (self.params["semitones"] / 12)
        n, W, H = len(x), self.W, len(self.hist)
        ext = np.concatenate([self.hist, x])
        t = self.phase + np.arange(n) * (1 - r) / W
        out = np.zeros(n)
        for off in (0.0, 0.5):
            f = (t + off) % 1.0
            pos = H + np.arange(n) - 1 - f * W
            i0 = np.floor(pos).astype(int)
            fr = pos - i0
            out += (ext[i0] * (1 - fr) + ext[i0 + 1] * fr) * np.sin(np.pi * f) ** 2
        self.phase = (self.phase + n * (1 - r) / W) % 1.0
        self.hist = ext[-H:]
        m = self.params["mix"]
        return x * (1 - m) + out * m


# --------------------------------------------------------------------------- spectral FX

class _SpectralNode(Node):
    defaults = {"n_fft": 2048, "hop": 512}

    def setup(self):
        self.stft = StreamingSTFT(int(self.params["n_fft"]), int(self.params["hop"]))
        self.stft.process_frame = self.frame
        self.k = np.arange(self.stft.n_bins)

    @property
    def latency(self):
        return self.stft.latency

    def frame(self, spec):
        return spec

    def process(self, x):
        return self.stft.process(x)


@register("FormantPitchShifter")
class FormantPitchShifter(_SpectralNode):
    """Phase-vocoder pitch shift with independent formant shift and optional
    autotune."""
    defaults = {
        **_SpectralNode.defaults,
        "semitones": 0.0,
        "formant_semitones": 0.0,
        "preserve_formants": True,
        "autotune": 0.0,          # 0..1 strength of snapping to scale
        "scale": "chromatic",     # chromatic | major | minor
        "key": 0,                 # 0 = C
        "autotune_speed": 0.6,    # 0 = slow glide, 1 = instant (robotic)
    }
    SCALES = {"chromatic": list(range(12)), "major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}

    def setup(self):
        super().setup()
        N, hop = self.stft.n_fft, self.stft.hop
        self.expct = 2 * np.pi * hop / N
        self.last_phase = np.zeros(self.stft.n_bins)
        self.sum_phase = np.zeros(self.stft.n_bins)
        self.n_ceps = max(8, int(self.sr / 650 * N / 2048))
        self.tune_ratio = 1.0

    def _detect_f0(self, spec):
        ac = np.fft.irfft(np.abs(spec) ** 2)
        lo, hi = int(self.sr / 800), int(self.sr / 70)
        if ac[0] <= 1e-9:
            return None
        seg = ac[lo:hi]
        i = int(np.argmax(seg))
        if seg[i] / ac[0] < 0.35 or i == 0 or i == len(seg) - 1:
            return None
        a, b, c = seg[i - 1], seg[i], seg[i + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
        return self.sr / (lo + i + shift)

    def _autotune_ratio(self, spec):
        p = self.params
        f0 = self._detect_f0(spec)
        target = 1.0
        if f0:
            midi = 69 + 12 * math.log2(f0 / 440)
            degrees = self.SCALES.get(p["scale"], self.SCALES["chromatic"])
            cands = [12 * o + d + p["key"] for o in range(-1, 11) for d in degrees]
            nearest = min(cands, key=lambda m: abs(m - midi))
            target = 2 ** ((nearest - midi) / 12 * p["autotune"])
        sp = float(p["autotune_speed"])
        self.tune_ratio = self.tune_ratio * (1 - sp) + target * sp
        return self.tune_ratio

    def frame(self, spec):
        p = self.params
        ratio = 2 ** (p["semitones"] / 12)
        if p["autotune"] > 0:
            ratio *= self._autotune_ratio(spec)
        mag = np.abs(spec)
        phase = np.angle(spec)
        dev = wrap_phase(phase - self.last_phase - self.k * self.expct)
        self.last_phase = phase
        true_bin = self.k + dev / self.expct

        preserve = p["preserve_formants"] or p["formant_semitones"] != 0
        if preserve:
            env = cepstral_envelope(mag, self.n_ceps)
            fine = mag / np.maximum(env, 1e-9)
        else:
            fine = mag

        tgt = np.round(self.k * ratio).astype(int)
        ok = tgt < len(self.k)
        syn_mag = np.zeros_like(mag)
        np.add.at(syn_mag, tgt[ok], fine[ok])
        syn_bin = np.zeros_like(mag)
        syn_bin[tgt[ok]] = true_bin[ok] * ratio

        if preserve:
            f_ratio = 2 ** (p["formant_semitones"] / 12)
            if not p["preserve_formants"]:
                f_ratio *= ratio
            syn_mag *= np.interp(self.k / f_ratio, self.k, env, right=env[-1] * 1e-3)

        self.sum_phase = wrap_phase(self.sum_phase + syn_bin * self.expct)
        return syn_mag * np.exp(1j * self.sum_phase)


@register("Vocoder")
class Vocoder(_SpectralNode):
    """Channel vocoder: speech envelope imposed on a synthetic carrier — classic robot."""
    defaults = {**_SpectralNode.defaults, "n_fft": 1024, "hop": 256,
                "carrier_hz": 100.0, "waveform": "saw", "bands": 24, "mix": 1.0}

    def setup(self):
        super().setup()
        self.cphase = 0.0
        self.win = self.stft.window

    def frame(self, spec):
        p = self.params
        N, hop = self.stft.n_fft, self.stft.hop
        f = p["carrier_hz"] / self.sr
        ph = (self.cphase + np.arange(N) * f) % 1.0
        self.cphase = (self.cphase + hop * f) % 1.0
        car = 2 * ph - 1 if p["waveform"] == "saw" else np.sign(np.sin(2 * np.pi * ph))
        cspec = np.fft.rfft(car * self.win)
        width = max(1, len(self.k) // int(p["bands"]))
        ker = np.ones(width) / width
        env_in = np.convolve(np.abs(spec), ker, mode="same")
        env_c = np.convolve(np.abs(cspec), ker, mode="same") + 1e-9
        out = cspec * env_in / env_c
        return spec * (1 - p["mix"]) + out * p["mix"]


@register("Whisperize")
class Whisperize(_SpectralNode):
    """Randomised phase → breathy whisper."""
    defaults = {**_SpectralNode.defaults, "n_fft": 1024, "hop": 256}

    def setup(self):
        super().setup()
        self.rng = np.random.default_rng(7)

    def frame(self, spec):
        return np.abs(spec) * np.exp(2j * np.pi * self.rng.random(len(spec)))


@register("ConvolutionReverb")
class ConvolutionReverb(Node):
    """Convolution reverb with a synthetic room impulse response.

    Uniformly-partitioned overlap-save convolution: cost per block is one
    FFT pair plus a multiply-accumulate over IR partitions, independent of
    the IR length. The partition delay is absorbed into the pre-delay so the
    dry path stays at zero latency."""
    defaults = {"decay_s": 1.2, "predelay_ms": 20.0, "damping": 0.5, "mix": 0.3, "seed": 3, "partition": 512}

    def setup(self):
        p = self.params
        P = self.P = int(p["partition"])
        rng = np.random.default_rng(int(p["seed"]))
        n = int(self.sr * p["decay_s"])
        t = np.arange(n) / self.sr
        ir = rng.standard_normal(n) * np.exp(-6.9 * t / max(p["decay_s"], 1e-3))
        if p["damping"] > 0:  # progressive high-frequency damping
            ir = lfilter([1 - p["damping"] * 0.9], [1, -p["damping"] * 0.9], ir)
        pre = max(0, int(self.sr * p["predelay_ms"] / 1000) - P)
        ir = np.concatenate([np.zeros(pre), ir])
        ir /= np.sqrt(np.sum(ir**2))
        K = -(-len(ir) // P)
        parts = np.zeros((K, P))
        parts.flat[: len(ir)] = ir
        self.H = np.fft.rfft(parts, n=2 * P, axis=1)
        self.X = np.zeros_like(self.H)
        self.w = 0
        self.prev = np.zeros(P)
        self.fifo_in = np.zeros(0)
        self.fifo_out = np.zeros(P)

    def on_change(self, changed):
        if set(changed) - {"mix"}:
            self.setup()

    def _block(self, b):
        P, K = self.P, len(self.H)
        self.w = (self.w + 1) % K
        self.X[self.w] = np.fft.rfft(np.concatenate([self.prev, b]))
        self.prev = b
        order = (self.w - np.arange(K)) % K
        Y = np.einsum("kb,kb->b", self.X[order], self.H)
        return np.fft.irfft(Y, 2 * P)[P:]

    def process(self, x):
        self.fifo_in = np.concatenate([self.fifo_in, x])
        outs = []
        while len(self.fifo_in) >= self.P:
            outs.append(self._block(self.fifo_in[: self.P]))
            self.fifo_in = self.fifo_in[self.P:]
        self.fifo_out = np.concatenate([self.fifo_out, *outs])
        wet, self.fifo_out = self.fifo_out[: len(x)], self.fifo_out[len(x):]
        m = self.params["mix"]
        return x * (1 - m) + wet * m


@register("SoftLimiter")
class SoftLimiter(Node):
    defaults = {"ceiling_db": -1.0}

    def process(self, x):
        c = db_to_lin(self.params["ceiling_db"])
        return c * np.tanh(x / c)


@register("LatencyDelay")
class LatencyDelay(Node):
    """Pure integer delay used to align parallel branches."""
    defaults = {"samples": 0}

    def setup(self):
        self.buf = np.zeros(int(self.params["samples"]))

    @property
    def latency(self):
        return len(self.buf)

    def process(self, x):
        if not len(self.buf):
            return x
        ext = np.concatenate([self.buf, x])
        self.buf = ext[len(x):]
        return ext[: len(x)]
