"""Real-time audio engine (PortAudio via sounddevice).

Topology::

    mic ──► input gain ──► voice processor ──► + soundboard ──► limiter ──► virtual cable / output
                                                         └──────────────► monitor ("hear myself")
"""
from __future__ import annotations

import collections
import sys
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from ..dsp.nodes import SoftLimiter, db_to_lin

PREFERRED_HOSTAPIS = ("Windows WASAPI", "Core Audio", "ALSA", "MME")


def hostapi_name(index: int) -> str:
    return sd.query_hostapis(index)["name"]


def list_devices(kind: str | None = None, hostapi: str | None = None) -> list[dict]:
    out = []
    for d in sd.query_devices():
        d = dict(d)
        d["hostapi_name"] = hostapi_name(d["hostapi"])
        if hostapi and d["hostapi_name"] != hostapi:
            continue
        if kind == "input" and d["max_input_channels"] < 1:
            continue
        if kind == "output" and d["max_output_channels"] < 1:
            continue
        out.append(d)
    return out


def default_hostapi() -> str:
    names = [h["name"] for h in sd.query_hostapis()]
    for n in PREFERRED_HOSTAPIS:
        if n in names:
            return n
    return names[0]


class NoVirtualCable(RuntimeError):
    pass


_CLI = "voxshift-cli" if getattr(sys, "frozen", False) else "python -m voxshift"
NO_CABLE_HELP = (
    "No virtual microphone found, so other apps cannot hear VoxShift.\n"
    f"  * install VB-CABLE:  {_CLI} cable --install   (or 'Set up virtual microphone' in the app)\n"
    "  * or pass --output <device> to send the voice somewhere else")


def resolve_device(spec: str | int | None, kind: str, hostapi: str | None = None) -> int | None:
    """Accept an index, an exact name, a case-insensitive substring, or
    ``"virtual"`` for the first detected virtual-cable playback endpoint."""
    if kind == "output" and isinstance(spec, str) and spec.lower() == "virtual":
        from .cable import find_cables
        cables = find_cables(hostapi)
        if not cables:
            raise NoVirtualCable(NO_CABLE_HELP)
        spec = cables[0].playback
    if spec is None or spec == "":
        if hostapi:
            h = next(h for h in sd.query_hostapis() if h["name"] == hostapi)
            idx = h[f"default_{kind}_device"]
            return idx if idx >= 0 else None
        return None
    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    cands = list_devices(kind, hostapi)
    for d in cands:
        if d["name"] == spec:
            return d["index"]
    for d in cands:
        if str(spec).lower() in d["name"].lower():
            return d["index"]
    raise ValueError(f"no {kind} device matching {spec!r}")


@dataclass
class Meters:
    input_db: float = -120.0
    output_db: float = -120.0
    cpu_ms: float = 0.0
    xruns: int = 0


def _db(x: np.ndarray) -> float:
    return 20 * np.log10(float(np.sqrt(np.mean(x**2))) + 1e-9) if len(x) else -120.0


class _MonitorFifo:
    def __init__(self, max_samples: int):
        self.buf: collections.deque = collections.deque()
        self.size = 0
        self.max = max_samples
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        with self.lock:
            self.buf.append(x)
            self.size += len(x)
            while self.size > self.max and self.buf:  # drop oldest to bound latency
                self.size -= len(self.buf.popleft())

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        pos = 0
        with self.lock:
            while pos < n and self.buf:
                chunk = self.buf[0]
                take = min(n - pos, len(chunk))
                out[pos:pos + take] = chunk[:take]
                pos += take
                if take == len(chunk):
                    self.buf.popleft()
                else:
                    self.buf[0] = chunk[take:]
                self.size -= take
        return out


class AudioEngine:
    def __init__(self, input_device=None, output_device=None, monitor_device=None,
                 samplerate: int = 48000, blocksize: int = 512, hostapi: str | None = None):
        self.hostapi = hostapi or default_hostapi()
        self.input_device = resolve_device(input_device, "input", self.hostapi)
        self.output_device = resolve_device(output_device, "output", self.hostapi)
        self.monitor_device = resolve_device(monitor_device, "output", self.hostapi) if monitor_device else None
        self.sr = samplerate
        self.default_blocksize = blocksize
        self.processor = None
        self.voice_enabled = True
        self.mic_muted = False
        self.hear_myself = self.monitor_device is not None
        self.input_gain_db = 0.0
        self.output_gain_db = 0.0
        self.monitor_gain_db = -6.0
        self.meters = Meters()
        self._limiter = SoftLimiter(samplerate, ceiling_db=-0.5)
        self._sounds: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self._monitor_stream = None
        self._monitor = _MonitorFifo(samplerate // 5)
        self._blocksize = blocksize

    # ------------------------------------------------------------------ control
    def set_processor(self, processor) -> None:
        old = self.processor
        with self._lock:
            self.processor = processor
        if old is not None and hasattr(old, "close"):
            old.close()
        want = getattr(processor, "preferred_blocksize", None) or self.default_blocksize
        if self._stream is not None and want != self._blocksize:
            self.stop()
            self.start()

    def play_sound(self, data: np.ndarray, sr: int, gain_db: float = 0.0) -> None:
        """Soundboard: mix a clip into the output (and monitor)."""
        from scipy.signal import resample_poly
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != self.sr:
            g = np.gcd(sr, self.sr)
            data = resample_poly(data, self.sr // g, sr // g)
        with self._lock:
            self._sounds.append(data.astype(np.float64) * db_to_lin(gain_db))

    def stop_sounds(self) -> None:
        with self._lock:
            self._sounds.clear()

    # ------------------------------------------------------------------ streams
    def _extra(self):
        if self.hostapi == "Windows WASAPI":
            return sd.WasapiSettings(auto_convert=True)
        return None

    def start(self) -> None:
        self._blocksize = getattr(self.processor, "preferred_blocksize", None) or self.default_blocksize
        out_ch = min(2, sd.query_devices(self.output_device or sd.default.device[1])["max_output_channels"])
        self._out_ch = out_ch
        extra = self._extra()
        self._stream = sd.Stream(device=(self.input_device, self.output_device), samplerate=self.sr,
                                 blocksize=self._blocksize, channels=(1, out_ch), dtype="float32",
                                 callback=self._callback, extra_settings=(extra, extra), latency="low")
        if self.monitor_device is not None:
            mch = min(2, sd.query_devices(self.monitor_device)["max_output_channels"])
            self._monitor_stream = sd.OutputStream(device=self.monitor_device, samplerate=self.sr,
                                                   blocksize=0, channels=mch, dtype="float32",
                                                   callback=self._monitor_cb, extra_settings=extra,
                                                   latency="low")
            self._monitor_stream.start()
        self._stream.start()

    def stop(self) -> None:
        for s in (self._stream, self._monitor_stream):
            if s is not None:
                s.stop()
                s.close()
        self._stream = self._monitor_stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def stream_latency_ms(self) -> float:
        if self._stream is None:
            return 0.0
        lat = self._stream.latency
        return 1000 * (sum(lat) if isinstance(lat, tuple) else lat)

    def _monitor_cb(self, outdata, frames, t, status):
        y = self._monitor.pull(frames) * db_to_lin(self.monitor_gain_db) if self.hear_myself else np.zeros(frames)
        outdata[:] = np.repeat(y[:, None], outdata.shape[1], axis=1)

    def _callback(self, indata, outdata, frames, t, status):
        import time
        t0 = time.perf_counter()
        if status:
            self.meters.xruns += 1
        x = indata[:, 0].astype(np.float64) * (0.0 if self.mic_muted else db_to_lin(self.input_gain_db))
        with self._lock:
            proc = self.processor
            y = proc.process(x) if (proc is not None and self.voice_enabled) else x
            if self._sounds:
                keep = []
                for s in self._sounds:
                    n = min(frames, len(s))
                    y[:n] += s[:n]
                    if len(s) > frames:
                        keep.append(s[frames:])
                self._sounds = keep
        y = self._limiter.process(y * db_to_lin(self.output_gain_db)).astype(np.float32)
        outdata[:] = np.repeat(y[:, None], outdata.shape[1], axis=1)
        if self.hear_myself and self._monitor_stream is not None:
            self._monitor.push(y)
        self.meters.input_db = _db(x)
        self.meters.output_db = _db(y)
        self.meters.cpu_ms = (time.perf_counter() - t0) * 1000
