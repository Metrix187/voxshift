"""Command-line interface: ``python -m voxshift <command>``."""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _ai_overrides(a) -> dict:
    return {"diffusion_steps": a.diffusion_steps, "block_time": a.block_time,
            "inference_cfg_rate": a.cfg_rate}


def cmd_devices(a):
    from .audio.engine import default_hostapi, list_devices
    api = a.hostapi or default_hostapi()
    print(f"host API: {api}")
    for kind in ("input", "output"):
        print(f"\n{kind} devices:")
        for d in list_devices(kind, api):
            print(f"  [{d['index']:>3}] {d['name']}  ({int(d['default_samplerate'])} Hz)")


def cmd_voices(a):
    from .presets import list_presets
    for key, spec in list_presets().items():
        tag = "AI " if spec.get("engine") else "DSP"
        print(f"  {key:<18} {tag}  {spec.get('name', key)} — {spec.get('description', '')}")


def cmd_convert(a):
    import soundfile as sf
    from scipy.signal import resample_poly

    from .presets import build_processor, load_spec

    data, sr_in = sf.read(a.input, always_2d=True)
    x = data.mean(axis=1)
    sr = a.sr
    if sr_in != sr:
        g = np.gcd(sr_in, sr)
        x = resample_poly(x, sr // g, sr_in // g)
    spec = load_spec(a.voice)
    t0 = time.perf_counter()
    proc = build_processor(spec, sr, reference=a.reference, device=a.device, **_ai_overrides(a))
    load_s = time.perf_counter() - t0
    block = getattr(proc, "preferred_blocksize", None) or a.blocksize
    lat = proc.latency
    padded = np.concatenate([x, np.zeros(lat + block)])
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(padded), block):
        chunk = padded[i:i + block]
        if len(chunk) < block:
            chunk = np.concatenate([chunk, np.zeros(block - len(chunk))])
        out.append(proc.process(chunk))
    proc_s = time.perf_counter() - t0
    y = np.concatenate(out)[lat: lat + len(x)]
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak > 0.99:
        y = y * (0.99 / peak)
    sf.write(a.output, y.astype(np.float32), sr)
    dur = len(x) / sr
    print(f"{spec.get('name')}: {dur:.2f}s audio processed in {proc_s:.2f}s "
          f"(RTF {proc_s / max(dur, 1e-9):.3f}, latency {1000 * lat / sr:.1f} ms"
          + (f", model load {load_s:.1f}s" if load_s > 0.5 else "") + f") -> {a.output}")


def cmd_cable(a):
    from .audio.cable import download_vbcable, find_cables, install_vbcable, self_test
    from .audio.engine import default_hostapi
    if a.download:  # used by the installer: it runs the verified setup elevated itself
        print(download_vbcable(a.download))
        return
    if a.install:
        install_vbcable(silent=a.silent)
        print("\nVB-CABLE installed. Reboot if the installer asked you to, then run `python -m voxshift cable`.")
        return
    api = a.hostapi or default_hostapi()
    cables = find_cables(api)
    if not cables:
        from .audio.engine import NO_CABLE_HELP
        raise SystemExit(NO_CABLE_HELP)
    for c in cables:
        ok, ratio = self_test(c, api) if not a.no_test else (None, 0.0)
        verdict = "untested" if ok is None else ("working" if ok else f"NOT passing audio ({ratio:.0f} dB)")
        print(f"{c.label}: VoxShift -> '{c.playback}' -> apps pick mic '{c.capture}'  [{verdict}]")
        if c.caveat:
            print(f"    note: {c.caveat}")


def cmd_live(a):
    import sounddevice as sd

    from .audio.cable import find_cables
    from .audio.engine import AudioEngine, NoVirtualCable
    from .presets import build_processor, load_spec

    try:
        eng = AudioEngine(a.input, a.output, a.monitor, samplerate=a.sr, blocksize=a.blocksize, hostapi=a.hostapi)
    except NoVirtualCable as e:
        raise SystemExit(str(e))
    spec = load_spec(a.voice)
    proc = build_processor(spec, a.sr, reference=a.reference, ai_mode=a.ai_mode, device=a.device,
                           **_ai_overrides(a))
    if hasattr(proc, "warmup"):
        print("warming up model…")
        proc.warmup()
    eng.set_processor(proc)
    eng.start()
    print(f"live: {spec.get('name')} | in={eng.input_device} out={eng.output_device} "
          f"monitor={eng.monitor_device} | Ctrl+C to stop")
    out_name = sd.query_devices(eng.output_device)["name"]
    cable = next((c for c in find_cables(eng.hostapi) if c.playback == out_name), None)
    if cable:
        print(f"-> in Discord/OBS/games choose the microphone '{cable.capture}'")
        if cable.caveat:
            print(f"   note: {cable.caveat}")
    try:
        while True:
            m = eng.meters
            extra = f" ai={proc.streamer.last_infer_ms:5.0f}ms" if hasattr(proc, "streamer") else ""
            sys.stdout.write(f"\r in {m.input_db:6.1f} dB  out {m.output_db:6.1f} dB  "
                             f"dsp {m.cpu_ms:5.1f} ms{extra}  xruns {m.xruns}   ")
            sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()
        print()


def cmd_gui(a):
    from .gui import main
    main(["--minimized"] if a.minimized else [])


def cmd_export_onnx(a):
    from .ai.onnx_export import export
    export(a.model, a.out_dir, opset=a.opset, fp16=not a.fp32)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voxshift", description="Local real-time voice changer.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("devices", help="list audio devices")
    s.add_argument("--hostapi")
    s.set_defaults(fn=cmd_devices)

    sub.add_parser("voices", help="list voice presets").set_defaults(fn=cmd_voices)

    def common(s):
        s.add_argument("--voice", "-v", default="robot")
        s.add_argument("--reference", "-r", help="target-speaker WAV for AI voices")
        s.add_argument("--sr", type=int, default=48000)
        s.add_argument("--blocksize", type=int, default=512)
        s.add_argument("--device", help="torch device for AI voices (cuda / cpu)")
        s.add_argument("--diffusion-steps", type=int)
        s.add_argument("--block-time", type=float)
        s.add_argument("--cfg-rate", type=float)

    s = sub.add_parser("convert", help="render a file through a voice")
    s.add_argument("input")
    s.add_argument("output")
    common(s)
    s.set_defaults(fn=cmd_convert)

    s = sub.add_parser("live", help="run the real-time voice changer")
    common(s)
    s.add_argument("--input", "-i", help="microphone (index or name substring)")
    s.add_argument("--output", "-o", default="virtual",
                   help="'virtual' (default: first detected virtual cable) or a device name/index")
    s.add_argument("--monitor", "-m", help="headphones for 'hear myself'")
    s.add_argument("--hostapi")
    s.add_argument("--ai-mode", choices=["inline", "threaded"], default="inline")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("cable", help="find and self-test virtual microphones, or install VB-CABLE")
    s.add_argument("--install", action="store_true", help="download, verify and launch the VB-CABLE installer")
    s.add_argument("--silent", action="store_true", help="with --install: hidden setup UI (-i -h)")
    s.add_argument("--download", metavar="DIR", help="only download + verify the VB-CABLE pack into DIR")
    s.add_argument("--no-test", action="store_true", help="skip the loopback tone test")
    s.add_argument("--hostapi")
    s.set_defaults(fn=cmd_cable)

    s = sub.add_parser("gui", help="open the desktop app")
    s.add_argument("--minimized", action="store_true", help="start in the system tray")
    s.set_defaults(fn=cmd_gui)

    s = sub.add_parser("export-onnx", help="export the AI networks to ONNX")
    s.add_argument("--model", default="whisper-small-bigvgan")
    s.add_argument("--out-dir", default="models/onnx")
    s.add_argument("--opset", type=int, default=17)
    s.add_argument("--fp32", action="store_true", help="keep every network in fp32")
    s.set_defaults(fn=cmd_export_onnx)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(sys, "frozen", False):  # the installer runs us hidden: keep a record
        import logging

        from .paths import log_dir
        logging.basicConfig(filename=log_dir() / "voxshift-cli.log", level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        a.fn(a)
    except Exception as e:
        import logging
        logging.getLogger("voxshift.cli").exception("command %s failed", a.cmd)
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
