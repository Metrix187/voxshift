"""Virtual microphone ("cable") discovery, self-test and VB-CABLE install.

Other apps can only pick VoxShift up as a microphone through a virtual audio
driver: audio rendered to its *playback* endpoint reappears on its *capture*
endpoint. VoxShift works with the common ones and prefers them in this order.
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

# (label, playback endpoint VoxShift writes to, capture endpoint apps select, caveat)
KNOWN_CABLES = [
    ("VB-CABLE", "CABLE Input", "CABLE Output", ""),
    ("VB-CABLE A", "CABLE-A Input", "CABLE-A Output", ""),
    ("VB-CABLE B", "CABLE-B Input", "CABLE-B Output", ""),
    ("VoiceMeeter", "VoiceMeeter Input", "VoiceMeeter Output", ""),
    ("VoiceMeeter AUX", "VoiceMeeter AUX Input", "VoiceMeeter AUX Output", ""),
    ("Virtual Audio Cable", "Line 1 (Virtual Audio Cable)", "Line 1 (Virtual Audio Cable)", ""),
]

VBCABLE_URL = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip"
VBCABLE_SIGNERS = ("BUREL VINCENT", "Vincent Burel", "VB-Audio")  # VB-Audio's code-signing subject


@dataclass
class Cable:
    label: str
    playback: str      # full device name VoxShift outputs to
    capture: str       # full device name to choose as the mic in Discord/OBS/games
    caveat: str = ""


def find_cables(hostapi: str | None = None, devices=None) -> list[Cable]:
    """Return every known virtual cable whose playback AND capture ends are present."""
    if devices is None:
        from .engine import default_hostapi, list_devices
        devices = list_devices(None, hostapi or default_hostapi())
    outs = [d["name"] for d in devices if d["max_output_channels"] > 0]
    ins = [d["name"] for d in devices if d["max_input_channels"] > 0]
    found = []
    for label, play, cap, caveat in KNOWN_CABLES:
        p = next((n for n in outs if n.lower().startswith(play.lower())), None)
        c = next((n for n in ins if n.lower().startswith(cap.lower())), None)
        if p and c:
            found.append(Cable(label, p, c, caveat))
    return found


def self_test(cable: Cable, hostapi: str | None = None, seconds: float = 1.0) -> tuple[bool, float]:
    """Play a quiet tone into the cable and listen on its capture side.

    Returns (ok, tone-to-noise ratio in dB). Uses explicit streams because
    sounddevice's play()/rec() helpers share one global stream.
    """
    from .engine import default_hostapi, resolve_device
    api = hostapi or default_hostapi()
    o, i = resolve_device(cable.playback, "output", api), resolve_device(cable.capture, "input", api)
    sr, f = 48000, 997.0
    tone = (0.05 * np.sin(2 * np.pi * f * np.arange(int(sr * seconds)) / sr)).astype(np.float32)
    ext = sd.WasapiSettings(auto_convert=True) if api == "Windows WASAPI" else None
    chunks, pos = [], [0]

    def play_cb(out, frames, t, status):
        seg = tone[pos[0]:pos[0] + frames]
        out[:] = 0
        out[: len(seg)] = seg[:, None]
        pos[0] += frames

    with sd.InputStream(device=i, samplerate=sr, channels=1, dtype="float32", extra_settings=ext,
                        callback=lambda d, f_, t, s: chunks.append(d[:, 0].copy())):
        time.sleep(0.15)
        with sd.OutputStream(device=o, samplerate=sr, channels=2, dtype="float32", extra_settings=ext,
                             callback=play_cb):
            time.sleep(seconds + 0.2)
    x = np.concatenate(chunks) if chunks else np.zeros(1)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = int(round(f * len(x) / sr))
    ratio_db = float(20 * np.log10(spec[max(k - 2, 0): k + 3].max() / (np.median(spec) + 1e-12) + 1e-12))
    return ratio_db > 40, ratio_db


def _env(exe: Path) -> dict:
    import os
    return {**os.environ, "VOXSHIFT_EXE": str(exe)}  # avoids quoting paths into PowerShell


def _verify_signature(exe: Path) -> str:
    """Require a valid Authenticode signature from VB-Audio before running anything."""
    ps = ("$s = Get-AuthenticodeSignature -LiteralPath $env:VOXSHIFT_EXE; "
          "'{0}|{1}' -f $s.Status, $s.SignerCertificate.Subject")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=_env(exe), capture_output=True,
                         text=True, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    status, _, subject = out.partition("|")
    if status != "Valid" or not any(s.lower() in subject.lower() for s in VBCABLE_SIGNERS):
        raise RuntimeError(f"refusing to run {exe.name}: signature {status!r}, signer {subject!r}")
    return subject


def download_vbcable(dest: Path | None = None) -> Path:
    """Download and unpack the official VB-CABLE pack and verify its setup's
    signature. Returns the verified setup executable (which needs the rest of
    the pack beside it)."""
    if platform.system() != "Windows":
        raise RuntimeError("VB-CABLE's installer is Windows-only (macOS: use BlackHole)")
    work = Path(tempfile.mkdtemp(prefix="vbcable-"))
    archive = work / "VBCABLE_Driver_Pack45.zip"
    print(f"downloading {VBCABLE_URL}")
    urllib.request.urlretrieve(VBCABLE_URL, archive)
    dest = Path(dest) if dest else work / "pack"
    if dest.exists():
        shutil.rmtree(dest)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():  # reject path traversal
            if Path(member).is_absolute() or ".." in Path(member).parts:
                raise RuntimeError(f"unsafe path in archive: {member}")
        z.extractall(dest)
    # the x64 setup also installs the pack's ARM64 driver on ARM64 Windows
    exe = dest / ("VBCABLE_Setup_x64.exe" if platform.machine().endswith("64") else "VBCABLE_Setup.exe")
    if not exe.is_file():
        raise RuntimeError(f"setup not found in pack: {sorted(p.name for p in dest.iterdir())}")
    signer = _verify_signature(exe)
    print(f"verified signature: {signer}")
    log.info("VB-CABLE pack %s verified, signer %s", exe, signer)
    return exe


def install_vbcable(silent: bool = False, hwnd: int | None = None) -> Path:
    """Download + verify VB-CABLE, then run its setup as administrator.

    Windows shows a UAC prompt (in front of ``hwnd`` when given). ``silent``
    passes the setup's documented ``-i -h`` switches (install, hidden UI);
    Windows still shows its own driver-consent dialog, which cannot be
    skipped. Otherwise click "Install Driver" in the VB-CABLE window. A
    reboot may be needed before the devices appear."""
    from ..winsys import run_elevated
    exe = download_vbcable()
    print("launching installer (approve the Windows prompts" + ("" if silent else ", click 'Install Driver'") + ")…")
    code = run_elevated(str(exe), "-i -h" if silent else "", hwnd)
    log.info("VB-CABLE setup exit code %s", code)
    if code != 0:
        raise RuntimeError(f"VB-CABLE setup returned exit code {code}")
    return exe
