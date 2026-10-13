"""Build the standalone app (and its installer when Inno Setup is available).

    python packaging/build.py            # -> build/dist/VoxShift/  (+ build/VoxShift-Setup-<ver>.exe)
    python packaging/build.py --no-installer

Requires the dev environment once, to export the AI model to ONNX
(`python -m voxshift export-onnx`); the resulting app needs no Python,
PyTorch or internet connection.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DIST = BUILD / "dist" / "VoxShift"
MODEL = ROOT / "models" / "onnx" / "whisper-small-bigvgan"
MODEL_FILES = ["config.json", "features.npz", "whisper_encoder.onnx", "length_regulator.onnx",
               "dit_step.onnx", "campplus.onnx", "vocoder.onnx"]
ISCC_CANDIDATES = [Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"), Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
                   Path.home() / r"AppData\Local\Programs\Inno Setup 6\ISCC.exe"]


def version() -> str:
    ns: dict = {}
    exec((ROOT / "voxshift" / "__init__.py").read_text(), ns)
    return ns["__version__"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-installer", action="store_true")
    args = ap.parse_args()

    missing = [f for f in MODEL_FILES if not (MODEL / f).is_file()]
    if missing:
        sys.exit(f"AI model not exported ({', '.join(missing)} missing): run `python -m voxshift export-onnx`")
    if not (ROOT / "voxshift" / "assets" / "voxshift.ico").is_file():
        subprocess.run([sys.executable, str(ROOT / "packaging" / "make_icon.py")], check=True)

    print("== PyInstaller")
    subprocess.run([sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "voxshift.spec"), "--noconfirm",
                    "--distpath", str(BUILD / "dist"), "--workpath", str(BUILD / "work")], check=True)

    print("== AI model")
    dest = DIST / "models" / "onnx" / MODEL.name
    dest.mkdir(parents=True, exist_ok=True)
    for f in MODEL_FILES:
        shutil.copy2(MODEL / f, dest / f)
    for doc in ("LICENSE", "THIRD_PARTY_NOTICES.md"):  # licence texts travel with the binaries
        shutil.copy2(ROOT / doc, DIST / doc)
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"app folder: {DIST} ({size / 2**30:.2f} GB)")

    if args.no_installer:
        return
    iscc = next((p for p in ISCC_CANDIDATES if p.is_file()), None) or shutil.which("ISCC")
    if not iscc:
        print("Inno Setup (ISCC.exe) not found - skipping the installer. The app folder above runs as-is.")
        return
    print("== Inno Setup")
    subprocess.run([str(iscc), f"/DAppVersion={version()}", f"/DDistDir={DIST}", f"/DOutDir={BUILD}",
                    str(ROOT / "packaging" / "voxshift.iss")], check=True)


if __name__ == "__main__":
    main()
