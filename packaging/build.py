"""Build the standalone app (and its installer when Inno Setup is available).

    python packaging/build.py            # -> build/dist/VoxShift/  (+ build/VoxShift-Setup-<ver>.exe)
    python packaging/build.py --no-installer
    python packaging/build.py --bundle-models   # offline installer (~880 MB) instead of the web installer

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
    ap.add_argument("--bundle-models", action="store_true", help="offline installer with the AI model inside")
    args = ap.parse_args()

    missing = [f for f in MODEL_FILES if not (MODEL / f).is_file()]
    if missing:
        sys.exit(f"AI model not exported ({', '.join(missing)} missing): run `python -m voxshift export-onnx`")
    if not (ROOT / "voxshift" / "assets" / "voxshift.ico").is_file():
        subprocess.run([sys.executable, str(ROOT / "packaging" / "make_icon.py")], check=True)

    print("== PyInstaller")
    subprocess.run([sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "voxshift.spec"), "--noconfirm",
                    "--distpath", str(BUILD / "dist"), "--workpath", str(BUILD / "work")], check=True)

    shutil.rmtree(DIST / "models", ignore_errors=True)
    iscc_defines = []
    if args.bundle_models:
        print("== AI model (bundled: offline installer)")
        dest = DIST / "models" / "onnx" / MODEL.name
        dest.mkdir(parents=True, exist_ok=True)
        for f in MODEL_FILES:
            shutil.copy2(MODEL / f, dest / f)
    else:
        print("== AI voice pack (web installer: downloaded during setup or from the app)")
        iscc_defines.append(f"/DModelInclude={write_model_pack()}")
    for doc in ("LICENSE", "THIRD_PARTY_NOTICES.md"):  # licence texts travel with the binaries
        shutil.copy2(ROOT / doc, DIST / doc)
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"app folder: {DIST} ({size / 2**20:.0f} MB)")

    if args.no_installer:
        return
    iscc = next((p for p in ISCC_CANDIDATES if p.is_file()), None) or shutil.which("ISCC")
    if not iscc:
        print("Inno Setup (ISCC.exe) not found - skipping the installer. The app folder above runs as-is.")
        return
    print("== Inno Setup")
    subprocess.run([str(iscc), f"/DAppVersion={version()}", f"/DDistDir={DIST}", f"/DOutDir={BUILD}",
                    *iscc_defines, str(ROOT / "packaging" / "voxshift.iss")], check=True)


def write_model_pack() -> Path:
    """Chunk the model into build/model-pack (what gets published as the models release) and
    emit the Pascal include that makes the installer download exactly those chunks."""
    sys.path.insert(0, str(ROOT))
    from voxshift.ai import model_pack

    pack_dir = BUILD / "model-pack"
    shutil.rmtree(pack_dir, ignore_errors=True)
    manifest = model_pack.pack(MODEL, pack_dir)
    lines = ["procedure AddModelDownloads;", "begin",
             f"  AddChunk('{model_pack.BASE_URL}manifest.json', 'manifest.json', "
             f"'{model_pack._sha256(pack_dir / 'manifest.json')}');"]
    lines += [f"  AddChunk('{c['url']}', '{c['name']}', '{c['sha256']}');" for c in manifest["chunks"]]
    lines += ["end;", ""]
    inc = BUILD / "model_downloads.iss"
    inc.write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(manifest['chunks'])} chunks ({manifest['total_size'] / 2**20:.0f} MB) in {pack_dir}; "
          f"publish them as release '{model_pack.PACK_TAG}' (python packaging/publish.py --models)")
    return inc


if __name__ == "__main__":
    main()
