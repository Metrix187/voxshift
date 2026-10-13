"""AI voice pack: split / download / reassemble the ONNX model.

The 820 MB model is published once as a GitHub release (``models-v1``) cut
into 25 MB chunks plus a ``manifest.json`` with SHA-256 for every chunk and
every reassembled file. Small chunks make both publishing and downloading
robust: a failed chunk is retried on its own, an interrupted download
resumes, and nothing is trusted until its hash matches.

    pack()      dev: model dir -> chunks + manifest (for upload)
    download()  app/installer: manifest + chunks -> verified model dir
    assemble()  chunks already on disk (e.g. fetched by the installer) -> model dir
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

from ..paths import install_dir, user_models_dir

REPO = "Metrix187/voxshift"
PACK_TAG = "models-v1"
MODEL_NAME = "whisper-small-bigvgan"
BASE_URL = os.environ.get("VOXSHIFT_PACK_URL", f"https://github.com/{REPO}/releases/download/{PACK_TAG}/")
CHUNK = 25 * 2**20
_BUF = 2**20

Progress = Callable[[int, int], None]  # (bytes_done, bytes_total)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_BUF), b""):
            h.update(block)
    return h.hexdigest()


def find_model(name: str = MODEL_NAME) -> Path | None:
    """Installed model folder: bundled next to the app / repo, else downloaded."""
    for root in (install_dir() / "models", user_models_dir()):
        d = root / "onnx" / name
        if (d / "config.json").is_file():
            return d
    return None


# ------------------------------------------------------------------ publishing
def pack(model_dir: Path, out_dir: Path, chunk_size: int = CHUNK, base_url: str = BASE_URL) -> dict:
    """Concatenate the model files and cut the stream into fixed-size chunks."""
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in model_dir.iterdir() if p.is_file())
    manifest = {"name": model_dir.name, "pack": PACK_TAG, "files": [], "chunks": []}
    chunk_idx, chunk_fill, out = 0, 0, None
    chunk_hash = None

    def close_chunk():
        nonlocal out
        if out:
            out.close()
            manifest["chunks"][-1]["sha256"] = chunk_hash.hexdigest()
            out = None

    for p in files:
        manifest["files"].append({"path": p.name, "size": p.stat().st_size, "sha256": _sha256(p)})
        with open(p, "rb") as f:
            while block := f.read(_BUF):
                while block:
                    if out is None:
                        chunk_idx += 1
                        name = f"{model_dir.name}.part{chunk_idx:03d}"
                        manifest["chunks"].append({"name": name, "url": base_url + name})
                        out, chunk_fill, chunk_hash = open(out_dir / name, "wb"), 0, hashlib.sha256()
                    take = min(len(block), chunk_size - chunk_fill)
                    out.write(block[:take])
                    chunk_hash.update(block[:take])
                    chunk_fill += take
                    block = block[take:]
                    if chunk_fill == chunk_size:
                        manifest["chunks"][-1]["size"] = chunk_fill
                        close_chunk()
    if out is not None:
        manifest["chunks"][-1]["size"] = chunk_fill
        close_chunk()
    manifest["total_size"] = sum(c["size"] for c in manifest["chunks"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ------------------------------------------------------------------ installing
def assemble(manifest: dict, chunk_dir: Path, dest_root: Path | None = None) -> Path:
    """Verify chunks, rebuild the files, verify them, then move into place atomically."""
    chunk_dir = Path(chunk_dir)
    for c in manifest["chunks"]:
        p = chunk_dir / c["name"]
        if not p.is_file() or p.stat().st_size != c["size"] or _sha256(p) != c["sha256"]:
            raise ValueError(f"chunk {c['name']} is missing or corrupt")
    dest = (Path(dest_root) if dest_root else user_models_dir()) / "onnx" / manifest["name"]
    staging = dest.with_name(dest.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    chunks = iter(manifest["chunks"])
    src, left_in_chunk = None, 0
    for f in manifest["files"]:
        h, remaining = hashlib.sha256(), f["size"]
        with open(staging / f["path"], "wb") as out:
            while remaining:
                if left_in_chunk == 0:
                    if src:
                        src.close()
                    c = next(chunks)
                    src, left_in_chunk = open(chunk_dir / c["name"], "rb"), c["size"]
                block = src.read(min(_BUF, remaining, left_in_chunk))
                out.write(block)
                h.update(block)
                remaining -= len(block)
                left_in_chunk -= len(block)
        if h.hexdigest() != f["sha256"]:
            raise ValueError(f"{f['path']} failed verification after reassembly")
    if src:
        src.close()
    if dest.exists():
        shutil.rmtree(dest)
    staging.rename(dest)
    return dest


def _fetch(url: str, path: Path, size: int, sha: str, cancel: threading.Event | None,
           on_bytes: Callable[[int], None], retries: int = 5) -> None:
    """Download one chunk with resume (HTTP Range) and retries; verify its hash."""
    for attempt in range(1, retries + 1):
        have = path.stat().st_size if path.exists() else 0
        if have > size:
            path.unlink()
            have = 0
        try:
            if have < size:
                req = urllib.request.Request(url, headers={"User-Agent": "VoxShift", "Range": f"bytes={have}-"})
                with urllib.request.urlopen(req, timeout=30) as r, open(path, "ab" if r.status == 206 else "wb") as f:
                    if r.status != 206:
                        on_bytes(-have)  # server ignored Range: restart this chunk
                    while block := r.read(_BUF):
                        if cancel and cancel.is_set():
                            raise InterruptedError("download cancelled")
                        f.write(block)
                        on_bytes(len(block))
            if _sha256(path) == sha:
                return
            on_bytes(-path.stat().st_size)
            path.unlink()
        except InterruptedError:
            raise
        except OSError:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 15))
    raise ValueError(f"{path.name}: checksum mismatch after {retries} attempts")


def download(progress: Progress | None = None, cancel: threading.Event | None = None,
             base_url: str = BASE_URL, dest_root: Path | None = None) -> Path:
    """Fetch the AI voice pack (resumable) and install it. Returns the model dir."""
    req = urllib.request.Request(base_url + "manifest.json", headers={"User-Agent": "VoxShift"})
    with urllib.request.urlopen(req, timeout=30) as r:
        manifest = json.loads(r.read())
    cache = (Path(dest_root) if dest_root else user_models_dir()) / ".download"
    cache.mkdir(parents=True, exist_ok=True)
    total = manifest["total_size"]
    done = [sum((cache / c["name"]).stat().st_size for c in manifest["chunks"] if (cache / c["name"]).exists())]

    def on_bytes(n):
        done[0] = max(0, done[0] + n)
        if progress:
            progress(min(done[0], total), total)

    for c in manifest["chunks"]:  # chunks live next to the manifest (works for mirrors too)
        _fetch(base_url + c["name"], cache / c["name"], c["size"], c["sha256"], cancel, on_bytes)
    model = assemble(manifest, cache, dest_root)
    shutil.rmtree(cache, ignore_errors=True)
    return model
