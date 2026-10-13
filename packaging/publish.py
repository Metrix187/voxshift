"""Publish build outputs to GitHub releases, one file at a time with retries.

    python packaging/publish.py --models          # build/model-pack -> release models-v1 (not "latest")
    python packaging/publish.py --app             # build/VoxShift-Setup-<ver>.exe -> release v<ver> (latest)

Files already on the release with the same SHA-256 are skipped, so re-running
after a network failure only uploads what is missing. Needs the GitHub CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
sys.path.insert(0, str(ROOT))
from voxshift.ai.model_pack import PACK_TAG, REPO  # noqa: E402


def gh(*args, check=True) -> str:
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"gh {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def ensure_release(tag: str, title: str, notes: str, latest: bool) -> int:
    if subprocess.run(["gh", "release", "view", tag, "--repo", REPO], capture_output=True).returncode:
        gh("release", "create", tag, "--repo", REPO, "--title", title, "--notes", notes,
           f"--latest={'true' if latest else 'false'}")
    return json.loads(gh("api", f"repos/{REPO}/releases/tags/{tag}"))["id"]


def assets(release_id: int) -> dict[str, dict]:
    return {a["name"]: a for a in json.loads(gh("api", f"repos/{REPO}/releases/{release_id}/assets?per_page=100"))}


def same_file(asset: dict, f: Path) -> bool:
    """GitHub reports a SHA-256 digest per asset; fall back to size for older assets."""
    if asset.get("digest", "").startswith("sha256:"):
        return asset["digest"] == "sha256:" + hashlib.sha256(f.read_bytes()).hexdigest()
    return asset["size"] == f.stat().st_size


def upload(tag: str, release_id: int, files: list[Path], tries: int = 6) -> None:
    for i, f in enumerate(files, 1):
        for attempt in range(1, tries + 1):
            have = assets(release_id).get(f.name)
            if have and have["state"] == "uploaded" and same_file(have, f):
                print(f"[{i}/{len(files)}] {f.name}: ok")
                break
            if have:  # partial / stale upload: remove before retrying
                gh("api", "-X", "DELETE", f"repos/{REPO}/releases/assets/{have['id']}", check=False)
            print(f"[{i}/{len(files)}] {f.name}: uploading ({f.stat().st_size / 2**20:.1f} MB, try {attempt})")
            subprocess.run(["gh", "release", "upload", tag, str(f), "--repo", REPO, "--clobber"],
                           capture_output=True)
            time.sleep(1)
        else:
            raise SystemExit(f"{f.name}: still failing after {tries} attempts; re-run to resume")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--app", action="store_true")
    ap.add_argument("--notes", type=Path, help="release notes file for --app")
    a = ap.parse_args()
    if a.models:
        pack = BUILD / "model-pack"
        files = [pack / "manifest.json", *sorted(pack.glob("*.part*"))]
        rid = ensure_release(PACK_TAG, "AI voice pack v1",
                             "Model chunks downloaded by the VoxShift installer and app. "
                             "You don't need to download these by hand.", latest=False)
        upload(PACK_TAG, rid, files)
    if a.app:
        ns: dict = {}
        exec((ROOT / "voxshift" / "__init__.py").read_text(), ns)
        ver = ns["__version__"]
        exe = BUILD / f"VoxShift-Setup-{ver}.exe"
        notes = a.notes.read_text(encoding="utf-8") if a.notes else f"VoxShift {ver}"
        rid = ensure_release(f"v{ver}", f"VoxShift {ver}", notes, latest=True)
        upload(f"v{ver}", rid, [exe])


if __name__ == "__main__":
    main()
