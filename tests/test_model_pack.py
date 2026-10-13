"""AI voice pack: pack -> serve -> resumable download -> verified reassembly."""
import http.server
import json
import threading
from pathlib import Path

import pytest

from voxshift.ai import model_pack as mp


@pytest.fixture
def model(tmp_path):
    d = tmp_path / "src" / "tiny-model"
    d.mkdir(parents=True)
    rng = __import__("random").Random(0)
    for name, size in (("a.onnx", 70_000), ("b.onnx", 5), ("config.json", 300), ("c.npz", 123_457)):
        (d / name).write_bytes(bytes(rng.getrandbits(8) for _ in range(size)))
    return d


def serve(root: Path, flaky: dict):
    """Static server honouring Range; drops chosen chunks halfway the first time."""

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            pass

        def do_GET(self):
            p = root / self.path.lstrip("/")
            data = p.read_bytes()
            start = int(self.headers["Range"].split("=")[1].split("-")[0]) if self.headers.get("Range") else 0
            body = data[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if flaky.pop(p.name, False):  # send half, then hang up
                self.wfile.write(body[: len(body) // 2])
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_pack_and_assemble_roundtrip(model, tmp_path):
    m = mp.pack(model, tmp_path / "pack", chunk_size=40_000, base_url="x/")
    assert len(m["chunks"]) == 5 and m["total_size"] == 193_762
    out = mp.assemble(json.loads((tmp_path / "pack" / "manifest.json").read_text()), tmp_path / "pack",
                      tmp_path / "dest")
    for f in model.iterdir():
        assert (out / f.name).read_bytes() == f.read_bytes()


def test_download_resumes_after_dropped_connection(model, tmp_path, monkeypatch):
    monkeypatch.setattr(mp.time, "sleep", lambda s: None)
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    flaky = {"tiny-model.part003": True}  # this chunk's first request is cut off halfway
    srv = serve(pack_dir, flaky)
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    m = mp.pack(model, pack_dir, chunk_size=40_000, base_url=base)
    seen = []
    out = mp.download(progress=lambda d, t: seen.append((d, t)), base_url=base, dest_root=tmp_path / "dest")
    srv.shutdown()
    assert not flaky, "the flaky chunk was requested"
    assert seen[-1] == (m["total_size"], m["total_size"])
    for f in model.iterdir():
        assert (out / f.name).read_bytes() == f.read_bytes()
    assert not (tmp_path / "dest" / ".download").exists()


def test_corrupt_chunk_is_rejected(model, tmp_path):
    m = mp.pack(model, tmp_path / "pack", chunk_size=40_000, base_url="x/")
    p = tmp_path / "pack" / m["chunks"][1]["name"]
    b = bytearray(p.read_bytes())
    b[100] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="corrupt"):
        mp.assemble(m, tmp_path / "pack", tmp_path / "dest")
