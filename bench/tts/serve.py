"""Listening page for the TTS comparison (stdlib only).

    python3 bench/tts/serve.py [--host 0.0.0.0] [--port 8766]

Port 8766, because Sofia's own web app uses 8765.

Ratings are saved to $SOFIA_BENCH_DIR/ratings.json as you click.
"""
import argparse
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SOFIA_BENCH_DIR", "~/.local/share/sofia/tts-bench")).expanduser()
OUT = DATA / "out"
RATINGS = DATA / "ratings.json"
_lock = threading.Lock()


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def data() -> dict:
    engines = []
    for folder in sorted(OUT.iterdir()) if OUT.exists() else []:
        results = _read(folder / "results.json", None)
        if results:
            results["check"] = _read(folder / "check.json", {}).get("lines", {})
            results["clips"] = sorted(p.stem for p in folder.glob("*.wav"))
            engines.append(results)
    return {
        "lines": json.loads((HERE / "lines.json").read_text()),
        "engines": engines,
        "ratings": _read(RATINGS, {}),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, kind: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            return self._send((HERE / "compare.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/data":
            return self._send(json.dumps(data(), ensure_ascii=False).encode(), "application/json")
        match = re.fullmatch(r"/audio/([\w@.-]+)/([\w-]+)\.wav", self.path)
        if match:
            folder, clip = match.groups()
            path = OUT / folder / f"{clip}.wav"
            if ".." not in folder and path.is_file():
                return self._send(path.read_bytes(), "audio/wav")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        if self.path != "/api/rate":
            return self._send(b"not found", "text/plain", 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            engine, line, score = str(body["engine"]), str(body["line"]), int(body["score"])
        except (ValueError, KeyError, TypeError):
            return self._send(b"bad request", "text/plain", 400)
        with _lock:
            ratings = _read(RATINGS, {})
            if score:
                ratings.setdefault(engine, {})[line] = max(1, min(5, score))
            else:
                ratings.get(engine, {}).pop(line, None)
            RATINGS.write_text(json.dumps(ratings, indent=2))
        self._send(b"{}", "application/json")

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    print(f"TTS comparison: http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
