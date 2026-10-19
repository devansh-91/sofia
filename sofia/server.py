"""Sofia for the phone (or any browser): a small web app served from the laptop.

    ./scripts/sofia.sh serve

Text chat always works. If the speech libraries are installed she also speaks
her replies, and the mic button sends your voice (browsers only allow the
microphone on https or localhost — see the README for `tailscale serve`).

Every request needs the access key printed at start-up: this is an unfiltered
model with your personal memory behind it, so it is never served openly.
"""
import argparse
import json
import os
import secrets
import signal
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from sofia import ollama
from sofia.chat import MODEL, Session
from sofia.log import ConversationLog
from sofia.memory import NEUTRAL, Memory

WEB_PAGE = Path(__file__).parent / "web" / "index.html"
ICON = Path(__file__).parent / "web" / "icon.svg"
KEY_PATH = Path(os.environ.get("SOFIA_KEY_FILE", "~/.local/share/sofia/access-key")).expanduser()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
KEPT_REPLIES = 20  # spoken replies kept on disk for the browser to fetch


def access_key() -> str:
    """The key is created once and reused, so a bookmark on the phone keeps working."""
    if KEY_PATH.exists():
        return KEY_PATH.read_text().strip()
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(18)
    descriptor = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(key)
    return key


class Sofia:
    """The one conversation this server holds, plus speech when available."""

    def __init__(self, speech: bool):
        ollama.preload(MODEL)  # alongside the speech models, which take longer
        self.session = Session(Memory(), ConversationLog())
        self.lock = threading.Lock()  # one reply at a time: a single model, a single conversation
        self.replies: list[Path] = []
        self.audio_dir = Path(tempfile.mkdtemp(prefix="sofia-web-"))
        self.speech = False
        if speech:
            try:
                from sofia import emotion, stt, tts

                stt._load(), emotion._load(), tts._load()
                self.speech = True
            except (ImportError, FileNotFoundError) as error:
                print(f"Speech is off ({error}). Text chat still works; run ./scripts/sofia.sh setup for voice.")

    def _spoken(self, answer: str) -> str | None:
        """Render `answer` to audio; return the name the browser fetches it by."""
        if not self.speech:
            return None
        from sofia import tts

        wav, _ = tts.speak(answer, self.session.mood)
        kept = self.audio_dir / f"{secrets.token_hex(8)}.wav"
        wav.replace(kept)
        self.replies.append(kept)
        for old in self.replies[:-KEPT_REPLIES]:
            old.unlink(missing_ok=True)
        del self.replies[:-KEPT_REPLIES]
        return kept.name

    def hello(self, device: str) -> dict:
        """She speaks first when the page opens, unless the user was here a moment ago."""
        with self.lock:
            self.session.situation.device = device
            greeting = self.session.greet()
            return {"answer": greeting, "audio": self._spoken(greeting)} if greeting else {}

    def chat(self, text: str, sounds: str = NEUTRAL, device: str = "laptop") -> dict:
        with self.lock:
            self.session.situation.device = device
            result = self.session.command(text)
            if result is not None:
                return {"command": result}
            answer, learned = self.session.say(text, sounds)
            return {"answer": answer, "learned": learned, "looked": self.session.looked, "mood": self.session.mood, "audio": self._spoken(answer)}

    def voice(self, upload: Path, device: str = "laptop") -> dict:
        from sofia import audio
        from sofia.voice import hear

        wav = audio.to_wav(upload)
        try:
            text, sounds = hear(wav)
        finally:
            wav.unlink(missing_ok=True)
        if not text:
            return {"heard": ""}
        return {"heard": text, "sounds": sounds, **self.chat(text, sounds, device)}


class Handler(BaseHTTPRequestHandler):
    sofia: Sofia
    key: str

    def log_message(self, *args) -> None:  # the default logs every request, conversations included
        pass

    def _authorised(self) -> bool:
        offered = self.headers.get("X-Sofia-Key") or parse_qs(urlparse(self.path).query).get("key", [""])[0]
        return secrets.compare_digest(offered, self.key)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _device(self) -> str:
        """Which device the user is on: Sofia reads the phone as "they may be out"."""
        agent = self.headers.get("User-Agent", "")
        return "phone" if "Mobi" in agent or "Android" in agent else "laptop"

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD_BYTES:
            self._json({"error": "empty or too large"}, HTTPStatus.BAD_REQUEST)
            return None
        return self.rfile.read(length)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            # The page itself holds nothing private; the key it carries in the URL unlocks the API.
            self._send(HTTPStatus.OK, WEB_PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/icon.svg":
            self._send(HTTPStatus.OK, ICON.read_bytes(), "image/svg+xml")
        elif not self._authorised():
            self._json({"error": "wrong or missing key"}, HTTPStatus.FORBIDDEN)
        elif path == "/api/status":
            self._json({"speech": self.sofia.speech})
        elif path.startswith("/audio/"):
            wav = self.sofia.audio_dir / Path(path).name
            if wav in self.sofia.replies and wav.exists():
                self._send(HTTPStatus.OK, wav.read_bytes(), "audio/wav")
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not self._authorised():
            self._json({"error": "wrong or missing key"}, HTTPStatus.FORBIDDEN)
            return
        body = self._body()
        if body is None:
            return
        try:
            if path == "/api/chat":
                text = str(json.loads(body).get("text", "")).strip()
                self._json(self.sofia.chat(text, device=self._device()) if text else {"error": "empty message"})
            elif path == "/api/hello":
                self._json(self.sofia.hello(self._device()))
            elif path == "/api/voice" and self.sofia.speech:
                upload = self.sofia.audio_dir / f"upload-{secrets.token_hex(8)}"
                upload.write_bytes(body)
                try:
                    self._json(self.sofia.voice(upload, self._device()))
                finally:
                    upload.unlink(missing_ok=True)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # report it to the page instead of dropping the connection
            self._json({"error": f"{type(error).__name__}: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Sofia to a browser.")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this laptop only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-speech", action="store_true", help="text only, even if speech libraries are installed")
    arguments = parser.parse_args()
    # Started in the background (the app launcher does that), SIGINT arrives ignored, but it is how
    # the launcher stops her when her window closes.
    signal.signal(signal.SIGINT, signal.default_int_handler)

    Handler.key = access_key()
    Handler.sofia = Sofia(speech=not arguments.no_speech)
    server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
    print(f"Sofia is up ({'text + voice' if Handler.sofia.speech else 'text only'}).")
    print(f"  Open: http://{arguments.host}:{arguments.port}/?key={Handler.key}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.sofia.session.reflect()
        ollama.unload(MODEL)


if __name__ == "__main__":
    main()
