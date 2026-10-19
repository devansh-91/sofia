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
import queue
import secrets
import signal
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.parse
from urllib.parse import parse_qs, urlparse

from sofia import agent, care, chat, docs, ollama, settings, work
from sofia.chat import Session
from sofia.log import ConversationLog
from sofia.memory import NEUTRAL, Memory
from sofia.work import AGENT, ENTER, LEAVE, WORK_MODEL, Work

WEB_PAGE = Path(__file__).parent / "web" / "index.html"
ICON = Path(__file__).parent / "web" / "icon.svg"
KEY_PATH = Path(os.environ.get("SOFIA_KEY_FILE", "~/.local/share/sofia/access-key")).expanduser()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
KEPT_REPLIES = 40  # spoken pieces of replies kept on disk for the browser to fetch


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
        self.brain = self._brain(settings.load()["brain"])
        docs.refresh_soon()  # files added to ~/sofia-docs while she was off
        ollama.preload(chat.MODEL)  # alongside the speech models, which take longer
        self.session = Session(Memory(), ConversationLog())
        self.lock = threading.Lock()  # one reply at a time: a single model, a single conversation
        self.learned: list[str] = []  # learned from the latest message, after its answer went out
        self.replies: dict[str, threading.Event] = {}  # spoken pieces by name: set once rendered
        self.turn = 0  # a new reply stops the rest of the one before from being rendered
        self.audio_dir = Path(tempfile.mkdtemp(prefix="sofia-web-"))
        self.speech = False
        self.fillers: dict[str, list[Path]] = {}  # style -> her "hmm"s for calls, once rendered
        self.work: Work | None = None  # set in work mode ("kaam mode")
        self.switching_voice = False
        if speech:
            try:
                from sofia import emotion, stt, tts

                stt._load(), emotion._load()
                voice = settings.load()["voice"]
                if voice in dict(tts.PACKAGES):
                    tts.use(voice)
                else:
                    tts._load()
                self.speech = True
                threading.Thread(target=self._load_fillers, daemon=True).start()
            except (ImportError, FileNotFoundError) as error:
                print(f"Speech is off ({error}). Text chat still works; run ./scripts/sofia.sh setup for voice.")

    def _load_fillers(self) -> None:
        from sofia import tts

        try:
            self.fillers = tts.fillers()
        except Exception as error:  # a call simply has no fillers then
            print(f"(No fillers for calls: {type(error).__name__}: {error})", flush=True)

    def filler(self, name: str) -> Path | None:
        return next((path for paths in self.fillers.values() for path in paths if path.name == name), None)

    def style(self) -> str | None:
        """The style her voice speaks in for the user's current mood."""
        from sofia import tts

        return tts.STYLES.get(self.session.mood, tts.STYLES["neutral"])[0] if self.speech else None

    def _voice(self, wait: bool = True, mood: str | None = None) -> "Voice":
        """A reply's spoken pieces, rendered one after another in the background as they are added,
        possibly while the model is still writing the rest."""
        self.turn += 1
        return Voice(self, self.turn, mood or self.session.mood, wait)

    def _spoken(self, answer: str, wait: bool = True) -> list[str] | None:
        """Render `answer` to audio, a piece at a time in the background; return the names the
        browser fetches the pieces by, in order. Fetching one waits until it is rendered."""
        if not self.speech:
            return None
        from sofia import tts

        voice = self._voice(wait)
        for piece in tts.pieces(answer):
            voice.add(piece)
        voice.end()
        return voice.names

    def audio(self, name: str) -> Path | None:
        """A spoken piece, once rendered; None for anything else."""
        rendered = self.replies.get(name)
        if rendered is None or not rendered.wait(timeout=120):
            return None
        wav = self.audio_dir / name
        return wav if wav.exists() else None

    def hush(self) -> dict:
        """The user talked over her: stop rendering the rest of what she was saying."""
        self.turn += 1
        return {}

    # --- the settings page ---------------------------------------------------------------------

    @staticmethod
    def _has_model(name: str) -> bool:
        try:
            ollama._post("/api/show", {"model": name}, timeout=10)
            return True
        except OSError:
            return False

    def _brain(self, name: str) -> str:
        """Use brain `name` if its model is built, else the original; return the one in use."""
        name = name if name in settings.BRAINS and self._has_model(settings.BRAINS[name][0]) else "original"
        settings.use_brain(name)
        return name

    def settings(self) -> dict:
        from sofia import tts

        return {
            "voice": tts.package() if self.speech else None,
            "voices": [{"key": key, "name": name} for key, name in tts.PACKAGES] if self.speech else [],
            "switchingVoice": self.switching_voice,
            "brain": self.brain,
            "brains": [{"key": key, "name": label, "available": self._has_model(model)}
                       for key, (model, _, label) in settings.BRAINS.items()],
            "memory": self.session.memory.numbered(),
            "habits": self.session.memory.habits(),
        }

    def change(self, voice: str | None = None, brain: str | None = None) -> dict:
        from sofia import tts

        with self.lock:
            if brain and brain != self.brain and not self.work:
                ollama.unload(chat.MODEL)
                self.brain = self._brain(brain)
                settings.save(brain=self.brain)
                ollama.preload(chat.MODEL)
            if voice and self.speech and voice != tts.package() and not self.switching_voice:
                if voice not in dict(tts.PACKAGES):
                    return {"error": f"no voice called {voice}"}
                if self.work:  # the laughing voice is off in work mode: the bigger model has the GPU
                    return {**self.settings(), "error": "Switch voices after leaving work mode."}
                settings.save(voice=voice)
                self.switching_voice = True
                threading.Thread(target=self._switch_voice, args=(voice,), daemon=True).start()
        return self.settings()

    def _switch_voice(self, voice: str) -> None:
        """New voices take a minute or so: Turbo restarts, cloning the voice first if it is new."""
        from sofia import tts

        try:
            tts.use(voice)
            self._load_fillers()
        except Exception as error:
            print(f"(Couldn't switch her voice: {type(error).__name__}: {error})", flush=True)
        finally:
            self.switching_voice = False

    def memory(self, forget: int | str | None = None, add: str = "") -> dict:
        from sofia.memory import FACTS

        with self.lock:
            if forget == "all":
                self.session.memory.clear()
            elif isinstance(forget, int):
                self.session.memory.forget(forget)
            if add.strip():
                self.session.memory.add(FACTS, add.strip())
                self.session.memory.save()
        return self.settings()

    def history(self) -> list[dict]:
        return self.session.log.history()

    def forget_history(self) -> dict:
        """Delete the conversation log and what she holds of this conversation."""
        with self.lock:
            self.session.log.clear()
            self.session.history.clear()
            self.session.last, self.session.unreflected = None, []
            self.session.situation.last_turns = []
        return {"ok": True}

    def forget_everything(self) -> dict:
        """Every bit of the user's data: what she knows, their moods, the conversations."""
        self.memory(forget="all")
        return self.forget_history()

    @property
    def mode(self) -> str:
        return "work" if self.work else "normal"

    def _switch(self, to_work: bool) -> dict:
        """Into work mode: her 4B model and Turbo leave the GPU to the bigger model. Out of it:
        back again, with Kokoro speaking until Turbo has loaded."""
        if to_work == bool(self.work):
            line = "We're already in work mode, sir." if self.work else "I'm right here, sir, no work mode on."
        elif to_work:
            ollama.unload(chat.MODEL)
            if self.speech:
                from sofia import tts

                tts.stop_turbo()
            ollama.load(WORK_MODEL, {"num_ctx": work.CONTEXT})
            self.work = Work()
            line = "Work mode on, sir. What are we building?"
        else:
            if self.work.agent:
                self.work.agent.stop()
            ollama.unload(WORK_MODEL)
            ollama.unload(agent.MODEL)
            self.work = None
            ollama.preload(chat.MODEL)
            if self.speech:
                from sofia import tts

                tts._load()
            line = "Back to normal, sir. That was fun."
        return {"answer": line, "mode": self.mode, "audio": self._spoken(line, wait=False)}

    def _work_reply(self, text: str) -> dict:
        task = AGENT.match(text)
        if task and self.work.agent:
            said, cards = self.work.agent.run(task.group("task"))
            return self._agent_said(said, cards)
        return self._worked(self.work.answer(text))

    def _agent_said(self, said: str, cards: list[dict]) -> dict:
        if cards and not said:
            said = "Here's what I want to change, sir. Allow it and I'll carry on."
        return {"answer": said, "markdown": True, "looked": [], "proposed": cards, "mode": "work",
                "audio": self._spoken(work.spoken(said), wait=False)}

    def _worked(self, answer: str) -> dict:
        return {"answer": answer, "markdown": True, "looked": self.work.looked, "proposed": self.work.proposed,
                "mode": "work", "audio": self._spoken(work.spoken(answer), wait=False)}

    def decide(self, number: int | str, allowed: bool) -> dict:
        """The user pressed Allow or Deny on one of her proposals, or one of the agent's."""
        with self.lock:
            if self.work and self.work.agent and str(number).startswith("agent-"):
                answered = self.work.agent.decide(str(number), allowed)
                if answered is None:
                    return {"error": "That request is gone (it was already answered, or work mode was left)."}
                result = "allowed" if allowed else "denied"
                return {"decided": {"id": number, "allowed": allowed, "result": result}, **self._agent_said(*answered)}
            decided = self.work.decide(int(number), allowed) if self.work else None
            if decided is None:
                return {"error": "That proposal is gone (it was already decided, or work mode was left)."}
            result, answer = decided
            return {"decided": {"id": number, "allowed": allowed, "result": result}, **self._worked(answer)}

    def hello(self, device: str) -> dict:
        """She speaks first when the page opens, unless the user was here a moment ago."""
        with self.lock:
            if self.work:  # her greeting needs her own model, which work mode has unloaded
                return {}
            self.session.situation.device = device
            greeting = self.session.greet()
            return {"answer": greeting, "audio": self._spoken(greeting)} if greeting else {}

    def chat(self, text: str, sounds: str = NEUTRAL, device: str = "laptop") -> dict:
        """Her answer, with its audio already rendering. What she learns from the message (one more
        model call) is worked out after that, while she speaks, and comes with the next answer."""
        self.lock.acquire()
        learning = False
        try:
            self.session.situation.device = device
            if ENTER.match(text) or LEAVE.match(text):
                return self._switch(bool(ENTER.match(text)))
            if self.work:
                return self._work_reply(text)
            result = self.session.command(text)
            if result is not None:
                return {"command": result}
            # Her voice starts on the first sentence while the model writes the rest.
            voice = self._voice(mood=self.session.mood_for(text)) if self.speech else None
            try:
                answer = self.session.answer(text, sounds, emit=voice.add if voice else None)
                if voice and not voice.names:  # nothing was streamed (a lookup, or a slip into Hinglish)
                    from sofia import tts

                    for piece in tts.pieces(answer):
                        voice.add(piece)
            finally:
                if voice:
                    voice.end()
            learned, self.learned = self.learned, []
            reply = {"answer": answer, "learned": learned, "looked": self.session.looked,
                     "audio": voice.names if voice else None, "style": self.style(), "mode": "normal",
                     **({"care": care.HELPLINES} if self.session.crisis else {})}
            first = self.replies.get(reply["audio"][0]) if reply["audio"] else None
            threading.Thread(target=self._remember, args=(first,), daemon=True).start()
            learning = True
            return reply
        finally:
            if not learning:
                self.lock.release()

    def _remember(self, first_piece: threading.Event | None) -> None:
        """Learn from the latest message and let the next one in; holds the lock `chat` took.
        It waits for her first spoken piece: the model and her voice share the GPU."""
        try:
            if first_piece:
                first_piece.wait(timeout=15)
            _, learned = self.session.remember()
            self.learned += learned
        except Exception as error:  # Ollama hiccup: the answer is out, only this lesson is lost
            print(f"(Could not learn from that message: {type(error).__name__}: {error})", flush=True)
        finally:
            self.lock.release()

    def voice(self, upload: Path, device: str = "laptop", amend: bool = False) -> dict:
        """What the user said and her answer. With `amend` they went on talking while she was
        thinking: her answer to the first part is dropped and both parts get one answer."""
        from sofia import audio
        from sofia.voice import hear

        started = time.monotonic()
        wav = audio.to_wav(upload)
        try:
            text, sounds = hear(wav)
        finally:
            wav.unlink(missing_ok=True)
        if not text:
            return {"heard": ""}
        heard = time.monotonic()
        earlier = None
        if amend and not self.work:  # work mode keeps its own conversation
            with self.lock:
                earlier = self.session.retract()
        reply = self.chat(f"{earlier} {text}" if earlier else text, sounds, device)
        # How long the user waits in a call, without what was said: the log stays private.
        print(f"(voice turn: heard in {heard - started:.1f}s, answered in {time.monotonic() - heard:.1f}s)", flush=True)
        return {"heard": text, "sounds": sounds, **reply}


class Voice:
    """One reply's audio: pieces are rendered in order by one thread as they are added. A newer
    reply (or the user talking over her) stops the rest; their fetches get a 404."""

    def __init__(self, sofia: "Sofia", turn: int, mood: str, wait: bool):
        self.sofia, self.turn, self.mood, self.wait = sofia, turn, mood, wait
        self.names: list[str] = []
        self.queue: queue.Queue = queue.Queue()
        threading.Thread(target=self._render, daemon=True).start()

    def add(self, piece: str) -> str:
        name = f"{secrets.token_hex(8)}.wav"
        self.sofia.replies[name] = threading.Event()
        for old in list(self.sofia.replies)[:-KEPT_REPLIES]:
            (self.sofia.audio_dir / old).unlink(missing_ok=True)
            self.sofia.replies.pop(old).set()
        self.names.append(name)
        self.queue.put((name, piece))
        return name

    def end(self) -> None:
        self.queue.put(None)

    def _render(self) -> None:
        from sofia import tts

        while (item := self.queue.get()) is not None:
            name, piece = item
            try:
                if self.turn == self.sofia.turn:
                    wav, _ = tts.speak(piece, self.mood, self.wait)
                    wav.replace(self.sofia.audio_dir / name)
            finally:
                if name in self.sofia.replies:
                    self.sofia.replies[name].set()
                else:  # dropped as old while it rendered
                    (self.sofia.audio_dir / name).unlink(missing_ok=True)


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
        elif path == "/api/settings":
            self._json(self.sofia.settings())
        elif path == "/api/docs":
            self._json({"folder": str(docs.FOLDER), "docs": docs.listing()})
        elif path == "/api/history":
            self._json({"turns": self.sofia.history()})
        elif path == "/api/status":
            fillers = {style: [path.name for path in paths] for style, paths in self.sofia.fillers.items()}
            self._json({"speech": self.sofia.speech, "fillers": fillers, "mode": self.sofia.mode})
        elif path.startswith("/filler/"):
            wav = self.sofia.filler(Path(path).name)
            if wav and wav.exists():
                self._send(HTTPStatus.OK, wav.read_bytes(), "audio/wav")
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        elif path.startswith("/audio/"):
            wav = self.sofia.audio(Path(path).name)
            if wav:
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
            elif path == "/api/decide":
                request = json.loads(body)
                card = request.get("id", 0)
                card = card if isinstance(card, str) and card.startswith("agent-") else int(card)
                self._json(self.sofia.decide(card, request.get("allow") is True))
            elif path == "/api/settings":
                request = json.loads(body)
                self._json(self.sofia.change(voice=request.get("voice"), brain=request.get("brain")))
            elif path == "/api/memory":
                request = json.loads(body)
                forget = request.get("forget")
                forget = forget if forget == "all" else int(forget) if forget is not None else None
                self._json(self.sofia.memory(forget=forget, add=str(request.get("add", ""))))
            elif path == "/api/docs/upload":
                name = urllib.parse.unquote(self.headers.get("X-File-Name", ""))
                docs.add(name, body)
                self._json({"folder": str(docs.FOLDER), "docs": docs.listing()})
            elif path == "/api/docs/remove":
                docs.remove(str(json.loads(body).get("name", "")))
                self._json({"folder": str(docs.FOLDER), "docs": docs.listing()})
            elif path == "/api/history/clear":
                self._json(self.sofia.forget_history())
            elif path == "/api/data/clear":
                self._json(self.sofia.forget_everything())
            elif path == "/api/hush":
                self._json(self.sofia.hush())
            elif path == "/api/hello":
                self._json(self.sofia.hello(self._device()))
            elif path == "/api/voice" and self.sofia.speech:
                upload = self.sofia.audio_dir / f"upload-{secrets.token_hex(8)}"
                upload.write_bytes(body)
                try:
                    amend = parse_qs(urlparse(self.path).query).get("amend", [""])[0] == "1"
                    self._json(self.sofia.voice(upload, self._device(), amend))
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
        if Handler.sofia.work and Handler.sofia.work.agent:
            Handler.sofia.work.agent.stop()
            ollama.unload(agent.MODEL)
        ollama.unload(WORK_MODEL if Handler.sofia.work else chat.MODEL)


if __name__ == "__main__":
    main()
