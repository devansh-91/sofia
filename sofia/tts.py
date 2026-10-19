"""Sofia's voice.

With an NVIDIA GPU she speaks with Chatterbox-Turbo, in a clone of her Kokoro voice: it really
laughs, chuckles, sighs or gasps where her reply says [laugh], [chuckle], [sigh] or [gasp].
It runs in its own venv and process (sofia/chatterbox_worker.py) and needs ~1.4 GB of VRAM.

Kokoro-82M (ONNX, CPU) is always loaded too: it speaks when Turbo is not installed, is turned off
(SOFIA_TTS=kokoro) or fails on a reply. It sounds natural but skips the sounds, and the mood shows
only in her pace. Voice names starting with "a" are American English, "b" British;
SOFIA_VOICE=turbo picks Turbo's own built-in voice instead of a clone.
"""
import json
import os
import subprocess
import threading
from pathlib import Path

from sofia import audio
from sofia.polish import without_sounds

VOICE = os.environ.get("SOFIA_VOICE", "af_heart")
ENGINE = os.environ.get("SOFIA_TTS", "turbo")
VOICE_DIR = Path(os.environ.get("SOFIA_VOICE_DIR", "~/.local/share/sofia/voices")).expanduser()
MODEL = VOICE_DIR / "kokoro-v1.0.onnx"
VOICES = VOICE_DIR / "voices-v1.0.bin"
KOKORO_VOICE = "af_heart" if VOICE == "turbo" else VOICE

TURBO_PYTHON = Path(os.environ.get("SOFIA_TURBO_VENV", "~/.local/share/sofia/chatterbox")).expanduser() / "bin/python"
TURBO_WORKER = Path(__file__).with_name("chatterbox_worker.py")
TURBO_LOG = Path("~/.local/state/sofia/chatterbox.log").expanduser()
# Read in Kokoro's voice and cloned by Turbo (`clone`). Over 5 seconds, which Turbo needs.
CLONE_LINE = ("Good evening, sir. I was just thinking about you, actually. So tell me, how was your day? "
              "I hope it wasn't too long, because I really want to hear everything.")

# The user's mood -> how Sofia sounds: (name, Kokoro's speed).
STYLES = {
    "sad": ("soft", 0.9),
    "lonely": ("soft", 0.9),
    "tired": ("soft", 0.9),
    "anxious": ("calm", 0.95),
    "stressed": ("calm", 0.95),
    "angry": ("calm", 0.95),
    "happy": ("bright", 1.08),
    "bored": ("playful", 1.05),
    "neutral": ("warm", 1.0),
}

_kokoro = None
_turbo = None


def turbo_voice(voice: str = VOICE) -> Path | str:
    return "builtin" if voice == "turbo" else VOICE_DIR / f"turbo-{voice}.pt"


class Turbo:
    """The Chatterbox-Turbo worker process. It loads in the background while the rest of Sofia starts."""

    def __init__(self, voice: Path | str):
        TURBO_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TURBO_LOG.open("ab") as log:
            self.process = subprocess.Popen(
                [str(TURBO_PYTHON), "-I", str(TURBO_WORKER), str(voice)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
            )
        self.lock = threading.Lock()
        self.ready = False

    def _reply(self) -> dict | None:
        line = self.process.stdout.readline()
        return json.loads(line) if line else None

    def say(self, text: str, path: Path) -> bool:
        """Render `text` to `path`; False if Turbo could not."""
        with self.lock:
            try:
                if not self.ready:
                    self.ready = bool(self._reply())
                    if not self.ready:
                        raise BrokenPipeError
                self.process.stdin.write(json.dumps({"text": text, "path": str(path)}) + "\n")
                self.process.stdin.flush()
                reply = self._reply()
            except (BrokenPipeError, OSError):
                reply = None
        if reply is None:
            _turbo_stopped()
            return False
        if "error" in reply:
            print(f"(Kokoro said that one: the laughing voice failed with {reply['error']})")
        return bool(reply.get("ok"))


def _turbo_stopped() -> None:
    global _turbo
    _turbo = None
    print(f"(The laughing voice stopped — Sofia speaks with Kokoro now. Details: {TURBO_LOG})")


def _kokoro_model():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        for file in (MODEL, VOICES):
            if not file.exists():
                raise FileNotFoundError(f"{file} is missing — run: ./scripts/sofia.sh setup")
        _kokoro = Kokoro(str(MODEL), str(VOICES))
    return _kokoro


def _load():
    """Load Kokoro and start Turbo when it is installed and wanted."""
    global _turbo
    voice = turbo_voice()
    if ENGINE == "turbo" and _turbo is None and TURBO_PYTHON.exists() and (voice == "builtin" or voice.exists()):
        _turbo = Turbo(voice)
    return _kokoro_model()


def kokoro(text: str, path: Path, speed: float = 1.0, voice: str = KOKORO_VOICE) -> None:
    import soundfile

    lang = "en-gb" if voice.startswith("b") else "en-us"
    samples, rate = _kokoro_model().create(without_sounds(text), voice=voice, speed=speed, lang=lang)
    soundfile.write(str(path), samples, rate)


def speak(text: str, mood: str = "neutral") -> tuple[Path, str]:
    """Render `text` to a WAV file in the style that fits the user's `mood`; return (path, style name)."""
    style, speed = STYLES.get(mood, STYLES["neutral"])
    path = audio.temporary_wav()
    turbo = _turbo
    if not (turbo and turbo.say(text, path)):
        kokoro(text, path, speed)
    return path, style


def clone(voice: str = VOICE) -> Path | None:
    """Make Turbo's copy of a Kokoro voice (once; ./scripts/sofia.sh setup does it). None without Turbo."""
    target = turbo_voice(voice)
    if target == "builtin" or target.exists() or not TURBO_PYTHON.exists():
        return None
    reference = audio.temporary_wav()
    TURBO_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        kokoro(CLONE_LINE, reference, voice=voice)
        with TURBO_LOG.open("ab") as log:
            subprocess.run([str(TURBO_PYTHON), "-I", str(TURBO_WORKER), "--clone", str(reference), str(target)],
                           check=True, stdout=log, stderr=log)
    finally:
        reference.unlink(missing_ok=True)
    return target
