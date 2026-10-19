"""Sofia's voice.

With an NVIDIA GPU she speaks with Chatterbox-Turbo, in a clone of her Kokoro voice: it really
laughs, chuckles, sighs or gasps where her reply says [laugh], [chuckle], [sigh] or [gasp].
It runs in its own venv and process (sofia/chatterbox_worker.py) and needs ~1.4 GB of VRAM.
Turbo has no pace or emotion setting, but copies both from the voice it clones, so each style
(soft, calm, bright, ...) is a clone of its own, of Kokoro reading a fitting line at a fitting pace.

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

# Each style as Turbo speaks it: (line, pace) Kokoro reads for Turbo to clone (`clone`); lines are
# over 5 seconds, which cloning needs. Turbo copies the pace only in part. Measured against warm:
# soft ~12% slower, calm ~6% slower, bright and playful ~5% faster, bright with a livelier pitch.
TURBO_STYLES = {
    "warm": ("Good evening, sir. I was just thinking about you, actually. So tell me, how was your day? "
             "I hope it wasn't too long, because I really want to hear everything.", 1.0),
    "soft": ("Hey, sir. I'm right here with you. Take your time, there's no rush at all. "
             "Whatever happened today, we can talk about it slowly, together.", 0.8),
    "calm": ("Okay, sir, let's slow down for a second. Breathe with me. "
             "We'll sort this out one small step at a time, and it's going to be fine.", 0.92),
    "bright": ("Oh, sir, that's wonderful news! I'm so happy for you, honestly. "
               "Tell me everything, I want to hear every single detail right now!", 1.12),
    "playful": ("Hmm, sir... is there something you're not telling me? Come on, I can always tell "
                "when you're up to something. Go on then, spill it, I'm all ears.", 1.05),
}

_kokoro = None
_turbo = None


def turbo_voice(voice: str = VOICE, style: str = "warm") -> Path:
    """Where Turbo's clone of a Kokoro voice in one style is kept."""
    return VOICE_DIR / (f"turbo-{voice}.pt" if style == "warm" else f"turbo-{voice}-{style}.pt")


def turbo_voices() -> dict[str, Path | str]:
    """The styles Turbo can speak in, warm first: one for its built-in voice, else every clone made."""
    if VOICE == "turbo":
        return {"warm": "builtin"}
    voices = {style: turbo_voice(style=style) for style in TURBO_STYLES}
    return {style: path for style, path in voices.items() if path.exists()}


class Turbo:
    """The Chatterbox-Turbo worker process. It loads in the background while the rest of Sofia starts."""

    def __init__(self, voices: dict[str, Path | str]):
        warm, *styles = voices.items()
        TURBO_LOG.parent.mkdir(parents=True, exist_ok=True)
        with TURBO_LOG.open("ab") as log:
            self.process = subprocess.Popen(
                [str(TURBO_PYTHON), "-I", str(TURBO_WORKER), str(warm[1]), *(f"{s}={path}" for s, path in styles)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
            )
        self.lock = threading.Lock()
        self.ready = False

    def _reply(self) -> dict | None:
        line = self.process.stdout.readline()
        return json.loads(line) if line else None

    def say(self, text: str, path: Path, style: str = "warm") -> bool:
        """Render `text` to `path` in `style`; False if Turbo could not."""
        with self.lock:
            try:
                if not self.ready:
                    self.ready = bool(self._reply())
                    if not self.ready:
                        raise BrokenPipeError
                self.process.stdin.write(json.dumps({"text": text, "path": str(path), "style": style}) + "\n")
                self.process.stdin.flush()
                reply = self._reply()
            except (BrokenPipeError, OSError):
                reply = None
        if reply is None:
            try:
                code = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # alive but not answering
                self.process.kill()
                code = None
            _turbo_stopped(code)
            return False
        if "error" in reply:
            print(f"(Kokoro said that one: the laughing voice failed with {reply['error']})", flush=True)
        return bool(reply.get("ok"))


def _turbo_stopped(code: int | None) -> None:
    global _turbo
    _turbo = None
    # SIGKILL with no traceback in the log is the kernel's OOM killer: loading Turbo briefly needs ~3 GB of RAM.
    why = "it ran out of RAM (close something and restart Sofia)" if code == -9 else f"details: {TURBO_LOG}"
    print(f"(The laughing voice stopped, {why}. Sofia speaks with Kokoro now.)", flush=True)


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
    voices = turbo_voices()
    if ENGINE == "turbo" and _turbo is None and TURBO_PYTHON.exists() and "warm" in voices:
        _turbo = Turbo(voices)
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
    if not (turbo and turbo.say(text, path, style)):
        kokoro(text, path, speed)
    return path, style


def clone(voice: str = VOICE) -> list[Path]:
    """Make Turbo's copies of a Kokoro voice, one per style (once; ./scripts/sofia.sh setup does it).
    Returns the new ones: none when they exist, for Turbo's own voice, or without Turbo."""
    if voice == "turbo" or not TURBO_PYTHON.exists():
        return []
    missing = {style: turbo_voice(voice, style) for style in TURBO_STYLES}
    missing = {style: target for style, target in missing.items() if not target.exists()}
    references = {style: audio.temporary_wav() for style in missing}
    TURBO_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        pairs = []
        for style, reference in references.items():
            line, speed = TURBO_STYLES[style]
            kokoro(line, reference, speed, voice)
            pairs += [str(reference), str(missing[style])]
        if pairs:
            with TURBO_LOG.open("ab") as log:
                subprocess.run([str(TURBO_PYTHON), "-I", str(TURBO_WORKER), "--clone", *pairs],
                               check=True, stdout=log, stderr=log)
    finally:
        for reference in references.values():
            reference.unlink(missing_ok=True)
    return list(missing.values())
