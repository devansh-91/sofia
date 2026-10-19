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
import hashlib
import json
import os
import re
import select
import shutil
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

# Each style as Turbo speaks it: (line, pace, pitch, tone, gain) for `clone`. Kokoro reads the line
# at that pace (over 5 seconds, which cloning needs), then its pitch is shifted (semitones), its top
# end raised or cut (tone, dB above 3 kHz) and its level changed (gain, dB) before Turbo clones it.
# Turbo copies pace only in part, but copies pitch and tone, and more than it is given. Kokoro read
# at different paces alone gave five clones with the same pitch (187-201 Hz) and timbre; with these,
# measured over two lines each: soft ~1 semitone lower and darker, calm a little lower and darker,
# bright and playful higher, brighter and with more pitch movement, all still 0.87-0.94 similar to
# warm (speaker embeddings). Bigger shifts came out exaggerated: +2 semitones gave +5.
TURBO_STYLES = {
    "warm": ("Good evening, sir. I was just thinking about you, actually. So tell me, how was your day? "
             "I hope it wasn't too long, because I really want to hear everything.", 1.0, 0, 0, 0),
    "soft": ("Hey, sir. I'm right here with you. Take your time, there's no rush at all. "
             "Whatever happened today, we can talk about it slowly, together.", 0.76, -1.2, -4, -2),
    "calm": ("Okay, sir, let's slow down for a second. Breathe with me. "
             "We'll sort this out one small step at a time, and it's going to be fine.", 0.89, -0.6, -2, -1),
    "bright": ("Oh, sir, that's wonderful news! I'm so happy for you, honestly. "
               "Tell me everything, I want to hear every single detail right now!", 1.15, 1.6, 3, 1),
    "playful": ("Hmm, sir... is there something you're not telling me? Come on, I can always tell "
                "when you're up to something. Go on then, spill it, I'm all ears.", 1.05, 0.8, 2, 0),
}

_kokoro = None
_turbo = None


def turbo_voice(voice: str | None = None, style: str = "warm") -> Path:
    """Where Turbo's clone of a Kokoro voice in one style is kept. Its name carries the style's
    settings, so changing them makes a new clone instead of reusing the old one."""
    voice = voice or VOICE
    if style == "warm":
        return VOICE_DIR / f"turbo-{voice}.pt"
    settings = hashlib.sha1(repr(TURBO_STYLES[style]).encode()).hexdigest()[:6]
    return VOICE_DIR / f"turbo-{voice}-{style}-{settings}.pt"


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

    def started(self) -> bool:
        """Whether the worker has finished loading, without waiting for it."""
        if not self.ready and select.select([self.process.stdout], [], [], 0)[0]:
            self.ready = bool(self._reply())
        return self.ready

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
    """Load Kokoro and start Turbo when it is installed and wanted, making any style clone it lacks
    first (on the CPU, ~10 s each, after a change to TURBO_STYLES)."""
    global _turbo
    if ENGINE == "turbo" and _turbo is None and TURBO_PYTHON.exists() and VOICE != "turbo":
        if any(not turbo_voice(style=style).exists() for style in TURBO_STYLES):
            print("Making her voice styles (once)...", flush=True)
            clone()
    voices = turbo_voices()
    if ENGINE == "turbo" and _turbo is None and TURBO_PYTHON.exists() and "warm" in voices:
        _turbo = Turbo(voices)
    return _kokoro_model()


def kokoro(text: str, path: Path, speed: float = 1.0, voice: str | None = None) -> None:
    import soundfile

    voice = voice or KOKORO_VOICE
    lang = "en-gb" if voice.startswith("b") else "en-us"
    samples, rate = _kokoro_model().create(without_sounds(text), voice=voice, speed=speed, lang=lang)
    soundfile.write(str(path), samples, rate)


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
FIRST_PIECE = 30  # characters: short, so she starts talking quickly
PIECE = 90  # later pieces are longer: fewer joins, and each renders while the one before plays


def pieces(text: str) -> list[str]:
    """`text` cut at sentence ends into pieces to speak one after another, so the first can play
    while the rest are still being rendered. A sound tag or a short last sentence stays with the
    piece before it."""
    found: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text.strip()):
        current = f"{current} {sentence}".strip()
        if len(current) >= (PIECE if found else FIRST_PIECE):
            found.append(current)
            current = ""
    if current:
        if found and len(without_sounds(current)) < FIRST_PIECE:
            found[-1] += " " + current
        else:
            found.append(current)
    return found


def speak(text: str, mood: str = "neutral", wait: bool = True) -> tuple[Path, str]:
    """Render `text` to a WAV file in the style that fits the user's `mood`; return (path, style name).
    Without `wait`, Kokoro says it when Turbo is still loading."""
    style, speed = STYLES.get(mood, STYLES["neutral"])
    path = audio.temporary_wav()
    turbo = _turbo
    if not (turbo and (wait or turbo.started()) and turbo.say(text, path, style)):
        kokoro(text, path, speed)
    return path, style


# The voices she can speak in, for the settings page: (key, name). "turbo:<kokoro voice>" is Turbo
# speaking a clone of that Kokoro voice (laughs, sighs, a style per mood; ~1 min to clone the first
# time it is picked); "turbo:builtin" is Turbo's own voice; "kokoro:<voice>" is Kokoro alone, on
# the CPU: lighter and quicker, without the sounds.
PACKAGES = [
    ("turbo:af_heart", "Heart, with laughs (her own voice)"),
    ("turbo:af_bella", "Bella, with laughs"),
    ("turbo:af_nicole", "Nicole, with laughs (soft)"),
    ("turbo:bf_emma", "Emma, with laughs (British)"),
    ("turbo:builtin", "Turbo's own voice, with laughs"),
    ("kokoro:af_heart", "Heart, light (CPU, no laughs)"),
    ("kokoro:af_sky", "Sky, light"),
    ("kokoro:af_sarah", "Sarah, light"),
    ("kokoro:bf_isabella", "Isabella, light (British)"),
]


def package() -> str:
    """The voice she speaks in now, as a PACKAGES key."""
    return f"{ENGINE}:{'builtin' if VOICE == 'turbo' else VOICE}"


def use(key: str) -> None:
    """Speak in another voice from PACKAGES: Turbo restarts with that voice (cloning it first if
    it is new, on the CPU), and Kokoro speaks until it is ready."""
    global ENGINE, VOICE, KOKORO_VOICE
    if key not in dict(PACKAGES):
        raise ValueError(f"no voice called {key}")
    engine, name = key.split(":", 1)
    stop_turbo()
    ENGINE, VOICE = engine, "turbo" if name == "builtin" else name
    KOKORO_VOICE = "af_heart" if VOICE == "turbo" else VOICE
    _load()


def stop_turbo() -> None:
    """Free Turbo's VRAM (for work mode's bigger model); Kokoro speaks until `_load` starts it again."""
    global _turbo
    turbo, _turbo = _turbo, None
    if turbo:
        with turbo.lock:  # not in the middle of a reply
            turbo.process.terminate()
            try:
                turbo.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                turbo.process.kill()


# What she says while a reply is on its way in a call, so the line doesn't go silent.
FILLERS = ("Hmm.", "Mm-hmm.", "Okay, sir.", "Hmm, let me think.", "Right, sir.")
FILLER_DIR = VOICE_DIR / "fillers"


def fillers() -> dict[str, list[Path]]:
    """Her fillers in each style, rendered once in the voice she speaks with and kept on disk."""
    found: dict[str, list[Path]] = {}
    for style, speed in sorted(set(STYLES.values())):
        found[style] = []
        for line in FILLERS:
            slug = re.sub(r"\W+", "-", line.lower()).strip("-")
            voice = turbo_voice(style=style).stem if _turbo else f"kokoro-{KOKORO_VOICE}-{style}"
            path = FILLER_DIR / f"{voice}-{slug}.wav"  # a new clone of a style gets new fillers
            if not path.exists():
                FILLER_DIR.mkdir(parents=True, exist_ok=True)
                rendered = audio.temporary_wav()
                if not (_turbo and _turbo.say(line, rendered, style)):
                    kokoro(line, rendered, speed)
                shutil.move(rendered, path)
            found[style].append(path)
    return found


def clone(voice: str | None = None) -> list[Path]:
    """Make Turbo's copies of a Kokoro voice, one per style (once; ./scripts/sofia.sh setup does it).
    Returns the new ones: none when they exist, for Turbo's own voice, or without Turbo."""
    voice = voice or VOICE
    if voice == "turbo" or not TURBO_PYTHON.exists():
        return []
    missing = {style: turbo_voice(voice, style) for style in TURBO_STYLES}
    missing = {style: target for style, target in missing.items() if not target.exists()}
    references = {style: audio.temporary_wav() for style in missing}
    TURBO_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        pairs = []
        for style, reference in references.items():
            line, speed, pitch, tone, gain = TURBO_STYLES[style]
            kokoro(line, reference, speed, voice)
            if pitch or tone or gain:
                shaped = audio.temporary_wav()
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(reference), "-af",
                                f"rubberband=pitch={2 ** (pitch / 12):.5f},highshelf=f=3000:g={tone},volume={gain}dB",
                                str(shaped)], check=True)
                shaped.replace(reference)
            pairs += [str(reference), str(missing[style])]
        if pairs:
            with TURBO_LOG.open("ab") as log:
                subprocess.run([str(TURBO_PYTHON), "-I", str(TURBO_WORKER), "--clone", *pairs],
                               check=True, stdout=log, stderr=log)
    finally:
        for reference in references.values():
            reference.unlink(missing_ok=True)
    return list(missing.values())
