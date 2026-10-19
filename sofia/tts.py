"""Sofia's voice: Kokoro text-to-speech (82M, ONNX) on the CPU, so the GPU stays free for the LLM.

Kokoro sounds natural but has no emotion control, so the moment shows in her
pace: slower and gentler when the user is hurting, a little quicker when they
are happy. Voice names starting with "a" are American English, "b" British.
"""
import os
from pathlib import Path

from sofia import audio

VOICE = os.environ.get("SOFIA_VOICE", "af_heart")
VOICE_DIR = Path(os.environ.get("SOFIA_VOICE_DIR", "~/.local/share/sofia/voices")).expanduser()
MODEL = VOICE_DIR / "kokoro-v1.0.onnx"
VOICES = VOICE_DIR / "voices-v1.0.bin"

# The user's mood -> how Sofia sounds: (name, speed).
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


def _load():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        for file in (MODEL, VOICES):
            if not file.exists():
                raise FileNotFoundError(f"{file} is missing — run: ./scripts/sofia.sh setup")
        _kokoro = Kokoro(str(MODEL), str(VOICES))
    return _kokoro


def speak(text: str, mood: str = "neutral", voice: str = VOICE) -> tuple[Path, str]:
    """Render `text` to a WAV file in the style that fits the user's `mood`; return (path, style name)."""
    import soundfile

    style, speed = STYLES.get(mood, STYLES["neutral"])
    samples, rate = _load().create(text, voice=voice, speed=speed, lang="en-gb" if voice.startswith("b") else "en-us")
    path = audio.temporary_wav()
    soundfile.write(str(path), samples, rate)
    return path, style
