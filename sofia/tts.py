"""Sofia's voice: Piper text-to-speech on the CPU, with pitch and pace shaped by the moment.

Piper is not an emotional TTS — it has one way of speaking per voice. The
feeling comes from reshaping the audio afterwards: a softer, slower, lower
voice when the user is hurting, a brighter and quicker one when they are happy.
"""
import os
import re
import wave
from pathlib import Path

from sofia import audio, ollama
from sofia.lang import ENGLISH, HINGLISH

# One voice per language: the Hindi voice mangles English sentences, and the other way round.
VOICES = {
    HINGLISH: os.environ.get("SOFIA_VOICE", "hi_IN-priyamvada-medium"),
    ENGLISH: os.environ.get("SOFIA_VOICE_ENGLISH", "en_US-amy-medium"),
}
VOICE_DIR = Path(os.environ.get("SOFIA_VOICE_DIR", "~/.local/share/sofia/voices")).expanduser()

# The user's mood -> how Sofia sounds: (name, pitch shift in semitones, speed).
STYLES = {
    "sad": ("soft", -1.0, 0.92),
    "lonely": ("soft", -1.0, 0.92),
    "tired": ("soft", -1.0, 0.92),
    "anxious": ("calm", -0.5, 0.94),
    "stressed": ("calm", -0.5, 0.94),
    "angry": ("calm", -0.5, 0.94),
    "happy": ("bright", 1.5, 1.08),
    "bored": ("playful", 1.0, 1.05),
    "neutral": ("warm", 0.0, 1.0),
}

_DEVANAGARI_SYSTEM = (
    "Rewrite the Hinglish line in Devanagari script for a Hindi text-to-speech voice. Hindi words in "
    "proper Hindi spelling; English words written as they sound in Devanagari. Do not translate, do not "
    "add or drop words. Output only the rewritten line."
)
_DEVANAGARI_EXAMPLES = [
    ("Tension mat le yaar, abhi bhi time hai.", "टेंशन मत ले यार, अभी भी टाइम है।"),
    ("Main Sofia hoon, teri sabse pakki dost.", "मैं सोफ़िया हूँ, तेरी सबसे पक्की दोस्त।"),
    ("Waah, yeh toh mast news hai! Aaj party teri taraf se pakki.", "वाह, ये तो मस्त न्यूज़ है! आज पार्टी तेरी तरफ़ से पक्की।"),
]

_voices: dict = {}


def _load(language: str | None = None):
    """The Piper voice for `language`; with no language, load them all (warm-up)."""
    from piper import PiperVoice

    for name in VOICES.values() if language is None else [VOICES[language]]:
        if name not in _voices:
            model = VOICE_DIR / f"{name}.onnx"
            if not model.exists():
                raise FileNotFoundError(f"{model} is missing — run: ./scripts/sofia.sh setup")
            _voices[name] = PiperVoice.load(str(model))
    return _voices[VOICES[language]] if language else None


def to_devanagari(text: str) -> str:
    """The Hindi voice can only read Devanagari, so Roman Hinglish is respelled first."""
    messages = [{"role": "system", "content": _DEVANAGARI_SYSTEM}]
    for roman, devanagari in _DEVANAGARI_EXAMPLES:
        messages += [{"role": "user", "content": roman}, {"role": "assistant", "content": devanagari}]
    rewritten = ollama.chat(
        ollama.BASE_MODEL,
        messages + [{"role": "user", "content": text}],
        options={"temperature": 0, "num_predict": 300},
    )
    # If the model answered in Roman after all, speaking that is still better than silence.
    return rewritten if re.search(r"[ऀ-ॿ]", rewritten) else text


def speak(text: str, language: str = HINGLISH, mood: str = "neutral") -> tuple[Path, str]:
    """Render `text` to a WAV file in the style that fits the user's `mood`; return (path, style name)."""
    style, semitones, tempo = STYLES.get(mood, STYLES["neutral"])
    raw = audio.temporary_wav()
    with wave.open(str(raw), "wb") as wav:
        _load(language).synthesize_wav(to_devanagari(text) if language == HINGLISH else text, wav)
    return audio.reshape(raw, audio.temporary_wav(), semitones, tempo), style
