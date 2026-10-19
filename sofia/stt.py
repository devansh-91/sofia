"""Speech to text with faster-whisper, on the CPU so the GPU stays free for the LLM."""
import os
import re
from pathlib import Path

# "small" is the largest model that stays comfortably faster than real time on this CPU.
MODEL_SIZE = os.environ.get("SOFIA_STT_MODEL", "small")

# One per core, not per hyperthread: on the 8-core / 12-thread laptop 8 threads hear a sentence
# ~0.5 s sooner than 12, which counts in a call.
THREADS = int(os.environ.get("SOFIA_STT_THREADS", min(8, os.cpu_count() or 4)))

# For Hindi speech: keeps English words in the mix instead of translating them away.
_PROMPT = "Yeh ek Hinglish conversation hai. यह हिंदी और English की बातचीत है।"
# Whisper hallucinates these on silence or noise — and it parrots its own prompt.
_NOISE = {"", "you", "thank you.", "thanks for watching!", "धन्यवाद", "शुक्रिया"}
_PROMPT_WORDS = set(re.findall(r"\w+", _PROMPT.lower()))

# Telling Hindi from English is left to the tiny model: on the CPU the small one spends ~1.2 s on
# it, a full pass of its own, before transcribing; tiny takes ~0.1 s.
DETECT_SIZE = os.environ.get("SOFIA_STT_DETECT_MODEL", "tiny")

_model = None
_detector = None


def _load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=THREADS)
        _language_model()
    return _model


def _language_model():
    global _detector
    if _detector is None:
        from faster_whisper import WhisperModel

        _detector = _load() if DETECT_SIZE == MODEL_SIZE else WhisperModel(
            DETECT_SIZE, device="cpu", compute_type="int8", cpu_threads=THREADS)
    return _detector


def transcribe(wav: Path) -> str:
    """What was said in `wav` (16 kHz mono) — Hindi comes back in Devanagari, English in English; '' if nothing."""
    import soundfile

    # Decode the WAV ourselves: faster-whisper's own decoder breaks with newer PyAV releases.
    samples, _ = soundfile.read(str(wav), dtype="float32")
    model = _load()
    # Choose between Hindi and English only: left to itself Whisper guesses Urdu, Punjabi
    # or Nepali for Hinglish, and the Hinglish prompt would drag real English into Hindi.
    _, _, probabilities = _language_model().detect_language(samples)
    chances = dict(probabilities)
    language = "en" if chances.get("en", 0) > chances.get("hi", 0) else "hi"
    segments, _ = model.transcribe(
        samples,
        language=language,
        beam_size=2,
        vad_filter=True,  # drops silence, which is where most hallucinated text comes from
        condition_on_previous_text=False,
        initial_prompt=_PROMPT if language == "hi" else None,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    words = set(re.findall(r"\w+", text.lower()))
    return "" if text.lower() in _NOISE or words <= _PROMPT_WORDS else text
