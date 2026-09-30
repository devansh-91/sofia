"""Speech emotion recognition: how the user *sounds*, from the audio itself (pitch, energy, pace).

Runs on the CPU. The model was trained on acted English speech (IEMOCAP), so on
Hinglish it reads tone of voice only, and only four emotions.
"""
import os
from pathlib import Path

MODEL = os.environ.get("SOFIA_SER_MODEL", "superb/wav2vec2-base-superb-er")

# Model labels -> the mood names used everywhere else (sofia.memory.MOOD_NAMES).
_LABELS = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}
# Below this the model is guessing; say "neutral" rather than invent a feeling.
MIN_CONFIDENCE = 0.7
# Less than this is not enough speech to judge.
MIN_SECONDS = 1.0

_classifier = None


def _load():
    global _classifier
    if _classifier is None:
        from transformers import pipeline

        _classifier = pipeline("audio-classification", model=MODEL, device="cpu")
    return _classifier


def detect(wav: Path) -> tuple[str, float]:
    """(mood, confidence) for the speech in `wav`; ('neutral', 0.0) when unsure."""
    import soundfile

    audio, rate = soundfile.read(str(wav), dtype="float32")
    if len(audio) < MIN_SECONDS * rate:
        return "neutral", 0.0
    best = max(_load()({"raw": audio, "sampling_rate": rate}, top_k=None), key=lambda result: result["score"])
    mood = _LABELS.get(best["label"], "neutral")
    return (mood, best["score"]) if best["score"] >= MIN_CONFIDENCE else ("neutral", best["score"])
