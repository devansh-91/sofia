"""Baseline: Sofia's voice today. Piper's Hindi voice, then pitch and pace reshaped for the mood.

Runs in Sofia's own .venv. Sofia normally gets the Devanagari from the LLM; here it comes
from lines.json so every engine reads exactly the same words.
"""
import sys
import wave
from pathlib import Path

import soundfile as sf

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parents[1]))  # the Sofia repo
import common  # noqa: E402
from sofia import audio, tts  # noqa: E402
from sofia.lang import HINGLISH  # noqa: E402


def load(device: str):
    common.notes["model"] = f"{tts.VOICES[HINGLISH]} + rubberband pitch/tempo per mood"
    return tts._load(HINGLISH)


def synth(voice, line: dict):
    _, semitones, tempo = tts.STYLES.get(line["mood"], tts.STYLES["neutral"])
    raw = audio.temporary_wav()
    with wave.open(str(raw), "wb") as wav:
        voice.synthesize_wav(common.fill(line["devanagari"]), wav)
    shaped = audio.reshape(raw, audio.temporary_wav(), semitones, tempo)
    samples, rate = sf.read(shaped, dtype="float32")
    shaped.unlink()
    return samples, rate


if __name__ == "__main__":
    common.run("piper", load, synth, default_device="cpu")
