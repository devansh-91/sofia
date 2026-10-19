"""Kokoro-82M (ONNX, CPU), exactly as Sofia speaks: sofia/tts.py, with its pace per mood.

Runs in Sofia's own .venv. Pick the voice with --voice (af_heart, af_bella, bf_emma, ...).
"""
import sys
from pathlib import Path

import soundfile as sf

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parents[1]))  # the Sofia repo
import common  # noqa: E402
from sofia import tts  # noqa: E402


def load(device: str):
    voice = common.args.voice or tts.VOICE
    common.notes["model"] = f"kokoro-v1.0 fp32, voice {voice}, pace per mood"
    tts._load()
    return voice


def synth(voice, line: dict):
    wav, _ = tts.speak(common.fill(line["text"]), line["mood"], voice)
    samples, rate = sf.read(wav, dtype="float32")
    wav.unlink()
    return samples, rate


if __name__ == "__main__":
    common.run("kokoro", load, synth, default_device="cpu")
