"""Kokoro-82M (ONNX, CPU), exactly as Sofia speaks without a GPU: sofia/tts.py, with its pace per mood.

Runs in Sofia's own .venv. Pick the voice with --voice (af_heart, af_bella, bf_emma, ...).
"""
import sys
from pathlib import Path

import soundfile as sf

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parents[1]))  # the Sofia repo
import common  # noqa: E402
from sofia import audio, tts  # noqa: E402


def load(device: str):
    voice = common.args.voice or tts.KOKORO_VOICE
    common.notes["model"] = f"kokoro-v1.0 fp32, voice {voice}, pace per mood"
    tts._kokoro_model()
    return voice


def synth(voice, line: dict):
    wav = audio.temporary_wav()
    tts.kokoro(common.fill(line["text"]), wav, tts.STYLES[line["mood"]][1], voice)
    samples, rate = sf.read(wav, dtype="float32")
    wav.unlink()
    return samples, rate


if __name__ == "__main__":
    common.run("kokoro", load, synth, default_device="cpu")
