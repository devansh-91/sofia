"""Kyutai Pocket TTS (100M, CPU) with one of its built-in voices (--voice jane, eve, mary, anna, ...)."""
import sys
from pathlib import Path

from pocket_tts import TTSModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def load(device: str):
    voice = common.args.voice or "jane"
    model = TTSModel.load_model()
    common.notes["model"] = f"pocket-tts, built-in voice {voice}"
    return model, model.get_state_for_audio_prompt(voice)


def synth(loaded, line: dict):
    model, voice = loaded
    return model.generate_audio(voice, common.fill(line["text"])).numpy(), model.sample_rate


if __name__ == "__main__":
    common.run("pocket", load, synth, default_device="cpu")
