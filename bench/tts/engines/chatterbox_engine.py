"""Resemble Chatterbox-Turbo (350M) or its smaller Nano sibling (--voice turbo | nano), built-in voice.

English only, with paralinguistic tags: the {laugh} and {sigh} markers become [laugh] and [sigh].
"""
import sys
from pathlib import Path

from chatterbox.tts_turbo import ChatterboxTurboTTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def load(device: str):
    variant = common.args.voice or "turbo"
    if variant not in ("turbo", "nano"):
        raise SystemExit(f"--voice must be turbo or nano, not {variant}")
    common.notes["model"] = f"chatterbox-{variant}, built-in voice, [laugh]/[sigh] tags"
    return ChatterboxTurboTTS.from_pretrained(device, nano=variant == "nano")


def synth(model, line: dict):
    wav = model.generate(common.fill(line["text"], laugh="[laugh]", sigh="[sigh]"))
    return wav.squeeze(0).cpu().numpy(), model.sr


if __name__ == "__main__":
    common.run("chatterbox", load, synth)
