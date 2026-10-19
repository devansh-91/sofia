"""Resemble Chatterbox-Turbo (350M) or its smaller Nano sibling, with paralinguistic tags: the {laugh}
and {sigh} markers become [laugh] and [sigh].

    --voice turbo | nano   built-in voice, the whole model on one device
    --voice sofia          exactly as Sofia speaks: Turbo split between GPU and CPU in half precision
                           (sofia/chatterbox_worker.py), in her cloned voice (SOFIA_VOICE, default af_heart)
"""
import importlib.util
import os
import sys
from pathlib import Path

from chatterbox.tts_turbo import ChatterboxTurboTTS

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import common  # noqa: E402

# Sofia's worker by file: the sofia package itself needs her .venv.
_spec = importlib.util.spec_from_file_location("chatterbox_worker", BENCH.parents[1] / "sofia" / "chatterbox_worker.py")
chatterbox_worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chatterbox_worker)

VOICE_DIR = Path(os.environ.get("SOFIA_VOICE_DIR", "~/.local/share/sofia/voices")).expanduser()


def load(device: str):
    variant = common.args.voice or "turbo"
    if variant == "sofia":
        voice = os.environ.get("SOFIA_VOICE", "af_heart")
        conds = "builtin" if voice == "turbo" else VOICE_DIR / f"turbo-{voice}.pt"
        common.notes["model"] = f"chatterbox-turbo as in Sofia (fp16, GPU + CPU), voice {voice}"
        return chatterbox_worker.load(str(conds))
    if variant not in ("turbo", "nano"):
        raise SystemExit(f"--voice must be turbo, nano or sofia, not {variant}")
    common.notes["model"] = f"chatterbox-{variant}, built-in voice, [laugh]/[sigh] tags"
    return ChatterboxTurboTTS.from_pretrained(device, nano=variant == "nano")


def synth(model, line: dict):
    text = common.fill(line["text"], laugh="[laugh]", sigh="[sigh]")
    if common.args.voice == "sofia":
        return chatterbox_worker.speak(model, text), model.sr
    wav = model.generate(text)
    return wav.squeeze(0).cpu().numpy(), model.sr


if __name__ == "__main__":
    common.run("chatterbox", load, synth)
