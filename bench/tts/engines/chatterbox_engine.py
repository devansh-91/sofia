"""Resemble Chatterbox-Turbo (350M) or its smaller Nano sibling, with paralinguistic tags: the {laugh}
and {sigh} markers become [laugh] and [sigh].

    --voice turbo | nano   built-in voice, the whole model on one device
    --voice sofia          exactly as Sofia speaks: Turbo split between GPU and CPU in half precision
                           (sofia/chatterbox_worker.py), in her cloned voice (SOFIA_VOICE, default af_heart)
                           and the style for each line's mood
"""
import sys
from pathlib import Path

from chatterbox.tts_turbo import ChatterboxTurboTTS

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH.parents[1]))  # the Sofia repo; tts and the worker need nothing from her .venv
import common  # noqa: E402
from sofia import chatterbox_worker, tts  # noqa: E402


def load(device: str):
    variant = common.args.voice or "turbo"
    if variant == "sofia":
        warm, *styles = tts.turbo_voices().items()
        common.notes["model"] = (f"chatterbox-turbo as in Sofia (fp16, GPU + CPU), voice {tts.VOICE}, "
                                 f"styles {', '.join(tts.turbo_voices())}")
        return chatterbox_worker.load(str(warm[1])), {style: chatterbox_worker.cloned(str(path)) for style, path in styles}
    if variant not in ("turbo", "nano"):
        raise SystemExit(f"--voice must be turbo, nano or sofia, not {variant}")
    common.notes["model"] = f"chatterbox-{variant}, built-in voice, [laugh]/[sigh] tags"
    return ChatterboxTurboTTS.from_pretrained(device, nano=variant == "nano")


def synth(model, line: dict):
    text = common.fill(line["text"], laugh="[laugh]", sigh="[sigh]")
    if common.args.voice == "sofia":
        model, styles = model
        return chatterbox_worker.speak(model, text, styles.get(tts.STYLES[line["mood"]][0])), model.sr
    wav = model.generate(text)
    return wav.squeeze(0).cpu().numpy(), model.sr


if __name__ == "__main__":
    common.run("chatterbox", load, synth)
