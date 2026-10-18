"""IndicF5 (AI4Bharat, 330M) with hinglish-tts's duration patch, cloning the reference voice.

Fed Devanagari (preprocess=False), so the heavy IndicXlit transliterator isn't needed.
ai4bharat/IndicF5 is gated: accept its terms on Hugging Face first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

sys.path.insert(0, str(common.DATA / "src" / "hinglish-tts"))
import inference  # noqa: E402  (patches f5_tts's duration estimate on import)

SPEED = {"sad": 0.92, "lonely": 0.92, "tired": 0.92, "anxious": 0.94, "stressed": 0.94, "angry": 0.94,
         "happy": 1.08, "bored": 1.05}


def load(device: str):
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError

    common.notes["model"] = "ai4bharat/IndicF5 + hinglish-tts duration patch"
    try:
        auth_check("ai4bharat/IndicF5")
    except GatedRepoError:
        raise SystemExit("IndicF5 is gated: click 'Agree' on https://huggingface.co/ai4bharat/IndicF5, "
                         "then run bench/tts/run.sh indicf5") from None
    return inference.load_model(device=device)


def synth(model, line: dict):
    ref_wav, ref_text = common.reference()
    audio = inference.synthesize(model, common.fill(line["devanagari"]), ref_wav, ref_text,
                                 preprocess=False, speed=SPEED.get(line["mood"], 1.0))
    return audio, 24000


if __name__ == "__main__":
    common.run("indicf5", load, synth)
