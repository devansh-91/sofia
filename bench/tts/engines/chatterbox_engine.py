"""Chatterbox (Resemble AI), cloning the reference voice; exaggeration carries the mood.

Prefers the Hindi single-language finetune. That repo ships only its own T3 and decoder,
so the voice encoder comes from the base repo. Falls back to the multilingual V3 model.
"""
import sys
from pathlib import Path

import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

# 0.5 is neutral; higher is more animated, lower calmer.
EXAGGERATION = {"happy": 0.8, "bored": 0.7, "sad": 0.4, "lonely": 0.4, "tired": 0.4,
                "anxious": 0.35, "stressed": 0.35, "angry": 0.35}


def _hindi_pack(decoder: str) -> Path:
    base = Path(snapshot_download("ResembleAI/chatterbox", allow_patterns=["ve.pt", "conds.pt", decoder]))
    hindi = Path(snapshot_download("ResembleAI/Chatterbox-Multilingual-hi"))
    pack = common.DATA / f"chatterbox-hi-{decoder.removesuffix('.pt')}"
    pack.mkdir(parents=True, exist_ok=True)
    files = {
        "ve.pt": base / "ve.pt",
        "conds.pt": base / "conds.pt",
        "t3_hi.safetensors": hindi / "t3_hi.safetensors",
        "s3gen.pt": (hindi if decoder == "s3gen_v3.pt" else base) / decoder,
        "grapheme_mtl_merged_expanded_v1.json": hindi / "grapheme_mtl_merged_expanded_v1.json",
    }
    for name, source in files.items():
        link = pack / name
        link.unlink(missing_ok=True)
        link.symlink_to(source.resolve())
    return pack


def _with_v3_decoder(device: str):
    """The Hindi T3 is trained against s3gen_v3, whose checkpoint lacks two buffers that S3Gen
    rebuilds itself (mel filters and STFT window), so it only loads non-strictly."""
    model = ChatterboxMultilingualTTS.from_local(_hindi_pack("s3gen.pt"), device, t3_model="t3_hi.safetensors")
    hindi = Path(snapshot_download("ResembleAI/Chatterbox-Multilingual-hi", allow_patterns=["s3gen_v3.pt"]))
    state = torch.load(hindi / "s3gen_v3.pt", map_location=device, weights_only=True)
    result = model.s3gen.load_state_dict(state, strict=False)
    unexpected_missing = [key for key in result.missing_keys if not key.endswith(("_mel_filters", "window"))]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(f"s3gen_v3 mismatch: missing {unexpected_missing}, unexpected {result.unexpected_keys}")
    return model


def load(device: str):
    attempts = [
        ("Hindi pack (t3_hi + s3gen_v3)", lambda: _with_v3_decoder(device)),
        ("Hindi pack (t3_hi + base s3gen)",
         lambda: ChatterboxMultilingualTTS.from_local(_hindi_pack("s3gen.pt"), device, t3_model="t3_hi.safetensors")),
        ("Multilingual V3", lambda: ChatterboxMultilingualTTS.from_pretrained(device, t3_model="v3")),
    ]
    errors = []
    for label, attempt in attempts:
        try:
            model = attempt()
        except Exception as error:
            print(f"{label} failed to load: {error}", flush=True)
            errors.append(f"{label}: {error}")
            continue
        common.notes["model"] = label
        ref_wav, _ = common.reference()
        model.prepare_conditionals(str(ref_wav))
        return model
    raise RuntimeError("no Chatterbox checkpoint loaded. " + " | ".join(errors))


def synth(model, line: dict):
    wav = model.generate(common.fill(line["devanagari"]), language_id="hi",
                         exaggeration=EXAGGERATION.get(line["mood"], 0.5), cfg_weight=0.5)
    return wav.squeeze(0).cpu().numpy(), model.sr


if __name__ == "__main__":
    common.run("chatterbox", load, synth)
