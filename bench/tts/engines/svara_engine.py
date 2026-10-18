"""svara-TTS v1 (Orpheus-style 3B, Indic) served by Ollama; the audio tokens are decoded here with SNAC.

    python svara_engine.py [--device cuda|cpu] [--only l01,l02]
    python svara_engine.py --make-ref     # speak REF_TEXT: the voice the cloning engines copy

Ollama adds the BOS token itself; sending <|begin_of_text|> again makes the model drift into text.
"""
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from snac import SNAC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402

MODEL = os.environ.get("SVARA_MODEL", "hf.co/mradermacher/svara-tts-v1-GGUF:Q4_K_M")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
SPEAKER = "Hindi (Female)"
RATE = 24000
# svara's style tags go at the end of the sentence; it has none for calm, firm or playful.
TAGS = {"happy": " <happy>", "sad": " <sad>", "lonely": " <sad>", "tired": " <sad>"}

AUDIO_START = 10  # <custom_token_10> is code 0 of the first of the 7 interleaved SNAC slots


def tokens(text: str, seed: int | None = None) -> list[int]:
    prompt = f"<custom_token_3><|audio|>{SPEAKER}: {text}<custom_token_4><|eot_id|><custom_token_5><custom_token_1>"
    options = {"temperature": 0.75, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1,
               "num_predict": 2400, "num_ctx": 4096, "stop": ["<custom_token_2>"]}
    if seed is not None:
        options["seed"] = seed
    body = {"model": MODEL, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "10m", "options": options}
    request = urllib.request.Request(f"{OLLAMA}/api/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        text_out = json.load(response)["response"]
    codes = []
    for number in map(int, re.findall(r"<custom_token_(\d+)>", text_out)):
        code = number - AUDIO_START - (len(codes) % 7) * 4096
        if 0 <= code < 4096:
            codes.append(code)
    if len(codes) < 7:
        raise RuntimeError(f"no audio tokens in the reply: {text_out[:200]!r}")
    return codes[: len(codes) // 7 * 7]


def decode(snac: SNAC, codes: list[int]) -> np.ndarray:
    frames = torch.tensor(codes).view(-1, 7)
    layers = [frames[:, 0].reshape(1, -1), frames[:, [1, 4]].reshape(1, -1), frames[:, [2, 3, 5, 6]].reshape(1, -1)]
    with torch.inference_mode():
        return snac.decode(layers).squeeze().numpy()


def load(device: str) -> SNAC:
    # Ollama decides where the 3B model goes; `device` only matters for the small SNAC decoder.
    common.notes["model"] = f"{MODEL}, speaker {SPEAKER}"
    return SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cpu")


def synth(snac: SNAC, line: dict) -> tuple[np.ndarray, int]:
    text = common.fill(line["devanagari"]) + TAGS.get(line["mood"], "")
    return decode(snac, tokens(text)), RATE


def make_ref() -> None:
    snac = load("cpu")
    audio = decode(snac, tokens(common.REF_TEXT, seed=7))
    common.REF_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(common.REF_DIR / "ref.wav", audio, RATE)
    (common.REF_DIR / "ref.txt").write_text(common.REF_TEXT + "\n")
    print(f"reference voice: {common.REF_DIR / 'ref.wav'} ({len(audio) / RATE:.1f}s)")


if __name__ == "__main__":
    if "--make-ref" in sys.argv:
        make_ref()
    else:
        common.run("svara", load, synth)
