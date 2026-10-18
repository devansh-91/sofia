"""Intelligibility check: Whisper transcribes every clip and we score it against the line it should say.

    .venv/bin/python bench/tts/check.py [engine ...]     (Sofia's venv; default: every engine with output)

A rough signal only: CER after stripping punctuation, nukta and spacing. Whisper itself
misspells Hindi, so compare engines with each other rather than reading the numbers as absolute.
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MODEL = os.environ.get("BENCH_WHISPER", "large-v3-turbo")
# Nudges Whisper to write English loanwords in Devanagari too, so script choice doesn't count as an error.
# Deliberately uses none of the test lines' words.
PROMPT = "मेरा लैपटॉप, मोबाइल और ऑफ़िस का कंप्यूटर।"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower()).replace("़", "")  # nukta: फ़ == फ for scoring
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", text))


def cer(expected: str, heard: str) -> float:
    a, b = normalize(expected), normalize(heard)
    if not a:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1] / len(a)


def to_16k(path: Path) -> np.ndarray:
    samples, rate = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != 16000:
        positions = np.arange(0, len(samples), rate / 16000)
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def main() -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    names = sys.argv[1:] or sorted(p.name for p in common.OUT.iterdir() if "@" not in p.name)
    every = {line["id"]: line for line in common.lines()}
    for name in names:
        report = {}
        for wav in sorted((common.OUT / name).glob("*.wav")):
            line = every.get(wav.stem)
            if not line:
                continue
            segments, _ = model.transcribe(to_16k(wav), language="hi", beam_size=3, condition_on_previous_text=False,
                                           initial_prompt=PROMPT)
            heard = " ".join(segment.text.strip() for segment in segments).strip()
            score = cer(common.fill(line["devanagari"], laugh="", sigh=""), heard)
            # Whisper still writes very English-sounding stretches in Latin script; CER can't judge those.
            mixed = bool(re.search(r"[a-zA-Z]{2,}", heard))
            report[wav.stem] = {"heard": heard, "cer": round(score, 3), "mixed_script": mixed}
            print(f"{name} {wav.stem} CER {score:.2f}  {heard}", flush=True)
        if not report:
            continue
        (common.OUT / name / "check.json").write_text(json.dumps({"whisper": MODEL, "lines": report},
                                                                  indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
