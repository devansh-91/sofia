"""Intelligibility check: Whisper transcribes every clip and we score it against the line it should say.

    .venv/bin/python bench/tts/check.py [engine ...]     (Sofia's venv; default: every engine with output)

A rough signal only: CER after stripping punctuation, case and spacing. Compare engines with
each other rather than reading the numbers as absolute.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MODEL = os.environ.get("BENCH_WHISPER", "large-v3-turbo")


def normalize(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


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
            segments, _ = model.transcribe(to_16k(wav), language="en", beam_size=3, condition_on_previous_text=False)
            heard = " ".join(segment.text.strip() for segment in segments).strip()
            # Whisper writes numbers as digits ("42"), the lines spell them out.
            score = cer(common.fill(line["text"], laugh="", sigh=""), re.sub(r"\b42\b", "forty-two", heard))
            report[wav.stem] = {"heard": heard, "cer": round(score, 3)}
            print(f"{name} {wav.stem} CER {score:.2f}  {heard}", flush=True)
        if not report:
            continue
        (common.OUT / name / "check.json").write_text(json.dumps({"whisper": MODEL, "lines": report},
                                                                  indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
