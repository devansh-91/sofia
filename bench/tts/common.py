"""Shared harness for the TTS comparison: read the test lines, time an engine, save its WAVs and results.

Each engine script runs in its own venv (their dependencies clash), imports this module and calls run().
Needs only numpy and soundfile beyond the standard library.
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SOFIA_BENCH_DIR", "~/.local/share/sofia/tts-bench")).expanduser()
OUT = DATA / "out"
REF_DIR = DATA / "ref"

# The shared voice every cloning engine copies, so the comparison is about delivery, not timbre.
REF_TEXT = "नमस्ते, मैं सोफ़िया हूँ। आज मौसम बहुत अच्छा है, तो सोचा थोड़ी देर तुझसे बात कर लूँ।"

# Anything an engine wants to report alongside its numbers (which checkpoint it ended up using, etc.).
notes: dict = {}


def lines(only: str | None = None) -> list[dict]:
    every = json.loads((HERE / "lines.json").read_text())
    if not only:
        return every
    wanted = only.split(",")
    return [line for line in every if line["id"] in wanted]


def fill(text: str, laugh: str = "हाहा,", sigh: str = "") -> str:
    """Render the {laugh} / {sigh} markers the way this engine can say them."""
    return " ".join(text.replace("{laugh}", laugh).replace("{sigh}", sigh).split())


def reference() -> tuple[Path, str]:
    wav = REF_DIR / "ref.wav"
    if not wav.exists():
        raise SystemExit(f"{wav} is missing: run the svara engine with --make-ref first (run.sh does this)")
    return wav, (REF_DIR / "ref.txt").read_text().strip()


class GpuMemory:
    """Polls nvidia-smi, so engines running in another process (Ollama) are measured the same way."""

    def __init__(self):
        self.baseline = self._used()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    @staticmethod
    def _used() -> int:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            return int(out.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return 0

    def _poll(self) -> None:
        while not self._stop.wait(0.2):
            self.peak = max(self.peak, self._used())

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return self.peak - self.baseline


def _save(out: Path, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def run(engine: str, load, synth, default_device: str = "cuda") -> None:
    """`load(device)` returns a model; `synth(model, line)` returns (float32 samples, sample rate)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--only", help="comma-separated line ids")
    parser.add_argument("--tag", help="save as <engine>@<tag> (default: the device, if not the usual one)")
    args = parser.parse_args()
    tag = args.tag or (args.device if args.device != default_device else None)
    name = f"{engine}@{tag}" if tag else engine
    out = OUT / name
    summary = {"engine": engine, "name": name, "device": args.device, "notes": notes}

    gpu = GpuMemory()
    start = time.perf_counter()
    try:
        model = load(args.device)
        todo = lines(args.only)
        # The first call pays for compilation and caches; Sofia keeps the model warm, so time warm calls.
        synth(model, todo[0])
    except BaseException as error:  # SystemExit too: a missing reference must still show up on the page
        gpu.stop()
        traceback.print_exc()
        _save(out, {**summary, "error": f"{type(error).__name__}: {error}"})
        raise SystemExit(1)
    load_s = time.perf_counter() - start

    out.mkdir(parents=True, exist_ok=True)
    results = []
    for line in todo:
        started = time.perf_counter()
        try:
            audio, rate = synth(model, line)
        except Exception as error:
            traceback.print_exc()
            results.append({"id": line["id"], "error": f"{type(error).__name__}: {error}"})
            continue
        gen_s = time.perf_counter() - started
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        sf.write(out / f"{line['id']}.wav", audio, rate)
        seconds = len(audio) / rate
        rtf = gen_s / seconds if seconds else None
        results.append({"id": line["id"], "gen_s": round(gen_s, 2), "audio_s": round(seconds, 2),
                        "rtf": round(rtf, 3) if rtf else None})
        print(f"{name} {line['id']}: {seconds:.1f}s of audio in {gen_s:.1f}s (RTF {rtf or 0:.2f})", flush=True)

    rtfs = [r["rtf"] for r in results if r.get("rtf")]
    _save(out, {**summary, "load_s": round(load_s, 1), "vram_mb": gpu.stop(),
                "rtf": round(statistics.median(rtfs), 3) if rtfs else None, "lines": results})
