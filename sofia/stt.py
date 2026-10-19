"""Speech to text with faster-whisper: on the GPU when it can, else on the CPU.

On the GPU the small model takes ~0.4 GB of VRAM next to the LLM and Sofia's voice, and hears a
sentence in ~0.2 s instead of ~2 s on the CPU, which is most of the wait in a call. If the GPU
has no room for it (or CUDA is missing), or runs out of memory on a message, the CPU takes over.
"""
import ctypes
import os
import re
import sysconfig
from pathlib import Path

# "small" is the largest model that stays comfortably faster than real time on this CPU.
MODEL_SIZE = os.environ.get("SOFIA_STT_MODEL", "small")
DEVICE = os.environ.get("SOFIA_STT_DEVICE", "auto")  # auto, cuda or cpu

# One per core, not per hyperthread: on the 8-core / 12-thread laptop 8 threads hear a sentence
# ~0.5 s sooner than 12, which counts in a call.
THREADS = int(os.environ.get("SOFIA_STT_THREADS", min(8, os.cpu_count() or 4)))

# For Hindi speech: keeps English words in the mix instead of translating them away.
_PROMPT = "Yeh ek Hinglish conversation hai. यह हिंदी और English की बातचीत है।"
# Whisper hallucinates these on silence or noise — and it parrots its own prompt.
_NOISE = {"", "you", "thank you.", "thanks for watching!", "धन्यवाद", "शुक्रिया"}
_PROMPT_WORDS = set(re.findall(r"\w+", _PROMPT.lower()))
# Whisper spells her name the common way, and she then corrects the user for a spelling they never said.
_HER_NAME = re.compile(r"\b(Sophia|Sophie|Sofie)\b", re.I)

# On the CPU, telling Hindi from English is left to the tiny model: the small one spends ~1.2 s
# on it, a full pass of its own, before transcribing; tiny takes ~0.2 s. On the GPU the small
# one takes 0.1 s.
DETECT_SIZE = os.environ.get("SOFIA_STT_DETECT_MODEL", "tiny")

_models: dict[str, tuple] = {}  # device -> (model, language detector)


def _cuda_libraries() -> None:
    """Load the cuBLAS and cuDNN that pip installed into .venv (nvidia-*-cu12 wheels), which
    CTranslate2 would otherwise look for only on the system's library path."""
    root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for name in ("cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12", "cudnn/lib/libcudnn.so.9"):
        if (root / name).exists():
            ctypes.CDLL(str(root / name), mode=ctypes.RTLD_GLOBAL)
    for library in sorted((root / "cudnn/lib").glob("libcudnn_*.so.9")):
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _on(device: str) -> tuple:
    """(model, language detector) on `device`, loaded once."""
    if device not in _models:
        from faster_whisper import WhisperModel

        if device == "cuda":
            _cuda_libraries()
            model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="int8_float16")
            _models[device] = (model, model)
        else:
            model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=THREADS)
            detector = model if DETECT_SIZE == MODEL_SIZE else WhisperModel(
                DETECT_SIZE, device="cpu", compute_type="int8", cpu_threads=THREADS)
            _models[device] = (model, detector)
    return _models[device]


def _load() -> str:
    """Load the model on the GPU when it can be, else on the CPU; return which."""
    if not _models:
        if DEVICE in ("auto", "cuda"):
            try:
                _on("cuda")
                return "cuda"
            except (RuntimeError, OSError, ValueError) as error:  # no CUDA, or no room on the GPU
                if DEVICE == "cuda":
                    raise
                print(f"(Speech recognition runs on the CPU: {error})", flush=True)
        _on("cpu")
    return next(iter(_models))


def _transcribe(samples, device: str) -> str:
    model, detector = _on(device)
    # Choose between Hindi and English only: left to itself Whisper guesses Urdu, Punjabi
    # or Nepali for Hinglish, and the Hinglish prompt would drag real English into Hindi.
    _, _, probabilities = detector.detect_language(samples)
    chances = dict(probabilities)
    language = "en" if chances.get("en", 0) > chances.get("hi", 0) else "hi"
    segments, _ = model.transcribe(
        samples,
        language=language,
        beam_size=5 if device == "cuda" else 2,  # wider search costs nothing noticeable on the GPU
        vad_filter=True,  # drops silence, which is where most hallucinated text comes from
        condition_on_previous_text=False,
        initial_prompt=_PROMPT if language == "hi" else None,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def transcribe(wav: Path) -> str:
    """What was said in `wav` (16 kHz mono) — Hindi comes back in Devanagari, English in English; '' if nothing."""
    import soundfile

    # Decode the WAV ourselves: faster-whisper's own decoder breaks with newer PyAV releases.
    samples, _ = soundfile.read(str(wav), dtype="float32")
    device = _load()
    try:
        text = _transcribe(samples, device)
    except RuntimeError as error:  # the GPU ran out of memory: her voice can grow into it while she talks
        if device != "cuda":
            raise
        print(f"(Speech recognition fell back to the CPU for one message: {error})", flush=True)
        text = _transcribe(samples, "cpu")
    words = set(re.findall(r"\w+", text.lower()))
    return "" if text.lower() in _NOISE or words <= _PROMPT_WORDS else _HER_NAME.sub("Sofia", text)
