"""Minimal Ollama chat client (stdlib only)."""
import json
import os
import threading
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"

# Same weights as `sofia`, without her persona and examples getting in the way
# (built by scripts/sofia.sh). Ollama runs both on one loaded copy.
BASE_MODEL = os.environ.get("SOFIA_BASE_MODEL", "sofia-base")

# How long the model stays in VRAM after her last reply. With Ollama's default of 5 minutes, the
# first reply after a short pause waited ~6 s for it to load again. Sofia unloads it when she stops.
KEEP_ALIVE = os.environ.get("SOFIA_KEEP_ALIVE", "60m")


def _post(path: str, body: dict, timeout: float = 300) -> dict:
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def chat_message(model: str, messages: list[dict], **extra) -> dict:
    """One non-streaming chat turn, as the message Ollama returns: with `tools`, it may hold
    tool_calls instead of text. Thinking is always off: this model loops with it on."""
    body = {"model": model, "messages": messages, "stream": False, "think": False, "keep_alive": KEEP_ALIVE, **extra}
    return _post("/api/chat", body)["message"]


def chat(model: str, messages: list[dict], **extra) -> str:
    return chat_message(model, messages, **extra)["content"].strip()


def load(model: str, options: dict | None = None) -> None:
    """Load `model` into VRAM now. `options` must match the chats' (num_ctx above all), or the
    first chat loads it all over again."""
    try:
        _post("/api/generate", {"model": model, "keep_alive": KEEP_ALIVE, **({"options": options} if options else {})})
    except OSError:
        pass  # Ollama is down: the first reply says so


def preload(model: str, options: dict | None = None) -> None:
    """Load `model` into VRAM in the background, so her first reply doesn't wait for it."""
    threading.Thread(target=load, args=(model, options), daemon=True).start()


def unload(model: str) -> None:
    """Free the VRAM `model` takes (and `BASE_MODEL` with it: they share one loaded copy)."""
    try:
        _post("/api/generate", {"model": model, "keep_alive": 0}, timeout=10)
    except OSError:
        pass
