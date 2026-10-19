"""Minimal Ollama chat client (stdlib only)."""
import json
import os
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_HOST.startswith("http"):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"

# Same weights as `sofia`, without her persona and examples getting in the way
# (built by scripts/sofia.sh). Ollama runs both on one loaded copy.
BASE_MODEL = os.environ.get("SOFIA_BASE_MODEL", "sofia-base")


def chat(model: str, messages: list[dict], **extra) -> str:
    """One non-streaming chat turn. Thinking is always off: this model loops with it on."""
    body = {"model": model, "messages": messages, "stream": False, "think": False, **extra}
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["message"]["content"].strip()
