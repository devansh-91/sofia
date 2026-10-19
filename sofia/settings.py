"""What the user picks on the settings page, kept across restarts: her voice and her brain.

    {"voice": "turbo:af_heart", "brain": "original"}     ~/.local/share/sofia/settings.json

Her brain is the model she answers with: "original" is `sofia` (the persona and its few-shot
examples on sofia-base), "tuned" is `sofia-tuned` (fine-tuned by scripts/finetune.sh). Each comes
with the base model her memory prompts use, so both run on one loaded copy.
"""
import json
import os
from pathlib import Path

PATH = Path(os.environ.get("SOFIA_SETTINGS", "~/.local/share/sofia/settings.json")).expanduser()
BRAINS = {
    "original": ("sofia", "sofia-base", "Original: her persona and examples"),
    "tuned": ("sofia-tuned", "sofia-tuned-base", "Tuned: fine-tuned on her dialogues"),
}
DEFAULTS = {"voice": None, "brain": "original"}  # no voice: SOFIA_VOICE / SOFIA_TTS decide


def load() -> dict:
    try:
        saved = json.loads(PATH.read_text())
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULTS, **{key: value for key, value in saved.items() if key in DEFAULTS}}


def save(**changes) -> dict:
    settings = {**load(), **changes}
    PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, indent=2))
    temporary.replace(PATH)
    return settings


def use_brain(name: str) -> None:
    """Answer with brain `name` from now on (her chat model and her memory prompts' model)."""
    from sofia import chat, memory, ollama

    chat.MODEL, base, _ = BRAINS[name]
    ollama.BASE_MODEL = memory.BASE_MODEL = base
