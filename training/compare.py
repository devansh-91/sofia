"""Sofia before and after fine-tuning, side by side, on messages the tuning never saw.

    ./scripts/finetune.sh compare [more messages ...]

Each message goes to `sofia` (her persona and few-shot examples) and to `sofia-tuned` (the persona
only), both with the bracketed notes she gets at run time, through the same prompt as her chat.
Load one model at a time: both don't fit the GPU.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path.home() / ".local/share/sofia/training/data"  # training/build.py writes it
sys.path.insert(0, str(ROOT.parent))

from sofia import ollama  # noqa: E402

MODELS = ("sofia", "sofia-tuned")
EXTRA = [
    "yaar aaj gym nahi gaya, guilty feel ho raha hai",
    "I can't stop thinking about her",
    "ChatGPT gave me a better answer than you yesterday",
    "kal interview hai, dar lag raha hai",
    "roast my sleep schedule",
]


def held_out() -> list[list[dict]]:
    """The held-out chats up to their first reply: (system, maybe background, the message)."""
    chats = []
    for line in (DATA / "eval.jsonl").read_text().splitlines():
        messages = json.loads(line)["messages"]
        # The background exchange (sofia/memory.py) starts with a bracketed note: skip past it.
        first_user = min(i for i, m in enumerate(messages) if m["role"] == "user" and not m["content"].startswith("["))
        chats.append(messages[: first_user + 1])
    return chats


def main() -> None:
    from sofia.chat import _user_turn

    chats = held_out()
    for text in EXTRA + sys.argv[1:]:
        chats.append([_user_turn(text)])
    answers = {model: [] for model in MODELS}
    for model in MODELS:
        for chat in chats:
            # The served models carry the system prompt themselves.
            answers[model].append(ollama.chat(model, [m for m in chat if m["role"] != "system"]))
        ollama.unload(model)
    for number, chat in enumerate(chats):
        said = chat[-1]["content"].split("\n")[0]
        print(f"\n> {said}")
        for model in MODELS:
            print(f"  {model:12} {answers[model][number]}")


if __name__ == "__main__":
    main()
