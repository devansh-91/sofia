"""Build Sofia's fine-tuning set: the dialogues in training/dialogues/*.jsonl, plus every exchange
the user liked or corrected in her conversation log, as chat samples in exactly the shape she
sees at run time.

    .venv/bin/python training/build.py   -> ~/.local/share/sofia/training/data/{train,eval}.jsonl

Dialogues come from training/dialogues (in this repo) and ~/.local/share/sofia/training/dialogues
(about the user's own life, so kept out of the public repo).

A dialogue is one JSON line:

    {"bg": true, "heard": "sad", "turns": [{"u": "...", "s": "...", "n": ["[It is 2:40 AM, ...]"]}, ...]}

- "u" is the user's message, "s" Sofia's reply, "n" the bracketed notes that come with that
  message at run time (time of day, back after days, ...). Care notes (sofia/care.py) and the reply
  instruction are added here, as sofia/chat.py adds them.
- "heard" is the mood heard in the user's voice on the first message, if they spoke.
- "bg" puts what she knows about the user in front, from her real memory, as sofia/memory.py does.

The samples carry the Modelfile's system prompt but not its few-shot examples: those are what
the fine-tuning replaces, so the tuned model is served without them (a shorter prompt).
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from sofia import care  # noqa: E402
from sofia.chat import _user_turn  # noqa: E402
from sofia.log import GOOD, training_examples  # noqa: E402
from sofia.memory import NEUTRAL, Memory  # noqa: E402

MODELFILE = ROOT.parent / "models" / "Sofia.Modelfile"
EVAL_SHARE = 0.08
SEED = 7

# Dialogues about the user's own life stay out of this public repo, next to her memory.
PRIVATE = Path.home() / ".local/share/sofia/training/dialogues"
DIALOGUES = (ROOT / "dialogues", PRIVATE)
OUT = Path.home() / ".local/share/sofia/training/data"


def system_prompt() -> str:
    return re.search(r'^SYSTEM """(.*?)"""', MODELFILE.read_text(), re.S | re.M).group(1)


def background() -> list[dict]:
    """What she knows about the user, from her real memory, as she sees it at run time."""
    return Memory().background()


def user_turn(text: str, notes: list[str], heard: str = NEUTRAL) -> dict:
    voice = (f"[Their tone of voice might be {heard}, but that guess is often wrong: go by what they say.]"
             if heard != NEUTRAL else "")
    return _user_turn(text, voice, *notes, *care.notes(text)[0])


def sample(dialogue: dict, system: str) -> dict:
    messages = [{"role": "system", "content": system}]
    if dialogue.get("bg"):
        messages += background()
    for number, turn in enumerate(dialogue["turns"]):
        heard = dialogue.get("heard", NEUTRAL) if number == 0 else NEUTRAL
        messages.append(user_turn(turn["u"], turn.get("n", []), heard))
        messages.append({"role": "assistant", "content": turn["s"]})
    return {"messages": messages}


def main() -> None:
    system = system_prompt()
    dialogues = []
    for path in sorted(path for folder in DIALOGUES for path in folder.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    dialogues.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise SystemExit(f"{path.name}:{number}: {error}")
    # What the user liked or taught her (/good, /say) in real conversations counts too; unrated
    # exchanges don't: the early ones are from an older persona.
    for example in training_examples():
        if example["rating"] == GOOD:
            user, sofia = (m["content"] for m in example["messages"])
            dialogues.append({"turns": [{"u": user, "s": sofia}]})
    samples = [sample(d, system) for d in dialogues]
    random.Random(SEED).shuffle(samples)
    held = max(1, int(len(samples) * EVAL_SHARE))
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("eval", samples[:held]), ("train", samples[held:])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as file:
            for item in part:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
    replies = sum(len(d["turns"]) for d in dialogues)
    print(f"{len(dialogues)} dialogues, {replies} replies -> {out}/train.jsonl ({len(samples) - held}), eval.jsonl ({held})")


if __name__ == "__main__":
    main()
