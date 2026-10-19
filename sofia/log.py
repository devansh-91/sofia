"""Conversation log: every exchange, kept locally as the raw material for fine-tuning later.

One JSON object per line in ~/.local/share/sofia/conversations.jsonl:

    {"type": "turn", "id": "20260930T121500-3", "time": "...", "language": "hinglish",
     "user": "...", "sofia": "..."}
    {"type": "feedback", "turn": "20260930T121500-3", "rating": "good"}
    {"type": "correction", "turn": "20260930T121500-3", "sofia": "the reply she should have given"}

    python3 -m sofia.log                  # how much has been collected
    python3 -m sofia.log export out.jsonl # chat-format training examples
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_PATH = Path(os.environ.get("SOFIA_LOG", "~/.local/share/sofia/conversations.jsonl")).expanduser()

GOOD, BAD = "good", "bad"


class ConversationLog:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self.session = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.turns = 0

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Private conversations: readable by the owner only.
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def turn(self, user: str, sofia: str, language: str) -> str:
        """Record one exchange; return its id for later feedback."""
        self.turns += 1
        turn_id = f"{self.session}-{self.turns}"
        self._append({"type": "turn", "id": turn_id, "time": datetime.now().isoformat(timespec="seconds"),
                      "language": language, "user": user, "sofia": sofia})
        return turn_id

    def last_turns(self, count: int) -> list[dict]:
        """The latest `count` exchanges in the log, oldest first."""
        if not self.path.exists():
            return []
        turns = []
        for line in reversed(self.path.read_text(encoding="utf-8").splitlines()):
            record = json.loads(line)
            if record["type"] == "turn":
                turns.append(record)
                if len(turns) == count:
                    break
        return turns[::-1]

    def feedback(self, turn_id: str, rating: str) -> None:
        self._append({"type": "feedback", "turn": turn_id, "rating": rating})

    def correction(self, turn_id: str, sofia: str) -> None:
        self._append({"type": "correction", "turn": turn_id, "sofia": sofia})


def training_examples(path: Path = DEFAULT_PATH) -> list[dict]:
    """Exchanges worth training on: corrected replies use the correction, replies
    rated bad (and not corrected) are left out."""
    turns: dict[str, dict] = {}
    if not Path(path).exists():
        return []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["type"] == "turn":
            turns[record["id"]] = {**record, "rating": None}
        elif record["turn"] in turns:
            if record["type"] == "correction":
                turns[record["turn"]].update(sofia=record["sofia"], rating=GOOD)
            else:
                turns[record["turn"]]["rating"] = record["rating"]
    return [
        {"language": t["language"], "rating": t["rating"],
         "messages": [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["sofia"]}]}
        for t in turns.values()
        if t["rating"] != BAD
    ]


def main() -> None:
    examples = training_examples()
    if len(sys.argv) == 3 and sys.argv[1] == "export":
        with open(sys.argv[2], "w", encoding="utf-8") as file:
            file.writelines(json.dumps(example, ensure_ascii=False) + "\n" for example in examples)
        print(f"wrote {len(examples)} examples to {sys.argv[2]}")
        return
    liked = sum(example["rating"] == GOOD for example in examples)
    print(f"{DEFAULT_PATH}\n{len(examples)} usable exchanges, {liked} of them liked or corrected by you")


if __name__ == "__main__":
    main()
