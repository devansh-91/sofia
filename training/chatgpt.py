"""Read a ChatGPT data export: the user's own messages, for her memory and her training.

    .venv/bin/python training/chatgpt.py ~/Downloads/chatgpt-export.zip      (or its conversations.json)

Writes, next to nothing but this laptop's own folders:
- ~/.local/share/sofia/chatgpt/messages.jsonl: every message the user wrote, with its date and the
  conversation's title (the source for facts about them, and for how they really write);
- ~/.local/share/sofia/chatgpt/prompts.txt: short, chatty ones (no code, no pasted documents), which
  make good training messages once someone writes Sofia's replies to them.

ChatGPT's replies are left out: they are not how Sofia talks.
"""
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

OUT = Path.home() / ".local/share/sofia/chatgpt"
_CODE = re.compile(r"```|\bdef |\bimport |\bfunction\b|[{};]\s*$|Traceback|<\w+>", re.M)


def conversations(source: Path) -> list[dict]:
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            name = next(n for n in archive.namelist() if n.endswith("conversations.json"))
            return json.loads(archive.read(name))
    return json.loads(source.read_text(encoding="utf-8"))


def user_messages(conversation: dict):
    """(when, text) of each message the user wrote in one conversation, oldest first."""
    found = []
    for node in (conversation.get("mapping") or {}).values():
        message = node.get("message") or {}
        if (message.get("author") or {}).get("role") != "user":
            continue
        content = message.get("content") or {}
        parts = [part for part in content.get("parts") or [] if isinstance(part, str)]
        text = "\n".join(parts).strip()
        if text:
            found.append((message.get("create_time") or conversation.get("create_time") or 0, text))
    return sorted(found)


def chatty(text: str) -> bool:
    return 3 <= len(text) <= 280 and "\n\n" not in text and not _CODE.search(text) and "http" not in text


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, prompts, seen = [], [], set()
    for conversation in conversations(Path(sys.argv[1]).expanduser()):
        title = conversation.get("title") or ""
        for when, text in user_messages(conversation):
            date = datetime.fromtimestamp(when).strftime("%Y-%m-%d") if when else ""
            rows.append({"date": date, "title": title, "text": text})
            key = re.sub(r"\W+", " ", text.lower()).strip()
            if chatty(text) and key not in seen:
                seen.add(key)
                prompts.append(text.replace("\n", " "))
    with (OUT / "messages.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"{len(rows)} messages from you -> {OUT}/messages.jsonl; {len(prompts)} chatty ones -> {OUT}/prompts.txt")


if __name__ == "__main__":
    main()
