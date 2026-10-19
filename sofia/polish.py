"""Clean up Sofia's reply after generation.

Every reply is spoken aloud, so markdown, emojis and the bracketed reply
instruction the 4B model sometimes leaks into it are stripped.
"""
import re

_BRACKET_INSTRUCTION = re.compile(r"\s*\[[^\]]*\]\s*")
_MARKDOWN = re.compile(r"[*_`#]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")


def polish(answer: str) -> str:
    answer = _BRACKET_INSTRUCTION.sub(" ", answer)
    answer = _EMOJI.sub("", _MARKDOWN.sub("", answer))
    return re.sub(r"\s+", " ", answer).strip()
