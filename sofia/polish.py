"""Clean up Sofia's reply after generation.

Fixes the slips the 4B model keeps making no matter what the prompt says:
calling the user "bhai", using masculine verbs for herself, and leaking
markdown / emojis / the bracketed language instruction into a spoken reply.
"""
import re

from sofia.lang import HINGLISH

_QUOTED = re.compile(r'("[^"]*"|“[^”]*”)')
_BRACKET_INSTRUCTION = re.compile(r"\s*\[[^\]]*\]\s*")
_MARKDOWN = re.compile(r"[*_`#]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")

# "bhai" as a way of addressing the user — not "tera bhai" (a real brother).
_BHAI_OPENING = re.compile(r"(^|[.!?]\s+)((?:arre|are|oye|haan|nahi|chal|dekh|sun|bol|bolo)\s+)?bha+i(?:ya)?\b", re.I)
_BHAI_BEFORE_PUNCTUATION = re.compile(r"\bbha+i(?:ya)?\b(?=\s*[,.!?]|\s*$)", re.I)

# First-person masculine -> feminine. "hoon" is only ever first person, so these are safe.
_FEMININE = [
    (re.compile(r"\b(\w{2,})ta (hoon|hun|hu)\b", re.I), r"\1ti \2"),
    (re.compile(r"\braha (hoon|hun|hu)\b", re.I), r"rahi \1"),
    (re.compile(r"\b(ga|aa)ya (hoon|hun|hu)\b", re.I), r"\1yi \2"),
    (re.compile(r"\b(\w+)unga\b", re.I), r"\1ungi"),
]


def _match_case(original: str, replacement: str) -> str:
    return replacement.capitalize() if original[:1].isupper() else replacement


def _fix_hinglish(text: str) -> str:
    text = _BHAI_OPENING.sub(
        lambda m: f"{m.group(1)}{m.group(2)}yaar" if m.group(2) else f"{m.group(1)}Yaar", text
    )
    text = _BHAI_BEFORE_PUNCTUATION.sub(lambda m: _match_case(m.group(0), "yaar"), text)
    for pattern, replacement in _FEMININE:
        text = pattern.sub(replacement, text)
    return text


def polish(answer: str, language: str) -> str:
    answer = _BRACKET_INSTRUCTION.sub(" ", answer)
    answer = _EMOJI.sub("", _MARKDOWN.sub("", answer))
    if language == HINGLISH:
        # Leave quoted speech alone: that is usually a line she is suggesting the user say.
        parts = _QUOTED.split(answer)
        answer = "".join(p if i % 2 else _fix_hinglish(p) for i, p in enumerate(parts))
    return re.sub(r"\s+", " ", answer).strip()
