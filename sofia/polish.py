"""Clean up Sofia's reply after generation.

Every reply is spoken aloud, so markdown, emojis and the bracketed reply
instruction the 4B model sometimes leaks into it are stripped. Sound tags stay:
Chatterbox-Turbo really laughs or sighs at [laugh] or [sigh], and Kokoro skips them.
Once the user's mood is known, `fit_sounds` drops the ones that don't fit it.
"""
import re

SOUNDS = ("laugh", "chuckle", "sigh", "gasp")
# The sounds that can fit how the user feels (other moods allow every sound). The Modelfile asks
# for the same, but the 4B model now and then gasps at sad news or laughs at an angry user.
MOOD_SOUNDS = {"sad": (), "lonely": (), "anxious": (), "angry": (), "stressed": ("sigh",), "tired": ("sigh",)}

# The model also writes them as stage directions: *laughs*, (sighs), [Chuckling].
_SOUND = re.compile(r"[\[(*]\s*(laugh|chuckle?|sigh|gasp)(?:s|ing)?\s*[\])*]", re.I)
_SOUND_TAG = re.compile(r"\s*\[(%s)\]\s*" % "|".join(SOUNDS))
_BRACKET = re.compile(r"\s*\[[^\]]*\]\s*")
_MARKDOWN = re.compile(r"[*_`#]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")


def _tag(match: re.Match) -> str:
    sound = match.group(1).lower()
    return f" [{'chuckle' if sound.startswith('chuckl') else sound}] "


def polish(answer: str) -> str:
    answer = _SOUND.sub(_tag, answer)
    answer = _BRACKET.sub(lambda m: m.group(0) if _SOUND_TAG.fullmatch(m.group(0)) else " ", answer)
    answer = _EMOJI.sub("", _MARKDOWN.sub("", answer))
    return re.sub(r"\s+", " ", answer).strip()


def without_sounds(text: str) -> str:
    """The reply as a voice that cannot make the sounds should say it."""
    return re.sub(r"\s+", " ", _SOUND_TAG.sub(" ", text)).strip()


def fit_sounds(answer: str, mood: str) -> str:
    """Keep at most one sound in `answer`, and only one that fits the user's `mood`."""
    allowed = MOOD_SOUNDS.get(mood, SOUNDS)
    kept = []

    def keep(match: re.Match) -> str:
        if kept or match.group(1) not in allowed:
            return " "
        kept.append(match.group(1))
        return match.group(0)

    return re.sub(r"\s+", " ", _SOUND_TAG.sub(keep, answer)).strip()
