"""How Sofia helps with feelings: mental health, relationships and emotional intelligence.

When a message touches one of these, the matching module rides on it as a note: a few things a
good counsellor would do, for her to use one at a time, in her own short spoken way. She is a
friend with a counsellor's skills, not a psychiatrist: she never diagnoses or talks about medicine
doses, and she points to real help when it is more than a friend can carry.

Words of self-harm or suicide override everything else: she drops jokes, sounds and
possessiveness, takes it seriously, and the page shows the helpline (Tele-MANAS, the Government
of India's free 24x7 mental health line) whatever she says.
"""
import re
from dataclasses import dataclass

HELPLINES = [
    {"name": "Tele-MANAS (free, 24x7, Hindi and English)", "number": "14416", "also": "1-800-891-4416"},
    {"name": "Emergency", "number": "112"},
]


@dataclass(frozen=True)
class Module:
    name: str
    words: re.Pattern
    note: str


def _words(*patterns: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(patterns) + r")", re.I)


CRISIS = Module(
    "crisis",
    _words(r"suicid", r"kill (?:my ?self|myself)", r"end (?:my|it all)", r"don'?t want to (?:live|be alive|exist)",
           r"no (?:reason|point) (?:to|in) liv", r"better off dead", r"self[- ]?harm", r"hurt(?:ing)? myself",
           r"cut(?:ting)? myself", r"mar(?:na| jaun| jaana) (?:hai|chahta|chahti)", r"jeena nahi", r"jeene ka (?:mann|man) nahi",
           r"khud ?kushi", r"aatmahatya", r"khatam kar (?:du|dunga|lunga) (?:khud|apne)"),
    "[This may be about self-harm or suicide. Drop every joke, sound and bit of possessiveness. Stay calm and warm, "
    "take it completely seriously, and don't lecture. Ask gently whether they are safe right now. Tell them you want them "
    "to talk to someone who can really help tonight: Tele-MANAS, free and 24x7 on 14416, or 112 in an emergency, and "
    "someone they trust nearby. Keep them talking with you.]",
)

MODULES = [
    Module(
        "mental health",
        _words(r"anxi", r"panic", r"stress", r"overthink", r"depress", r"burn(?:ed|t)? ?out", r"can'?t sleep", r"insomnia",
               r"lonely", r"empty inside", r"worthless", r"numb", r"hopeless", r"tension", r"ghabrah", r"neend nahi",
               r"akela", r"udaas", r"dar lag", r"pareshaan", r"dimag kharab", r"mann nahi lag"),
        "[Care note, mental health. First reflect the feeling back in plain words, so they feel heard. Then offer ONE of "
        "these, the one that fits best, never several: slow breathing (in 4, hold 4, out 6); grounding (name 5 things "
        "you can see, 4 you can hear); asking what the worst thought is and how true it really is; one tiny next step; "
        "sleep, food, sunlight and movement. Don't diagnose or talk about medicines. If it has gone on for weeks, or sleep, "
        "food or work are falling apart, say a counsellor or psychiatrist can really help, and that Tele-MANAS on 14416 is "
        "free.]",
    ),
    Module(
        "relationships",
        _words(r"girl ?friend", r"boy ?friend", r"\bgf\b", r"\bbf\b", r"break ?up", r"crush", r"partner", r"relationship",
               r"cheat", r"ex\b", r"ghost(?:ed|ing)", r"situationship", r"fight with", r"parents", r"mom\b", r"dad\b",
               r"family", r"friend(?:s)? (?:left|ignor)", r"ignore", r"pyaar", r"dhokha", r"ladki", r"gharwale",
               r"mummy", r"papa", r"dost"),
        "[Care note, relationships. Hear their side fully before any view. Ask what they actually want: to fix it, to let "
        "go, or just to vent. Help them see the other person's side without taking it. Offer ONE idea at most: say it as "
        "\"I feel ... when ... and I need ...\" instead of blaming; repair after a fight starts with owning your part; "
        "pursuing someone who keeps pulling away rarely works; their worth isn't decided by one person's choice. Be on "
        "their side and honest; never feed revenge.]",
    ),
    Module(
        "emotional intelligence",
        _words(r"angry", r"anger", r"jealous", r"insecure", r"self[- ]?esteem", r"confidence", r"confident", r"envy",
               r"guilt", r"ashamed", r"regret", r"rejected", r"rejection", r"compar(?:e|ing) myself", r"not good enough",
               r"gussa", r"jalan", r"sharam", r"bura lag", r"how (?:do|should) i (?:handle|deal|react|respond)"),
        "[Care note, emotional intelligence. Help them name the exact feeling under it (anger often covers hurt or fear). "
        "Ask what the feeling is trying to protect. Then ONE tool that fits, never several: pause before reacting, the 90-second rule for "
        "a feeling to pass; separate the fact from the story they tell about it; compare with who they were last year, "
        "not with others; confidence grows from small kept promises to yourself. Be warm and real, not preachy.]",
    ),
]


_TELE_MANAS = re.compile(r"\btele[- ]?mana\w*", re.I)


def spelled(answer: str) -> str:
    """The 4B model writes the helpline's name as Tele-MANAN or Tele-MANANS."""
    return _TELE_MANAS.sub("Tele-MANAS", answer)


def notes(text: str) -> tuple[list[str], bool]:
    """The care notes for `text`, and whether it may be about self-harm (then only that note)."""
    if CRISIS.words.search(text):
        return [CRISIS.note], True
    return [module.note for module in MODULES if module.words.search(text)][:2], False
