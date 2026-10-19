"""Sofia's long-term memory: what she learns about the user while talking.

The model's weights never change. "Learning" here means: after every user
message a second, small prompt pulls out lasting facts, things the user taught
her, instructions, their mood, and whether the user liked her last reply.
Every few exchanges a reflection prompt looks at the conversation for patterns
in how the user behaves. All of it is saved to a JSON file and fed back into
every later prompt — in this session and the next ones. Replies the user liked
or corrected become extra examples of how to talk.
"""
import difflib
import json
import os
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from sofia import ollama

BASE_MODEL = ollama.BASE_MODEL
DEFAULT_PATH = Path(os.environ.get("SOFIA_MEMORY", "~/.local/share/sofia/memory.json")).expanduser()

FACTS = "facts"
BEHAVIOUR = "behaviour"  # patterns in how the user acts, found by `reflect`
TAUGHT = "taught"
PREFERENCES = "preferences"
EXAMPLES = "examples"  # {"user", "sofia", "language" of her reply}: replies the user liked or corrected
KINDS = (FACTS, BEHAVIOUR, TAUGHT, PREFERENCES, EXAMPLES)
MOODS = "moods"  # {"time", "mood", "language"} per user message; `habits` summarises them
# Facts, behaviour, taught items and preferences go into every prompt, so keep them well
# inside the 8192-token context, which lookups share. Only a few matching examples are used per reply.
LIMITS = {FACTS: 40, BEHAVIOUR: 12, TAUGHT: 20, PREFERENCES: 15, EXAMPLES: 40, MOODS: 300}
# Habits are only reported once there is enough to go on.
HABIT_MIN_MESSAGES = 10
HABIT_MIN_DAYS = 3  # one long chat says nothing about what is usual
HABIT_WINDOW = 60
EXAMPLES_PER_REPLY = 3
# An example is only shown for a message that resembles the one it answered (share of
# words in common). Shown for unrelated messages, she copies its pattern everywhere.
EXAMPLE_MIN_SIMILARITY = 0.3

FEEDBACK = "feedback"
GOOD, BAD, NONE = "good", "bad", "none"

MOOD = "mood"
NEUTRAL = "neutral"
MOOD_NAMES = ("happy", "sad", "angry", "stressed", "anxious", "tired", "bored", "lonely", NEUTRAL)

_EXTRACT_SYSTEM = """You maintain the long-term memory of a chat companion called Sofia. Read the user's message and extract only what is worth remembering for weeks.

"facts": lasting facts about the user's life: name, age, city, job, studies, family, partner, friends' names, pets, hobbies, favourite things, important upcoming events, long-term goals or problems.
"taught": things the user explains or teaches the companion about the world: what a word or slang means, how something works, a rule or opinion they want her to know.
"preferences": instructions about how the companion itself should talk or behave (what to call the user, tone, things to stop or start doing).
"mood": how the user feels in this message: happy, sad, angry, stressed, anxious, tired, bored, lonely, or neutral when it does not show.
"feedback": how the user reacts to the companion's previous reply: "good" only if they clearly praise or enjoy it, "bad" only if they clearly criticise it or say it made no sense, otherwise "none".

Rules:
- Write each item as one short English sentence: facts start with "User", preferences are imperatives, taught items state the thing taught.
- Do NOT record passing moods, what happened today, questions, greetings, jokes or requests.
- A request for right now ("roast me", "tell a joke", "gaali de") is NOT a preference. Preferences are standing rules only: "always", "never", "stop", "from now on", "mat kiya kar", "kiya kar".
- "Sofia" is the companion's own name, never the user's.
- Facts must stay true later: never write "today", "tomorrow", "kal", "next week" or "now". Say "User is 30 years old.", not "User turned 30 today." For an upcoming event, work out the real date from today's date and write that date.
- Most messages contain nothing worth remembering: then return empty lists.
- Never invent anything that is not stated in the message."""

def _found(facts=(), taught=(), preferences=(), feedback=NONE, mood=NEUTRAL) -> dict:
    return {FACTS: list(facts), TAUGHT: list(taught), PREFERENCES: list(preferences), MOOD: mood, FEEDBACK: feedback}


def _after(previous_reply: str, message: str) -> str:
    return f'(Companion\'s previous reply: "{previous_reply}")\n{message}' if previous_reply else message


_EXTRACT_EXAMPLES = [
    ("Mera naam Rohan hai aur main Pune mein rehta hoon.",
     _found(facts=["User's name is Rohan.", "User lives in Pune."])),
    (_after("Pura din kaam kiya hai tune, aaj bas so ja.", "Yaar aaj bahut thak gaya hoon."), _found(mood="tired")),
    (_after("Bas paani pi le aur sab bhool ja.", "Mujhe bhai mat bola kar, aur itna lamba jawab mat diya kar."),
     _found(preferences=["Never call the user 'bhai'.", "Keep replies short."], mood="angry")),
    ("I work as a nurse and my sister Priya is getting married in December.",
     _found(facts=["User works as a nurse.", "User's sister Priya is getting married in December."])),
    (_after("Tu wahi hai jo alarm laga ke bhi late uthta hai.", "Hahaha yeh mast tha, ekdum sahi pakda!"),
     _found(feedback=GOOD, mood="happy")),
    ("Ek ganda joke suna.", _found()),
    ("Hey Sofia, kya chal raha hai?", _found()),
    (_after("Kya hua yaar?", "Mujhe roast kar, full savage."), _found()),
    (_after("Main sun rahi hoon.", "Aaj boss ne sabke saamne daant diya."), _found(mood="sad")),
    ("Kal presentation hai aur kuch ready nahi, dimaag phat raha hai.", _found(mood="stressed")),
    (_after("Kya kar raha hai?", "Kuch nahi yaar, akela baitha hoon, koi baat karne wala nahi."), _found(mood="lonely")),
    ("Guess what, I turned 30 today!", _found(facts=["User is 30 years old."], mood="happy")),
    (_after("Uska jhanda khaate hain baad mein.", "Yeh kya bol rahi hai? Iska koi matlab nahi banta."),
     _found(feedback=BAD, mood="angry")),
    ("Kal meri behen ki shaadi hai.", _found(facts=["User's sister is getting married on 13 March 2025."])),
    (_after("Just order pizza, keep it simple.", "What should I eat tomorrow then?"), _found()),
    ("Hyderabad mein 'mama' ka matlab dost hota hai, mama nahi.",
     _found(taught=["In Hyderabad, 'mama' means friend, not uncle."])),
    (_after("Kya hua yaar, bata mujhe.", "haha ok"), _found()),
    ("Mujhe Sam bula, aur mujhe chai bahut pasand hai.",
     _found(facts=["User loves chai."], preferences=["Call the user Sam."])),
    (_after("I'm right here, tell me what happened.", "That's exactly what I needed to hear, thanks."),
     _found(feedback=GOOD, mood="happy")),
]

_SENTENCE = {"type": "string", "maxLength": 100}

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        # The caps matter: without them the model sometimes spins out an endless list,
        # or starts thinking aloud inside one item.
        FACTS: {"type": "array", "items": _SENTENCE, "maxItems": 6},
        TAUGHT: {"type": "array", "items": _SENTENCE, "maxItems": 2},
        PREFERENCES: {"type": "array", "items": _SENTENCE, "maxItems": 3},
        MOOD: {"type": "string", "enum": list(MOOD_NAMES)},
        FEEDBACK: {"type": "string", "enum": [GOOD, BAD, NONE]},
    },
    "required": [FACTS, TAUGHT, PREFERENCES, MOOD, FEEDBACK],
}

_UPDATE_SYSTEM = """You keep a list of facts about a user up to date. You get the numbered existing facts and one NEW fact.
If the NEW fact corrects or updates one existing fact (same topic, the old one is no longer true), answer with that fact's number.
If it is simply additional information, answer 0."""

_UPDATE_EXAMPLES = [
    ("1. User lives in Pune.\n2. User has a cat named Mitthu.\nNEW: User lives in Mumbai.", 1),
    ("1. User plays cricket.\n2. User is 22 years old.\nNEW: User plays chess.", 0),
    ("1. User works as a teacher.\n2. User's girlfriend's name is Neha.\nNEW: User is single.", 2),
]

_UPDATE_SCHEMA = {"type": "object", "properties": {"replaces": {"type": "integer"}}, "required": ["replaces"]}


def _few_shot(system: str, examples, final: str, prefix=lambda day: "") -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for user, answer in examples:
        if not isinstance(answer, dict):
            answer = {"replaces": answer}
        user = prefix(_EXAMPLE_DAY) + user
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": json.dumps(answer)}]
    return messages + [{"role": "user", "content": prefix(date.today()) + final}]


# The extraction examples pretend to be written on this day ("kal" = 13 March 2025).
_EXAMPLE_DAY = date(2025, 3, 12)


def _dated(day: date) -> str:
    return f"(Today is {day:%A}, {day.day} {day:%B %Y}.)\n"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# Dates are worked out from today's date, so they need not appear in the message.
_CALENDAR_WORDS = frozenset(
    "January February March April May June July August September October November December "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
)


def is_grounded(item: str, message: str) -> bool:
    """Reject items that name something the user never said (the extractor's
    usual way of making things up is to reuse a name from its own examples)."""
    said = set(re.findall(r"\w+", message.lower()))
    body = item.split(" ", 1)[1] if " " in item else ""
    names = re.findall(r"\b[A-Z][\w]*", body) + re.findall(r"'([^']+)'", body)
    names = [n for n in names if n not in {"User", "User's"} and n not in _CALENDAR_WORDS]
    return all(word in said for name in names for word in re.findall(r"\w+", name.lower()))


_MONTHS = "January February March April May June July August September October November December".split()
_FULL_DATE = re.compile(rf"\b(\d{{1,2}}) ({'|'.join(_MONTHS)}) (\d{{4}})\b")


def has_real_dates(item: str) -> bool:
    """The model's date arithmetic sometimes lands on days like "31 September"."""
    for day, month, year in _FULL_DATE.findall(item):
        try:
            date(int(year), _MONTHS.index(month) + 1, int(day))
        except ValueError:
            return False
    return True


# Facts phrased relative to "now" go stale the next day.
_PASSING = re.compile(r"\b(today|tonight|tomorrow|yesterday|currently|right now|this morning|this evening)\b", re.I)


# A preference is a standing rule, and a wrong one bends every later reply ("Mujhe roast
# kar." once became "Roast the user." and she roasted everything). So a preference is only
# accepted from a message that actually words a rule.
_STANDING_RULE = re.compile(
    r"\b(always|never|stop|don'?t|do not|from now|whenever|every ?time|call me|keep|instead|less|more"
    r"|hamesha|kabhi|mat|band|bula|bulaya|har baar|ab se|aage se|zyada|kam"
    r"|(kiya|diya|liya|bola|bula|suna|rakha) kar)\b",
    re.I,
)


def extract(message: str, previous_reply: str = "") -> dict:
    """Ask the model what in this user message is worth remembering, and whether
    it praises or criticises Sofia's previous reply."""
    raw = ollama.chat(
        BASE_MODEL,
        _few_shot(_EXTRACT_SYSTEM, _EXTRACT_EXAMPLES, _after(previous_reply, message), prefix=_dated),
        format=_EXTRACT_SCHEMA,
        options={"temperature": 0, "num_predict": 400},
    )
    try:
        found = json.loads(raw)
    except json.JSONDecodeError:
        return _found()
    kept = {
        kind: [
            item.strip()
            for item in found.get(kind, [])
            if isinstance(item, str) and item.strip() and is_grounded(item, message) and has_real_dates(item)
        ]
        for kind in (FACTS, TAUGHT, PREFERENCES)
    }
    kept[FACTS] = [fact for fact in kept[FACTS] if not _PASSING.search(fact)]
    if not _STANDING_RULE.search(message):
        kept[PREFERENCES] = []
    feedback = found.get(FEEDBACK)
    mood = found.get(MOOD)
    return {
        **kept,
        MOOD: mood if mood in MOOD_NAMES else NEUTRAL,
        FEEDBACK: feedback if previous_reply and feedback in (GOOD, BAD) else NONE,
    }


def replaced_fact(existing: list[str], new: str) -> int | None:
    """Index of the existing fact that `new` makes outdated, if any."""
    if not existing:
        return None
    listing = "\n".join(f"{i}. {fact}" for i, fact in enumerate(existing, 1))
    raw = ollama.chat(
        BASE_MODEL,
        _few_shot(_UPDATE_SYSTEM, _UPDATE_EXAMPLES, f"{listing}\nNEW: {new}"),
        format=_UPDATE_SCHEMA,
        options={"temperature": 0, "num_predict": 20},
    )
    try:
        number = json.loads(raw).get("replaces", 0)
    except json.JSONDecodeError:
        return None
    return number - 1 if isinstance(number, int) and 1 <= number <= len(existing) else None


_REFLECT_SYSTEM = """You study how a user behaves when chatting with their companion Sofia, so that she can understand them better.

From the conversation, write the single clearest observation about the USER's behaviour or personality: how they talk, what they keep coming back to, how they react to Sofia's replies, what cheers them up or annoys them, their habits and how they cope.

Rules:
- One short English sentence starting with "User".
- It must be a pattern supported by at least two of the user's messages, not a single event.
- Not plain facts (name, job, city) and not what happened today.
- Do not repeat anything in the "Already known" list.
- If no clear pattern shows, return an empty list."""

_REFLECT_EXAMPLES = [
    ("""Already known: (nothing)
User: Boss ne aaj phir daant diya.
Sofia: Kya bola usne?
User: Chhod na, mujhe roast kar thoda.
Sofia: Tu wahi hai jo alarm laga ke bhi late uthta hai.
User: Hahaha mast tha.
User: Office ka soch ke hi dimaag kharab hota hai.
Sofia: Thoda relax kar le.
User: Tu relax mat bola kar, irritate hota hoon.
User: Ek aur roast maar.""",
     {"observations": ["User copes with work stress by asking for jokes and roasts, and gets irritated when told to relax."]}),
    ("""Already known: User likes being roasted.
User: Hey
Sofia: Hey, what's up?
User: Nothing much.
Sofia: Same here.
User: ok bye""",
     {"observations": []}),
]

_REFLECT_SCHEMA = {
    "type": "object",
    "properties": {"observations": {"type": "array", "items": {"type": "string", "maxLength": 160}, "maxItems": 1}},
    "required": ["observations"],
}


def observe(exchanges: list[dict], known: list[str]) -> list[str]:
    """Ask the model for a pattern in how the user behaved over `exchanges` ({"user", "sofia"})."""
    lines = [f"Already known: {' '.join(known) or '(nothing)'}"]
    for exchange in exchanges:
        lines += [f"User: {exchange['user']}", f"Sofia: {exchange['sofia']}"]
    messages = [{"role": "system", "content": _REFLECT_SYSTEM}]
    for conversation, answer in _REFLECT_EXAMPLES:
        messages += [{"role": "user", "content": conversation}, {"role": "assistant", "content": json.dumps(answer)}]
    raw = ollama.chat(
        BASE_MODEL,
        messages + [{"role": "user", "content": "\n".join(lines)}],
        format=_REFLECT_SCHEMA,
        options={"temperature": 0, "num_predict": 120},
    )
    try:
        observations = json.loads(raw).get("observations", [])
    except json.JSONDecodeError:
        return []
    return [o.strip() for o in observations if isinstance(o, str) and o.strip().startswith("User")]


_DAY_PARTS = ((5, "in the morning"), (12, "in the afternoon"), (17, "in the evening"), (22, "late at night"))


def day_part(moment: datetime) -> str:
    part = "late at night"  # 22:00-04:59
    for start, name in _DAY_PARTS:
        if moment.hour >= start:
            part = name
    return part


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _show(kind: str, item) -> str:
    return f'Example — you: "{item["user"]}" → Sofia: "{item["sofia"]}"' if kind == EXAMPLES else item


class Memory:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self.items: dict[str, list] = {kind: [] for kind in (*KINDS, MOODS)}
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            for kind in self.items:
                wanted = dict if kind in (EXAMPLES, MOODS) else str
                self.items[kind] = [i for i in saved.get(kind, []) if isinstance(i, wanted)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.items, indent=2, ensure_ascii=False))
        temporary.replace(self.path)

    def _known(self, kind: str, item: str) -> bool:
        # Exact match only (ignoring case and punctuation): "User is 24" vs "User is 25"
        # is an update, not a duplicate, and fuzzy matching can't tell those apart.
        return any(_norm(item) == _norm(old) for old in self.items[kind])

    def add(self, kind: str, item: str, replaces: int | None = None) -> bool:
        """Store one item (dropping the oldest past the limit). False if already known."""
        if self._known(kind, item):
            return False
        if replaces is not None:
            del self.items[kind][replaces]
        self.items[kind].append(item)
        del self.items[kind][: -LIMITS[kind]]
        return True

    def add_example(self, user: str, sofia: str, language: str) -> None:
        """Keep a reply the user liked (or the corrected version) as a model of how to talk."""
        examples = self.items[EXAMPLES]
        examples[:] = [e for e in examples if _norm(e["user"]) != _norm(user)]
        examples.append({"user": user, "sofia": sofia, "language": language})
        del examples[: -LIMITS[EXAMPLES]]
        self.save()

    def _add_preference(self, preference: str) -> bool:
        """A reworded or changed rule ("Call the user Sam" -> "Call the user Dev") replaces the old one."""
        for index, old in enumerate(self.items[PREFERENCES]):
            if difflib.SequenceMatcher(None, _norm(preference), _norm(old)).ratio() >= 0.7:
                if _norm(preference) == _norm(old):
                    return False
                return self.add(PREFERENCES, preference, replaces=index)
        return self.add(PREFERENCES, preference)

    def record_mood(self, mood: str, language: str, moment: datetime | None = None) -> None:
        moment = moment or datetime.now()
        self.items[MOODS].append({"time": moment.isoformat(timespec="minutes"), "mood": mood, "language": language})
        del self.items[MOODS][: -LIMITS[MOODS]]

    def habits(self) -> list[str]:
        """Patterns counted (not guessed) from the user's recent messages."""
        recent = self.items[MOODS][-HABIT_WINDOW:]
        days = {m["time"][:10] for m in recent}
        if len(recent) < HABIT_MIN_MESSAGES or len(days) < HABIT_MIN_DAYS:
            return []
        found = []
        part, count = Counter(day_part(datetime.fromisoformat(m["time"])) for m in recent).most_common(1)[0]
        if count / len(recent) >= 0.6:
            found.append(f"User usually chats {part}.")
        language, count = Counter(m["language"] for m in recent).most_common(1)[0]
        if count / len(recent) >= 0.8:
            found.append(f"User mostly writes in {language.capitalize()}.")
        felt = [m["mood"] for m in recent if m["mood"] != NEUTRAL]
        if len(felt) >= HABIT_MIN_MESSAGES // 2:
            mood, count = Counter(felt).most_common(1)[0]
            if count / len(felt) >= 0.4:
                found.append(f"Lately the user has often been {mood}.")
        return found

    def reflect(self, exchanges: list[dict]) -> list[str]:
        """Look for a behaviour pattern in the latest exchanges; store and return what is new."""
        learned = []
        for observation in observe(exchanges, self.items[BEHAVIOUR]):
            replaces = replaced_fact(self.items[BEHAVIOUR], observation)
            if self.add(BEHAVIOUR, observation, replaces):
                learned.append(observation)
        if learned:
            self.save()
        return learned

    def learn(
        self, message: str, language: str, previous: dict | None = None, heard_mood: str = NEUTRAL
    ) -> tuple[list[str], str]:
        """Extract, store and save anything worth remembering.

        `previous` is the exchange before this message ({"user", "sofia", "language"}):
        if the user praises that reply, it is kept as an example. Returns what was
        new, and the user's reaction to that reply (GOOD, BAD or NONE).
        """
        learned = []
        found = extract(message, previous["sofia"] if previous else "")
        for fact in found[FACTS]:
            if not self._known(FACTS, fact) and self.add(FACTS, fact, replaced_fact(self.items[FACTS], fact)):
                learned.append(fact)
        learned += [item for item in found[TAUGHT] if self.add(TAUGHT, item)]
        learned += [item for item in found[PREFERENCES] if self._add_preference(item)]
        # Their words beat the mood heard in their voice: the emotion model often hears "happy"
        # in a flat or disappointed voice.
        self.record_mood(found[MOOD] if found[MOOD] != NEUTRAL else heard_mood, language)
        if previous and found[FEEDBACK] == GOOD:
            self.add_example(**previous)
            learned.append(f'Reply you liked: "{previous["sofia"]}"')
        self.save()
        return learned, found[FEEDBACK]

    def numbered(self) -> list[str]:
        """Everything remembered, as shown to the user; `forget` uses the same numbering."""
        return [_show(kind, item) for kind in KINDS for item in self.items[kind]]

    def forget(self, number: int) -> str | None:
        """Remove item `number` (1-based, as listed by `numbered`); return it."""
        index = number - 1
        for kind in KINDS:
            if 0 <= index < len(self.items[kind]):
                removed = _show(kind, self.items[kind].pop(index))
                self.save()
                return removed
            index -= len(self.items[kind])
        return None

    def clear(self) -> None:
        self.items = {kind: [] for kind in self.items}
        self.save()

    def background(self, situation: list[str] = ()) -> list[dict]:
        """What she knows about the user (and `situation`, the moment around them: see
        sofia/situation.py), as an exchange placed before the conversation. Kept away from the
        newest message on purpose: right next to it, she drags every fact into every reply."""
        parts = []
        if situation:
            parts.append("[Where things stand right now. Bring it up only when it matters.]\n" + " ".join(situation))
        if known := self.items[FACTS] + self.items[TAUGHT]:
            parts.append("[Things you already know. Do not list them; use one only when it really fits.]\n"
                         + " ".join(known))
        if behaviour := self.items[BEHAVIOUR] + self.habits():
            parts.append("[How I tend to behave. Let it shape how you treat me; never say it back to me.]\n"
                         + " ".join(behaviour))
        if not parts:
            return []
        return [
            {"role": "user", "content": "\n".join(parts)},
            {"role": "assistant", "content": "Yes, sir, I remember."},
        ]

    def examples_for(self, message: str, language: str) -> list[dict]:
        """Saved examples that answered a message like this one, the closest last."""
        words = _words(message)

        def similarity(example: dict) -> float:
            other = _words(example["user"])
            return len(words & other) / len(words | other) if words | other else 0.0

        close = [e for e in self.items[EXAMPLES] if e["language"] == language and similarity(e) >= EXAMPLE_MIN_SIMILARITY]
        return sorted(close, key=similarity)[-EXAMPLES_PER_REPLY:]

    def preference_note(self) -> str:
        """Instructions the user gave her, attached to the newest message so they are obeyed every turn."""
        if not self.items[PREFERENCES]:
            return ""
        return "[Rules the user set for you. Obey them silently and never mention them: " + " ".join(self.items[PREFERENCES]) + "]"
