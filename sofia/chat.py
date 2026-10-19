"""Text-only Sofia: chat in the terminal. Write in English or Hinglish; she answers in English.

    python3 -m sofia.chat

Needs the `sofia` Ollama model (see models/Sofia.Modelfile). Stdlib only.
"""
import os
import re
from datetime import datetime, timedelta
from typing import Callable

from sofia import ollama, tools
from sofia.lang import ENGLISH, HINGLISH, REPLY_INSTRUCTION, detect
from sofia.log import GOOD, ConversationLog
from sofia.memory import FACTS, MOODS, NEUTRAL, NONE, Memory
from sofia.polish import fit_sounds, polish
from sofia.situation import LAST_TIME_TURNS, Situation

MODEL = os.environ.get("SOFIA_MODEL", "sofia")

# The Modelfile's persona, examples and tools already use ~2.7k of the 8192-token context, memory
# (facts + a few learned examples) can add several hundred more, and lookups need room too.
MAX_HISTORY_MESSAGES = 16

# Every few exchanges she looks back over them for patterns in how the user behaves.
REFLECT_EVERY_TURNS = 6
REFLECT_MIN_TURNS = 4

# While she knows little about the user, she asks something about them every few turns.
CURIOUS_BELOW_FACTS = 12
CURIOUS_EVERY_TURNS = 4
CURIOSITY_NOTE = "[After replying, ask one short casual question to get to know the user better.]"

# When the user opens the app, she speaks first; reopening it within a few minutes gets no new hello.
GREETING = ("[The user has just opened the app and not said anything yet. Say hello first, in one short "
            "line that fits the moment. If something from your last conversation is worth asking about, ask.]")
GREET_AFTER = timedelta(minutes=10)

# Lookups (files, web) she may chain for one message before she answers with what she found.
MAX_LOOKUPS = 4
EARLIER_RESULT = 300  # characters kept of a lookup once the next one is in
ENOUGH_LOOKING = "[No more lookups. Answer now from what you found, or say plainly that you couldn't find it.]"
LOOK_FIRST = "[You answered without looking it up. Use your tools first.]"
UNFOUND = ("[These numbers are not in what you looked up: {}. Answer again using only what you found, "
           "or say plainly that you couldn't check it. No numbers from memory.]")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Questions about what is going on in the world: left to itself the 4B model makes up a score
# or a forecast instead of looking it up.
CURRENT = re.compile(
    r"\b(score|match|news|khabar|headlines?|weather|mausam|temperature|forecast|baarish|rain|price|rate|daam|"
    r"stock|shares?|sensex|nifty|bitcoin|crypto|petrol|gold|result|election|latest|trending|release|won|jeeta)\b",
    re.I,
)
LOOK_IT_UP = ("[This needs current information: look it up with your tools before answering (news for news, "
              "scores and results; weather for the weather). If the results don't answer it, try other words. "
              "Say only what the results say, with no added details. Never guess.]")
# Questions about the user's files, which it otherwise answers with "let me check" and no check.
FILES = re.compile(r"\b(folders?|files?|downloads|documents|desktop|pictures|photos|pdfs?|resume|cv)\b", re.I)
LOOK_IN_FILES = "[Look in the user's files with your tools before answering. Never guess.]"

HELP = (
    "Commands: /good (last reply was good), /say <better reply> (teach the right reply), "
    "/memory, /forget <number>, /forget all, bye"
)


def _user_turn(text: str, *notes: str) -> dict:
    lines = [text, *notes, REPLY_INSTRUCTION]
    return {"role": "user", "content": "\n".join(line for line in lines if line)}


def _in_english(hinglish: str) -> str:
    """Last resort when she keeps answering in Hinglish: translate it."""
    system = ("Translate the Hinglish line into casual spoken English. Keep the meaning, tone and any "
              "swearing. Output only the translation.")
    return ollama.chat(
        ollama.BASE_MODEL,
        [{"role": "system", "content": system}, {"role": "user", "content": hinglish}],
        options={"temperature": 0, "num_predict": 160},
    )


def reply(
    history: list[dict],
    text: str,
    memory: Memory | None = None,
    curious: bool = False,
    sounds: str = NEUTRAL,
    situation: Situation | None = None,
    looked: list[str] | None = None,
) -> str:
    """Append the user's turn and Sofia's answer to `history`; return the answer.

    `sounds` is the mood heard in the user's voice, when they spoke rather than typed. With
    `looked`, she can look things up in the user's files and on the web, and each lookup is
    added to it.
    """
    history.append(_user_turn(text))
    del history[:-MAX_HISTORY_MESSAGES]
    heard = f"[The user sounds {sounds}.]" if sounds != NEUTRAL else ""
    # Notes ride only on the newest message, so they are never repeated in history.
    notes = situation.notes() if situation else []
    if looked is not None and CURRENT.search(text):
        notes.append(LOOK_IT_UP)
    if looked is not None and FILES.search(text):
        notes.append(LOOK_IN_FILES)
    messages = history[:-1] + [_user_turn(text, heard, *notes)]
    if memory:
        learned = []
        for example in memory.examples_for(text, ENGLISH):
            learned += [_user_turn(example["user"]), {"role": "assistant", "content": example["sofia"]}]
        newest = _user_turn(text, heard, *notes, memory.preference_note(), CURIOSITY_NOTE if curious else "")
        messages = memory.background(situation.background() if situation else []) + learned + history[:-1] + [newest]
    answer = _answer(messages, looked, LOOK_IT_UP in notes or LOOK_IN_FILES in notes)
    history.append({"role": "assistant", "content": answer})
    return answer


def _ungrounded(answer: str, found: str) -> list[str]:
    """Numbers in `answer` that appear nowhere in what she looked up."""
    seen = {number.replace(",", "") for number in NUMBER.findall(found)}
    return [number for number in NUMBER.findall(answer) if number.replace(",", "") not in seen]


def _look(messages: list[dict], looked: list[str], must_look: bool = False) -> str:
    """Let her look things up with her tools until she answers. With `must_look` (a question about
    the news, prices or the user's files) she is held to looking first and to the numbers she found."""
    found = []
    for step in range(MAX_LOOKUPS):
        message = ollama.chat_message(MODEL, messages, tools=tools.SCHEMAS)
        calls = message.get("tool_calls") or []
        if not calls and must_look and not found and step == 0:
            messages = messages + [message, _user_turn(LOOK_FIRST)]  # she answered without looking
            continue
        if not calls:
            return _grounded(message.get("content", "").strip(), found, messages, must_look)
        # Earlier results shrink to their start, so a chain of lookups can't fill her context.
        messages = [{**m, "content": m["content"][:EARLIER_RESULT]} if m["role"] == "tool" else m for m in messages]
        messages.append(message)
        for tool_call in calls:
            name, arguments = tool_call["function"]["name"], tool_call["function"].get("arguments") or {}
            looked.append(tools.describe(name, arguments))
            found.append(tools.call(name, arguments))
            messages.append({"role": "tool", "tool_name": name, "content": found[-1]})
    # Out of lookups: answer with what she found. Without tools offered she sometimes still writes
    # a tool call as text, so that never reaches the user.
    answer = ollama.chat(MODEL, messages + [_user_turn(ENOUGH_LOOKING)])
    if "<tool" in answer or "<function" in answer:
        return "Sorry, sir, I couldn't find that one."
    return _grounded(answer, found, messages, must_look)


def _grounded(answer: str, found: list[str], messages: list[dict], must_look: bool) -> str:
    """The 4B model fills gaps with prices and scores from its training. A number she didn't find
    gets her one more try, told which numbers those are."""
    missing = _ungrounded(answer, "\n".join(found)) if found or must_look else []
    if not missing:
        return answer
    retry = [{"role": "assistant", "content": answer}, _user_turn(UNFOUND.format(", ".join(missing)))]
    return ollama.chat(MODEL, messages + retry)


def _answer(messages: list[dict], looked: list[str] | None = None, must_look: bool = False) -> str:
    answer = _look(messages, looked, must_look) if looked is not None else ollama.chat(MODEL, messages)
    # The model slips into Hinglish now and then, mostly when the user wrote in it.
    # Show it the slip and ask again — a plain retry repeats it.
    for _ in range(2):
        if detect(answer, ENGLISH) != HINGLISH:
            break
        correction = [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "[That was Hindi. Say the same thing in English only.]"},
        ]
        answer = ollama.chat(MODEL, messages + correction)
    if detect(answer, ENGLISH) == HINGLISH:
        answer = _in_english(answer)
    return polish(answer)


class Session:
    """One terminal conversation: replies, learns from each message, logs everything."""

    def __init__(self, memory: Memory, log: ConversationLog, now: Callable[[], datetime] = datetime.now):
        self.memory = memory
        self.log = log
        self.situation = Situation(log.last_turns(LAST_TIME_TURNS), now)
        self.history: list[dict] = []
        self.language = HINGLISH  # of the user's messages; her replies are always English
        self.turns = 0
        self.last: dict | None = None  # {"user", "sofia", "language" (of her reply)} of the latest exchange
        self.last_id = ""
        self.unreflected: list[dict] = []
        self.looked: list[str] = []  # what she looked up for the latest message

    @property
    def mood(self) -> str:
        """The user's mood in their latest message (heard in their voice, else read from the text)."""
        moods = self.memory.items[MOODS]
        return moods[-1]["mood"] if moods else NEUTRAL

    def say(self, text: str, sounds: str = NEUTRAL) -> tuple[str, list[str]]:
        """Answer `text`; return (answer, things she learned from it).

        `sounds` is the mood heard in the user's voice, when they spoke rather than typed.
        """
        self.language = detect(text, previous=self.language)
        self.turns += 1
        curious = (
            len(self.memory.items[FACTS]) < CURIOUS_BELOW_FACTS and self.turns % CURIOUS_EVERY_TURNS == 2
        )
        previous, previous_id = self.last, self.last_id
        self.looked = []
        answer = reply(self.history, text, self.memory, curious, sounds, self.situation, self.looked)
        # learn() keeps a praised reply as an example; the log records praise and criticism.
        learned, reaction = self.memory.learn(text, self.language, previous, heard_mood=sounds)
        # learn() has just read the user's mood: drop a sound that doesn't fit it.
        answer = self.history[-1]["content"] = fit_sounds(answer, self.mood)
        if reaction != NONE:
            self.log.feedback(previous_id, reaction)
        self.last = {"user": text, "sofia": answer, "language": ENGLISH}
        self.last_id = self.log.turn(text, answer, self.language)
        self.situation.seen()
        self.unreflected.append(self.last)
        if len(self.unreflected) >= REFLECT_EVERY_TURNS:
            learned += self.reflect()
        return answer, learned

    def greet(self) -> str | None:
        """Her opening line when the user opens the app, fitted to the moment; None when they
        were here a moment ago."""
        seen = self.situation.last_seen
        if seen and self.situation.now() - seen < GREET_AFTER:
            return None
        notes = self.situation.notes()
        messages = self.memory.background(self.situation.background()) + self.history + [_user_turn(GREETING, *notes)]
        answer = fit_sounds(_answer(messages), NEUTRAL)
        self.history += [{"role": "user", "content": "[The user opened the app.]"},
                         {"role": "assistant", "content": answer}]
        self.situation.seen()
        return answer

    def reflect(self) -> list[str]:
        """Learn from how the user behaved over the exchanges since the last reflection."""
        exchanges, self.unreflected = self.unreflected, []
        return self.memory.reflect(exchanges) if len(exchanges) >= REFLECT_MIN_TURNS else []

    def mark_good(self) -> str:
        if not self.last:
            return "nothing to mark yet"
        self.memory.add_example(**self.last)
        self.log.feedback(self.last_id, GOOD)
        return "saved that reply as an example of how to talk"

    def correct(self, better: str) -> str:
        if not self.last:
            return "nothing to correct yet"
        self.last["sofia"] = better
        self.history[-1]["content"] = better
        self.memory.add_example(**self.last)
        self.log.correction(self.last_id, better)
        return "learned: next time I'll answer that way"

    def command(self, text: str) -> str | None:
        """Run a /command and return what to print; None if `text` isn't one."""
        if not text.startswith("/"):
            return None
        name, _, argument = text[1:].partition(" ")
        argument = argument.strip()
        if name == "good":
            return self.mark_good()
        if name == "say" and argument:
            return self.correct(argument)
        if name == "memory":
            lines = [f"{i}. {item}" for i, item in enumerate(self.memory.numbered(), 1)]
            lines += [f"-  {habit} (counted from your messages)" for habit in self.memory.habits()]
            return "\n  ".join(lines) or "(nothing yet)"
        if name == "forget" and argument == "all":
            self.memory.clear()
            return "forgot everything"
        if name == "forget" and argument.isdigit():
            removed = self.memory.forget(int(argument))
            return f"forgot: {removed}" if removed else "no such number — see /memory"
        return HELP


def main() -> None:
    ollama.preload(MODEL)
    session = Session(Memory(), ConversationLog())
    print(f"Sofia — write in English or Hinglish, she answers in English.\n{HELP}\n")
    if greeting := session.greet():
        print(f"sofia> {greeting}\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"bye", "/bye", "exit", "quit"}:
            break
        result = session.command(text)
        if result is not None:
            print(f"  {result}\n")
            continue
        answer, learned = session.say(text)
        print(f"sofia> {answer}")
        for item in session.looked:
            print(f"  (looked up: {item})")
        for item in learned:
            print(f"  (remembered: {item})")
        print()
    for item in session.reflect():
        print(f"  (remembered: {item})")
    ollama.unload(MODEL)


if __name__ == "__main__":
    main()
