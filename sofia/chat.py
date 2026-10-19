"""Text-only Sofia: chat in the terminal. Write in English or Hinglish; she answers in English.

    python3 -m sofia.chat

Needs the `sofia` Ollama model (see models/Sofia.Modelfile). Stdlib only.
"""
import os

from sofia import ollama
from sofia.lang import ENGLISH, HINGLISH, REPLY_INSTRUCTION, detect
from sofia.log import GOOD, ConversationLog
from sofia.memory import FACTS, MOODS, NEUTRAL, NONE, Memory
from sofia.polish import polish

MODEL = os.environ.get("SOFIA_MODEL", "sofia")

# The Modelfile's persona + examples already use ~1.4k of the 4096-token context,
# and memory (facts + a few learned examples) can add several hundred more.
MAX_HISTORY_MESSAGES = 16

# Every few exchanges she looks back over them for patterns in how the user behaves.
REFLECT_EVERY_TURNS = 6
REFLECT_MIN_TURNS = 4

# While she knows little about the user, she asks something about them every few turns.
CURIOUS_BELOW_FACTS = 12
CURIOUS_EVERY_TURNS = 4
CURIOSITY_NOTE = "[After replying, ask one short casual question to get to know the user better.]"

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
) -> str:
    """Append the user's turn and Sofia's answer to `history`; return the answer.

    `sounds` is the mood heard in the user's voice, when they spoke rather than typed.
    """
    history.append(_user_turn(text))
    del history[:-MAX_HISTORY_MESSAGES]
    heard = f"[The user sounds {sounds}.]" if sounds != NEUTRAL else ""
    messages = history[:-1] + [_user_turn(text, heard)]
    if memory:
        learned = []
        for example in memory.examples_for(text, ENGLISH):
            learned += [_user_turn(example["user"]), {"role": "assistant", "content": example["sofia"]}]
        # Notes ride only on the newest message, so they are never repeated in history.
        newest = _user_turn(text, heard, memory.preference_note(), CURIOSITY_NOTE if curious else "")
        messages = memory.background() + learned + history[:-1] + [newest]
    answer = ollama.chat(MODEL, messages)
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
    answer = polish(answer)
    history.append({"role": "assistant", "content": answer})
    return answer


class Session:
    """One terminal conversation: replies, learns from each message, logs everything."""

    def __init__(self, memory: Memory, log: ConversationLog):
        self.memory = memory
        self.log = log
        self.history: list[dict] = []
        self.language = HINGLISH  # of the user's messages; her replies are always English
        self.turns = 0
        self.last: dict | None = None  # {"user", "sofia", "language" (of her reply)} of the latest exchange
        self.last_id = ""
        self.unreflected: list[dict] = []

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
        answer = reply(self.history, text, self.memory, curious, sounds)
        # learn() keeps a praised reply as an example; the log records praise and criticism.
        learned, reaction = self.memory.learn(text, self.language, previous, heard_mood=sounds)
        if reaction != NONE:
            self.log.feedback(previous_id, reaction)
        self.last = {"user": text, "sofia": answer, "language": ENGLISH}
        self.last_id = self.log.turn(text, answer, self.language)
        self.unreflected.append(self.last)
        if len(self.unreflected) >= REFLECT_EVERY_TURNS:
            learned += self.reflect()
        return answer, learned

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
    session = Session(Memory(), ConversationLog())
    print(f"Sofia — write in English or Hinglish, she answers in English.\n{HELP}\n")
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
        for item in learned:
            print(f"  (remembered: {item})")
        print()
    for item in session.reflect():
        print(f"  (remembered: {item})")


if __name__ == "__main__":
    main()
