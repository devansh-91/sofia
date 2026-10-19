"""Kaam mode: Sofia with a bigger model, for code and real work.

    "kaam mode" (or /kaam)       switch to it
    "normal mode" (or /normal)   back to her usual self

Her own 4B model and her laughing voice leave the GPU to a 7B coder (SOFIA_WORK_MODEL). She then
answers in text with whole code blocks, reads the user's files and the web to get it right, and
says only the words around the code (in her Kokoro voice, on the CPU). Nothing said here goes into
her memory or conversation log: those teach her how to talk, not how to code.

Here she can also change things: write files, make folders, move, delete and run commands, each
only as a proposal the user allows or denies on the page (sofia/actions.py).
"""
import json
import os
import re
from datetime import datetime

from sofia import actions, ollama, tools
from sofia.polish import polish

WORK_MODEL = os.environ.get("SOFIA_WORK_MODEL", "qwen2.5-coder:7b")
CONTEXT = 8192
MAX_HISTORY_MESSAGES = 12
MAX_LOOKUPS = 4
READ_SIZE = 12000  # characters of a file she reads here: source files are read whole
EARLIER_RESULT = 600  # characters kept of a lookup once the next one is in
TEMPERATURE = 0.3  # code and tool calls want a steady hand
SPOKEN = 350  # characters of the words around the code that she says out loud

# The whole message must be the switch, so "what is kaam mode?" stays a question. Speech
# recognition writes "kaam" as "kam" or "come", and may put her name in front.
_COMMAND = r"^\W*(?:(?:hey\W+)?sofia\W+)?(?:{})\W*$"
ENTER = re.compile(_COMMAND.format(r"/kaam|/work|(?:kaam|kam|come|work)\s*mode(?:\s*on)?"), re.I)
LEAVE = re.compile(_COMMAND.format(
    r"/normal|(?:normal|chill|baat)\s*mode|(?:kaam|kam|come|work)\s*mode\s*(?:off|band|khatam)|kaam\s*khatam"), re.I)

SYSTEM = """You are Sofia in work mode: the user's personal assistant, now working as a sharp senior software engineer. Call the user "sir", at most once per reply. Always answer in English, even when the user writes in Hindi or Hinglish.

- Code: give complete, working code in fenced blocks with the language name. No placeholders, no "rest of the code here".
- Words: your words are read aloud and the code is shown on screen, so keep the words short: one or two sentences before the code, and after it only what matters (how to run it, one caveat).
- Files: to see the user's code or files, use your tools (find_files, list_files, read_file) instead of guessing. Paths are in the user's home folder, e.g. ~/projects/app/main.py.
- Unknown errors, libraries or docs: use web_search, then read_webpage.
- Changes: you can write files, make folders, move or delete things and run commands, but only by proposing them with your tools. The user sees each proposal and presses Allow or Deny; nothing happens until then. Work in ~/sofia-workspace unless the user names another place. Don't ask "shall I?" in words: when the user asks for a change, propose it right away, since the Allow button is how they say yes. Propose only what was asked for, one step at a time, and after a proposal stop and wait: the result comes back in brackets.
- After a change is done, say so in a sentence: don't show its code again. Run Python with python3.
- If the request is unclear, ask one short question instead of guessing.

Today is {today}."""

TOOL_NAMES = {schema["function"]["name"] for schema in tools.SCHEMAS + actions.SCHEMAS}

# She says she proposed something, and no proposal was made: the 7B model does that after a result.
_CLAIMS_PROPOSAL = re.compile(r"\bpropos(?:e|ed|ing)\b|\bpress(?:ing)? allow\b|\ballow (?:it|this|that|to)\b", re.I)
CALL_IT = "[You said you proposed it, but you didn't call the tool, so nothing was proposed. Call the tool now.]"

_FENCE = re.compile(r"```[^\n]*\n.*?(?:```|$)", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def spoken(answer: str) -> str:
    """What she says of `answer` out loud: the words around the code, never the code itself."""
    words = polish(_FENCE.sub(" ", answer).replace("_", " "))  # fit_sounds is said "fit sounds"
    if len(words) > SPOKEN:
        cut = ""
        for sentence in _SENTENCE_END.split(words):
            if len(cut) + len(sentence) > SPOKEN:
                break
            cut = f"{cut} {sentence}".strip()
        words = cut or words[:SPOKEN]
    return words or "Here you go, sir."


_WRAPPING = re.compile(r"</?tool_call>|```(?:json)?")


_CALL_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```|<tool_call>(.*?)</tool_call>", re.S)


def text_calls(content: str, known: set[str]) -> list[dict]:
    """Tool calls written out as text: Qwen2.5-Coder often answers {"name": ..., "arguments": ...}
    in the reply itself instead of as a tool call, alone or in a block after a sentence or two."""
    calls = _bare_calls(content, known)
    for block in ([] if calls else _CALL_BLOCK.finditer(content)):
        calls += _bare_calls(block.group(1) or block.group(2) or "", known)
    return calls


def _bare_calls(content: str, known: set[str]) -> list[dict]:
    """The calls in `content` when it is nothing but calls, else none."""
    text, decoder, position, calls = _WRAPPING.sub(" ", content), json.JSONDecoder(), 0, []
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            return calls
        try:
            found, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return []
        for call in found if isinstance(found, list) else [found]:
            if not (isinstance(call, dict) and call.get("name") in known and isinstance(call.get("arguments", {}), dict)):
                return []
            calls.append({"function": {"name": call["name"], "arguments": call.get("arguments") or {}}})


def _call(name: str, arguments: dict) -> str:
    if name == "read_file":  # whole source files, not the chat's short excerpt
        try:
            return tools.read_file(str(arguments.get("path", "")), size=READ_SIZE)
        except Exception as error:
            return f"Error: {error}"
    return tools.call(name, arguments)


class Work:
    """One work-mode conversation."""

    def __init__(self):
        self.history: list[dict] = []
        self.looked: list[str] = []  # what she looked at for the latest message
        self.pending = actions.Pending()
        self.proposed: list[dict] = []  # cards for what she proposed in the latest message

    def decide(self, number: int, allowed: bool) -> tuple[str, str] | None:
        """Run or drop proposal `number` as the user chose; (what came of it, her answer to that)."""
        decided = self.pending.decide(number, allowed)
        if decided is None:
            return None
        action, result = decided
        if not allowed:
            return result, self.answer(f"[The user declined #{number} ({action.title}).]")
        answer = self.answer(f"[The user allowed #{number} ({action.title}). What happened, word for word:\n"
                             f"{result}\nTell the user in a sentence.]")
        # The 7B model retells output, and the file it just wrote, from memory and gets them wrong:
        # her code blocks go, and a command's real output is shown instead.
        answer = _FENCE.sub("", answer).strip() or "Done, sir."
        if action.name == "run_command":
            answer += f"\n\n```\n{result}\n```"
        self.history[-1]["content"] = answer
        return result, answer

    def answer(self, text: str) -> str:
        """Her answer to `text`, as Markdown with code blocks."""
        self.looked, self.proposed = [], []
        self.history.append({"role": "user", "content": text})
        del self.history[:-MAX_HISTORY_MESSAGES]
        system = SYSTEM.format(today=datetime.now().strftime("%A, %d %B %Y"))
        messages = [{"role": "system", "content": system}, *self.history]
        options = {"num_ctx": CONTEXT, "temperature": TEMPERATURE}
        answer, nudged = "", False
        for _ in range(MAX_LOOKUPS):
            message = ollama.chat_message(WORK_MODEL, messages, tools=tools.SCHEMAS + actions.SCHEMAS, options=options)
            calls = message.get("tool_calls") or text_calls(message.get("content", ""), TOOL_NAMES)
            if not calls:
                answer = message.get("content", "").strip()
                waiting = self.proposed or self.pending.waiting
                if not waiting and not nudged and _CLAIMS_PROPOSAL.search(answer):
                    nudged = True
                    messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": CALL_IT}]
                    continue
                break
            messages = [{**m, "content": m["content"][:EARLIER_RESULT]} if m["role"] == "tool" else m for m in messages]
            messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            for tool_call in calls:
                name, arguments = tool_call["function"]["name"], tool_call["function"].get("arguments") or {}
                if name in actions.NAMES:
                    action, told = self.pending.add(name, arguments)
                    if action:
                        self.proposed.append(action.card())
                else:
                    self.looked.append(tools.describe(name, arguments))
                    told = _call(name, arguments)
                messages.append({"role": "tool", "tool_name": name, "content": told})
        else:
            answer = ollama.chat(WORK_MODEL, messages + [
                {"role": "user", "content": "[No more lookups. Answer now with what you found.]"}], options=options)
        answer = answer or "Sorry, sir, I came up empty on that one."
        self.history.append({"role": "assistant", "content": answer})
        return answer
