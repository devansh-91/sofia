"""What Sofia can change, in work mode: write a file, make a folder, move, delete (to the Trash) and
run a command. She can only propose these. Each proposal is shown to the user on the page with
exactly what it will do, and runs only when they press Allow there; Deny drops it.

Paths are held to the same rules as her read-only tools (sofia/tools.py): inside the home folder,
nothing hidden. Commands run in a folder of the home folder, never with sudo, for at most a minute.
"""
import itertools
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from sofia import tools
from sofia.tools import _size, _tool, resolve, shown

COMMAND_SECONDS = 60
OUTPUT = 4000  # characters of a command's output she gets back
PREVIEW = 1500  # characters of a file's new content shown on the card
WORKSPACE = Path.home() / "sofia-workspace"  # where she works unless told otherwise
_ROOT_ONLY = ("sudo", "su", "pkexec", "doas")


@dataclass
class Action:
    """One proposed change, waiting for the user."""

    number: int
    name: str
    arguments: dict
    title: str  # one line, e.g. "Write ~/sofia-workspace/app.py"
    detail: str = ""  # what exactly: the new content, the command
    done: bool = field(default=False)

    def card(self) -> dict:
        return {"id": self.number, "title": self.title, "detail": self.detail}


def _write_title(path: Path, content: str) -> str:
    if path.exists():
        return f"Replace {shown(path)} ({_size(path.stat().st_size)} now, {_size(len(content.encode()))} after)"
    return f"Create {shown(path)} ({_size(len(content.encode()))})"


def _folder(raw: str) -> Path:
    """Where a relative path lands: her workspace, so a stray name never litters the home folder."""
    raw = (raw or "").strip()
    if not raw:
        return WORKSPACE
    if raw.startswith(("~", "/")) or raw.split("/")[0].casefold() in {e.name.casefold() for e in tools.HOME.iterdir()}:
        return resolve(raw)
    return resolve(str(WORKSPACE / raw))


def propose(number: int, name: str, arguments: dict) -> Action:
    """Check a proposal and describe it for the card; raises with a reason she can pass on."""
    arguments = {key: str(value) for key, value in arguments.items()}
    if name == "write_file":
        path, content = _folder(arguments.get("path", "")), arguments.get("content", "")
        if path.is_dir():
            raise IsADirectoryError(f"{shown(path)} is a folder")
        detail = content if len(content) <= PREVIEW else content[:PREVIEW] + f"\n... ({len(content) - PREVIEW} more characters)"
        return Action(number, name, {"path": str(path), "content": content}, _write_title(path, content), detail)
    if name == "make_folder":
        path = _folder(arguments.get("path", ""))
        return Action(number, name, {"path": str(path)}, f"Make the folder {shown(path)}")
    if name == "move":
        source, target = _folder(arguments.get("path", "")), _folder(arguments.get("to", ""))
        if not source.exists():
            raise FileNotFoundError(f"{shown(source)} doesn't exist")
        if target.is_dir():
            target = target / source.name
        if target.exists():
            raise FileExistsError(f"{shown(target)} already exists")
        return Action(number, name, {"path": str(source), "to": str(target)}, f"Move {shown(source)} to {shown(target)}")
    if name == "delete":
        path = _folder(arguments.get("path", ""))
        if not path.exists():
            raise FileNotFoundError(f"{shown(path)} doesn't exist")
        if path == tools.HOME:
            raise PermissionError("the home folder itself can't be deleted")
        kind = "folder" if path.is_dir() else f"file, {_size(path.stat().st_size)}"
        return Action(number, name, {"path": str(path)}, f"Move {shown(path)} ({kind}) to the Trash")
    if name == "run_command":
        command, folder = arguments.get("command", "").strip(), _folder(arguments.get("folder", ""))
        if not command:
            raise ValueError("the command is empty")
        if any(word in _ROOT_ONLY for word in command.replace(";", " ").replace("&", " ").replace("|", " ").split()):
            raise PermissionError("commands that need root (sudo) are off limits; the user can run those themselves")
        if not folder.is_dir():
            raise NotADirectoryError(f"{shown(folder)} is not a folder")
        return Action(number, name, {"command": command, "folder": str(folder)}, f"Run a command in {shown(folder)}", command)
    raise ValueError(f"there is no action called {name}")


def run(action: Action) -> str:
    """Do what the user allowed; the result is for her to report."""
    arguments = action.arguments
    if action.name == "write_file":
        path = Path(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"], encoding="utf-8")
        return f"Wrote {shown(path)} ({_size(path.stat().st_size)})"
    if action.name == "make_folder":
        Path(arguments["path"]).mkdir(parents=True, exist_ok=True)
        return f"Made {shown(Path(arguments['path']))}"
    if action.name == "move":
        Path(arguments["to"]).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(arguments["path"], arguments["to"])
        return f"Moved to {shown(Path(arguments['to']))}"
    if action.name == "delete":
        subprocess.run(["gio", "trash", arguments["path"]], check=True, capture_output=True, timeout=30)
        return f"Moved {shown(Path(arguments['path']))} to the Trash"
    if action.name == "run_command":
        try:
            done = subprocess.run(["bash", "-c", arguments["command"]], cwd=arguments["folder"], capture_output=True,
                                  text=True, timeout=COMMAND_SECONDS, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return f"Stopped after {COMMAND_SECONDS} seconds"
        output = (done.stdout + ("\n" + done.stderr if done.stderr else "")).strip() or "(no output)"
        return f"Exit code {done.returncode}\n" + tools._short(output, OUTPUT)
    raise ValueError(action.name)


def _propose_only(name: str):
    def function(**_) -> str:
        raise RuntimeError("actions are proposed, never run directly")
    function.__name__ = name
    return function


SCHEMAS = [
    _tool(_propose_only("write_file"), "Propose to create or overwrite a text file (the user must allow it).",
          path="where; a bare name goes in ~/sofia-workspace", content="the whole new content of the file"),
    _tool(_propose_only("make_folder"), "Propose to make a folder (the user must allow it).", path="the folder"),
    _tool(_propose_only("move"), "Propose to move or rename a file or folder (the user must allow it).",
          path="what to move", to="where to, or the new name"),
    _tool(_propose_only("delete"), "Propose to move a file or folder to the Trash (the user must allow it).", path="what"),
    _tool(_propose_only("run_command"), "Propose to run a shell command, e.g. to run a script or its tests "
          "(the user must allow it; no sudo).", command="the command", **{"folder?": "where to run it, default ~/sofia-workspace"}),
]
NAMES = {schema["function"]["name"] for schema in SCHEMAS}


class Pending:
    """The proposals waiting for the user, numbered across the conversation."""

    def __init__(self):
        self.numbers = itertools.count(1)
        self.waiting: dict[int, Action] = {}

    def add(self, name: str, arguments: dict) -> tuple[Action | None, str]:
        """(the action, what she is told), or (None, why it was refused)."""
        try:
            action = propose(next(self.numbers), name, arguments)
        except Exception as error:
            return None, f"Not proposed: {error}"
        self.waiting[action.number] = action
        return action, (f"Proposed as #{action.number}: {action.title}. Nothing has happened yet: it runs only "
                        "when the user presses Allow. Say in one short sentence what you propose (not that it is "
                        "done), then wait.")

    def decide(self, number: int, allowed: bool) -> tuple[Action, str] | None:
        """Run or drop proposal `number`: (the action, what came of it); None if there is no such one."""
        action = self.waiting.pop(number, None)
        if action is None:
            return None
        if not allowed:
            return action, "The user declined it."
        try:
            return action, run(action)
        except Exception as error:
            return action, f"It failed: {type(error).__name__}: {error}"
