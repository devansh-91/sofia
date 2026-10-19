"""Work mode's agent: OpenCode doing a whole coding task, with the user's Allow for every change.

    "agent, make a todo app in python and test it"       (in work mode)

OpenCode (https://opencode.ai, MIT) is a coding agent: it reads the project, edits files and runs
commands until the task is done. Sofia runs it headless (`opencode serve`) in ~/sofia-workspace,
on a local model, so nothing leaves the laptop. Every edit and every command it wants is held by
OpenCode as a permission request; Sofia shows each one as a card with the diff or the command,
and answers it only when the user presses Allow or Deny.

The model is Sofia's own weights (sofia-base, Qwen3.5-4B) with a 16k context for OpenCode's long
prompt and thinking off: Qwen2.5-Coder writes its tool calls as text, which OpenCode can't run,
and Qwen3.5 with thinking on spent five minutes thinking about one file.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from sofia import ollama

OPENCODE = os.environ.get("SOFIA_OPENCODE") or shutil.which("opencode") or str(Path.home() / ".opencode/bin/opencode")
PORT = int(os.environ.get("SOFIA_OPENCODE_PORT", "4096"))
URL = f"http://127.0.0.1:{PORT}"
WORKSPACE = Path.home() / "sofia-workspace"
LOG = Path("~/.local/state/sofia/opencode.log").expanduser()
MODEL = "kaam-agent"
CONTEXT = 16384  # OpenCode's own instructions and tools are ~10k tokens
WAIT = 300  # seconds she waits for the agent before telling the user it's still at it
DIFF = 2500  # characters of a diff shown on the card

CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {"ollama": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Ollama (this laptop)",
        "options": {"baseURL": f"{ollama.OLLAMA_HOST}/v1"},
        "models": {MODEL: {"name": "Sofia's Qwen3.5-4B, 16k", "tools": True, "options": {"reasoningEffort": "none"}}},
    }},
    "model": f"ollama/{MODEL}",
    "small_model": f"ollama/{MODEL}",
    # Every change waits for the user; reading the project and the web does not.
    "permission": {"edit": "ask", "bash": "ask", "webfetch": "allow", "external_directory": "ask"},
    "share": "disabled",
    "autoupdate": False,
}


def available() -> bool:
    return Path(OPENCODE).exists()


def _request(method: str, path: str, body: dict | None = None, timeout: float = 30):
    request = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                     method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    return json.loads(data) if data else None


def _ensure_model() -> None:
    """kaam-agent: sofia-base's weights with the context OpenCode needs (no extra download)."""
    try:
        ollama._post("/api/show", {"model": MODEL}, timeout=10)
    except urllib.error.HTTPError:
        ollama._post("/api/create", {"model": MODEL, "from": ollama.BASE_MODEL, "stream": False,
                                     "parameters": {"num_ctx": CONTEXT, "temperature": 0.6, "top_p": 0.95, "top_k": 20}})


class Agent:
    """One OpenCode server and one session, for as long as work mode lasts."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.session = ""
        self.cards: dict[str, str] = {}  # card id -> OpenCode permission id
        self.numbers = 0
        self.since = 0  # messages in the session before the current task

    def _start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        _ensure_model()
        WORKSPACE.mkdir(exist_ok=True)
        (WORKSPACE / "opencode.json").write_text(json.dumps(CONFIG, indent=2))
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("ab") as log:
            self.process = subprocess.Popen([OPENCODE, "serve", "--port", str(PORT), "--hostname", "127.0.0.1"],
                                            cwd=WORKSPACE, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        for _ in range(60):
            try:
                _request("GET", "/session")
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError(f"OpenCode didn't start (see {LOG})")
        self.session = _request("POST", "/session", {"title": "Sofia"})["id"]

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process, self.session, self.cards = None, "", {}

    def run(self, task: str) -> tuple[str, list[dict]]:
        """Give the agent `task`; (what it says, cards for what it wants to change) once it is done
        or waits for the user."""
        self._start()
        self.since = len(_request("GET", f"/session/{self.session}/message") or [])
        _request("POST", f"/session/{self.session}/prompt_async", {"parts": [{"type": "text", "text": task}]})
        return self._until_needed()

    def decide(self, card: str, allowed: bool) -> tuple[str, list[dict]] | None:
        """The user's answer to one of the agent's requests; then on until it needs them again."""
        permission = self.cards.pop(card, None)
        if permission is None:
            return None
        _request("POST", f"/permission/{permission}/reply", {"reply": "once" if allowed else "reject"})
        return self._until_needed()

    def _until_needed(self) -> tuple[str, list[dict]]:
        started = time.monotonic()
        time.sleep(1)  # the prompt or the reply takes a moment to register as busy
        while time.monotonic() - started < WAIT:
            waiting = [p for p in _request("GET", "/permission") or [] if p.get("sessionID") == self.session]
            fresh = [p for p in waiting if p["id"] not in self.cards.values()]
            if fresh:
                return self._latest_text(), [self._card(p) for p in fresh]
            status = (_request("GET", "/session/status") or {}).get(self.session, {}).get("type", "idle")
            if status == "idle" and not waiting:
                return self._latest_text() or "Done, sir.", []
            time.sleep(0.5)
        return "Still working on it, sir. Ask me again in a bit.", []

    def _card(self, permission: dict) -> dict:
        self.numbers += 1
        card = f"agent-{self.numbers}"
        self.cards[card] = permission["id"]
        meta = permission.get("metadata") or {}
        if permission["permission"] == "edit":
            path = meta.get("filepath") or ", ".join(permission.get("patterns") or [])
            title, detail = f"Agent: edit {_home(path)}", meta.get("diff", "")
        elif permission["permission"] == "bash":
            title, detail = "Agent: run a command", meta.get("command") or " ".join(permission.get("patterns") or [])
        else:
            title, detail = f"Agent: {permission['permission']}", json.dumps(permission.get("patterns"))
        detail = detail if len(detail) <= DIFF else detail[:DIFF] + f"\n... ({len(detail) - DIFF} more characters)"
        return {"id": card, "title": title, "detail": detail}

    def _latest_text(self) -> str:
        messages = (_request("GET", f"/session/{self.session}/message") or [])[self.since:]
        for message in reversed(messages):
            if message.get("info", {}).get("role") == "assistant":
                text = " ".join(p.get("text", "") for p in message.get("parts", []) if p.get("type") == "text").strip()
                if text:
                    return text
        return ""


def _home(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home) else path
