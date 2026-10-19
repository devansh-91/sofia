"""The user's documents, for Sofia to answer from: notes, PDFs, Word files, code.

Put files in ~/sofia-docs (or add them on the settings page). Each is cut into overlapping
passages, and each passage gets an embedding from nomic-embed-text, on the CPU so the GPU stays
with her model and voice. `search` returns the passages closest in meaning to a question; her
search_docs tool (sofia/tools.py) passes them on. Everything stays on this laptop.

The index is kept in ~/.local/share/sofia/docs-index.json and follows the folder: a new or changed
file is read again, a deleted one is dropped.
"""
import json
import os
import re
import threading
import zipfile
from pathlib import Path

from sofia import ollama

FOLDER = Path(os.environ.get("SOFIA_DOCS", "~/sofia-docs")).expanduser()
INDEX = Path("~/.local/share/sofia/docs-index.json").expanduser()
MODEL = os.environ.get("SOFIA_EMBED_MODEL", "nomic-embed-text")
PASSAGE = 900  # characters
OVERLAP = 150
TOP = 4
MAX_BYTES = 25 * 1024 * 1024
TEXT = {".txt", ".md", ".markdown", ".csv", ".json", ".py", ".js", ".ts", ".html", ".css", ".java", ".c", ".cpp",
        ".h", ".go", ".rs", ".sh", ".yaml", ".yml", ".toml", ".ini", ".sql", ".tex", ".rst", ".log"}
KINDS = TEXT | {".pdf", ".docx"}

_lock = threading.Lock()


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if path.suffix.lower() == ".docx":
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="replace")
        return re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))
    return path.read_text(encoding="utf-8", errors="replace")


def passages(text: str) -> list[str]:
    """`text` in overlapping passages of about PASSAGE characters, cut at a line or word end."""
    text = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()
    found, start = [], 0
    while start < len(text):
        end = min(len(text), start + PASSAGE)
        if end < len(text):
            cut = max(text.rfind("\n", start + PASSAGE // 2, end), text.rfind(" ", start + PASSAGE // 2, end))
            end = cut if cut > start else end
        found.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - OVERLAP, start + 1)
    return [passage for passage in found if passage]


def _embed(texts: list[str], kind: str) -> list[list[float]]:
    """nomic-embed-text wants the task in front: "search_document: " or "search_query: "."""
    vectors = []
    for start in range(0, len(texts), 32):
        batch = [f"{kind}: {text}" for text in texts[start:start + 32]]
        vectors += ollama._post("/api/embed", {"model": MODEL, "input": batch, "options": {"num_gpu": 0},
                                               "keep_alive": "10m"}, timeout=600)["embeddings"]
    return vectors


def files() -> list[Path]:
    if not FOLDER.exists():
        return []
    return sorted(p for p in FOLDER.rglob("*") if p.is_file() and p.suffix.lower() in KINDS
                  and not any(part.startswith(".") for part in p.relative_to(FOLDER).parts)
                  and p.stat().st_size <= MAX_BYTES)


def _load_index() -> dict:
    try:
        return json.loads(INDEX.read_text())
    except (OSError, ValueError):
        return {}


def refresh() -> dict:
    """Bring the index in line with the folder; return it."""
    with _lock:
        index = _load_index()
        present = {str(path): path for path in files()}
        changed = False
        for name in list(index):
            if name not in present:
                del index[name]
                changed = True
        for name, path in present.items():
            stamp = path.stat().st_mtime
            if index.get(name, {}).get("mtime") == stamp:
                continue
            try:
                chunks = passages(_read(path))
            except Exception as error:  # a broken PDF: skip it, say why on the settings page
                index[name] = {"mtime": stamp, "error": f"{type(error).__name__}: {error}", "passages": []}
                changed = True
                continue
            vectors = _embed(chunks, "search_document") if chunks else []
            index[name] = {"mtime": stamp, "passages": [{"text": t, "vector": v} for t, v in zip(chunks, vectors)]}
            changed = True
        if changed:
            INDEX.parent.mkdir(parents=True, exist_ok=True)
            temporary = INDEX.with_suffix(".tmp")
            temporary.write_text(json.dumps(index))
            temporary.replace(INDEX)
        return index


def refresh_soon() -> None:
    """Index new files in the background (embedding a big PDF takes a while on the CPU)."""
    threading.Thread(target=_refresh_quietly, daemon=True).start()


def _refresh_quietly() -> None:
    try:
        refresh()
    except OSError as error:  # Ollama is down or the model isn't pulled: searching says so later
        print(f"(Couldn't index the documents: {error})", flush=True)


def search(query: str, top: int = TOP) -> list[dict]:
    """The passages closest to `query`: [{"file", "text", "score"}], best first."""
    import numpy

    index = refresh()
    rows = [(name, passage) for name, entry in index.items() for passage in entry.get("passages", [])]
    if not rows:
        return []
    matrix = numpy.array([passage["vector"] for _, passage in rows], dtype=numpy.float32)
    question = numpy.array(_embed([query], "search_query")[0], dtype=numpy.float32)
    scores = matrix @ question / (numpy.linalg.norm(matrix, axis=1) * numpy.linalg.norm(question) + 1e-9)
    best = numpy.argsort(-scores)[:top]
    return [{"file": rows[i][0], "text": rows[i][1]["text"], "score": float(scores[i])} for i in best]


def listing() -> list[dict]:
    """The documents and their state, for the settings page."""
    index = _load_index()
    found = []
    for path in files():
        entry = index.get(str(path), {})
        state = entry.get("error") or (f"{len(entry['passages'])} passages" if entry.get("mtime") == path.stat().st_mtime
                                       else "being read…")
        found.append({"name": str(path.relative_to(FOLDER)), "size": path.stat().st_size, "state": state})
    return found


def add(name: str, data: bytes) -> Path:
    """Save an uploaded file into the folder (a plain name only) and index it in the background."""
    name = Path(name).name
    if not name or name.startswith(".") or Path(name).suffix.lower() not in KINDS:
        raise ValueError(f"can't read {name or 'that file'}: use PDF, Word (.docx), text, Markdown or code")
    FOLDER.mkdir(parents=True, exist_ok=True)
    target = FOLDER / name
    target.write_bytes(data)
    refresh_soon()
    return target


def remove(name: str) -> None:
    target = (FOLDER / name).resolve()
    if not target.is_relative_to(FOLDER.resolve()) or not target.is_file():
        raise FileNotFoundError(name)
    target.unlink()
    refresh_soon()
