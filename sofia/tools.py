"""Sofia's eyes: tools the model calls (Ollama tool calling) to look through the user's files and
the web. They only look: nothing here changes, moves or deletes a file or runs a program.

Files are reachable inside the home folder only, and never hidden ones (keys, configs, browser
data, her own memory). Web pages are fetched as plain text.

    python3 -m sofia.tools web_search '{"query": "india vs australia score"}'   # try one by hand
"""
import fnmatch
import glob
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

HOME = Path.home()
MAX_OUTPUT = 1200  # characters of a result the model sees: her context is only 8192 tokens
FIND_SECONDS = 5
TOP_PAGE = 900  # characters of the top search result's page that come with the results
BRAVE_REST = 600  # seconds to leave Brave alone after it says "too many requests"
SEARXNG = os.environ.get("SOFIA_SEARXNG", "http://127.0.0.1:8888")
_brave_resting_until = 0.0
BROWSER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
# WMO weather codes, as Open-Meteo reports them.
SKY = {0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers",
       82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail"}


def _short(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + f"\n... ({len(text) - MAX_OUTPUT} more characters)"


def shown(path: Path) -> str:
    """A path the way she says it: ~/Downloads/x.pdf."""
    if path == HOME:
        return "~"
    return "~/" + str(path.relative_to(HOME)) if path.is_relative_to(HOME) else str(path)


def _in_home(relative: Path) -> Path:
    """HOME / relative, matching the first folder loosely ("downloads" finds ~/Downloads)."""
    if not relative.parts or (HOME / relative).exists():
        return HOME / relative
    first = relative.parts[0].casefold()
    for entry in HOME.iterdir():
        if entry.name.casefold() == first:
            return entry.joinpath(*relative.parts[1:])
    return HOME / relative


def resolve(raw: str) -> Path:
    """Where `raw` points; relative paths are in the home folder ("Downloads" is ~/Downloads).
    Hidden places and anything outside the home folder are off limits."""
    raw = raw.strip().strip("'\"") or "~"
    if raw == "~" or raw.startswith("~/"):
        raw = str(HOME) + raw[1:]
    path = Path(raw)
    if not path.is_absolute():
        path = _in_home(path)
    path = Path(os.path.realpath(path))
    if not path.is_relative_to(HOME):
        raise PermissionError(f"{path} is outside the home folder, which is off limits")
    if any(part.startswith(".") for part in path.relative_to(HOME).parts):
        raise PermissionError(f"{shown(path)} is a hidden file or folder, which is off limits")
    return path


def _size(count: float) -> str:
    for unit in ("B", "KB", "MB"):
        if count < 1024:
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} GB"


# --- the tools ---------------------------------------------------------------------------------

def list_files(path: str = "~") -> str:
    folder = resolve(path)
    entries = sorted((e for e in folder.iterdir() if not e.name.startswith(".")), key=lambda e: (e.is_file(), e.name.lower()))
    if not entries:
        return f"{shown(folder)} is empty"
    # A summary first: the list itself gets cut short in a big folder.
    files = [e for e in entries if e.is_file()]
    kinds = Counter(e.suffix.lower().lstrip(".") or "no extension" for e in files)
    summary = f"{shown(folder)}: {len(entries) - len(files)} folders, {len(files)} files"
    if files:
        biggest = max(files, key=lambda e: e.stat().st_size)
        newest = max(files, key=lambda e: e.stat().st_mtime)
        summary += (" (" + ", ".join(f"{count} {kind}" for kind, count in kinds.most_common(8)) + ")"
                    f"\nBiggest: {biggest.name} ({_size(biggest.stat().st_size)})"
                    f"\nNewest: {newest.name} ({datetime.fromtimestamp(newest.stat().st_mtime):%d %b %Y})")
    lines = [f"{e.name}/" if e.is_dir() else f"{e.name} ({_size(e.stat().st_size)})" for e in entries]
    return _short(summary + "\n" + "\n".join(lines))


def find_files(name: str, folder: str = "~") -> str:
    """Files whose name matches `name` (a part of it, or a pattern like *.pdf), below `folder`."""
    pattern = (name if glob.has_magic(name) else f"*{name}*").lower()
    start, found, deadline = resolve(folder), [], time.monotonic() + FIND_SECONDS
    for directory, folders, files in os.walk(start):
        folders[:] = [f for f in folders if not f.startswith(".") and f not in ("node_modules", "__pycache__")]
        found += [Path(directory) / f for f in files + folders if fnmatch.fnmatch(f.lower(), pattern)]
        if time.monotonic() > deadline:
            break
    if not found:
        return f"Nothing named like {name!r} in {shown(start)}"
    cut = time.monotonic() > deadline
    header = f"{len(found)}{'+' if cut else ''} found" + (", the first 30:" if len(found) > 30 else ":")
    return _short(header + "\n" + "\n".join(shown(p) for p in found[:30]))


def read_file(path: str) -> str:
    file = resolve(path)
    data = file.read_bytes()
    if b"\0" in data[:4096]:
        return f"{shown(file)} is not a text file ({_size(len(data))}), so it can't be read here"
    return _short(data.decode("utf-8", errors="replace"))


def _fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": BROWSER_AGENT, "Accept-Language": "en"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.read(2_000_000).decode(response.headers.get_content_charset() or "utf-8", errors="replace")


def _text(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style|noscript|svg|head)\b.*?</\1>", " ", markup)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


def brave_results(page: str, count: int = 5) -> list[str]:
    """'title — url\nwhat it says' for the results on a Brave Search page."""
    results = []
    for block in page.split('data-type="web"')[1:]:
        block = block.split('<div class="snippet ')[0]
        link = re.search(r'<a href="(https?://[^"]+)"', block)
        title = re.search(r'search-snippet-title[^"]*" title="([^"]*)"', block)
        if not (link and title):
            continue
        title = html.unescape(title.group(1))
        about = _text(block).split(title, 1)[-1].strip(" -")
        results.append(f"{title} — {html.unescape(link.group(1))}\n{about[:220]}")
        if len(results) == count:
            break
    return results


def headlines(feed: str, now: datetime, count: int = 8) -> list[str]:
    """'title (5 hours ago)' for the newest items in a Google News RSS feed."""
    items = []
    for item in re.findall(r"<item>(.*?)</item>", feed, re.S):
        title, published = re.search(r"<title>(.*?)</title>", item, re.S), re.search(r"<pubDate>(.*?)</pubDate>", item)
        if title and published:
            items.append((parsedate_to_datetime(published.group(1)), html.unescape(title.group(1)).strip()))
    lines = []
    for moment, title in sorted(items, reverse=True)[:count]:
        minutes = max(0, int((now - moment).total_seconds() // 60))
        age = (f"{minutes} minutes ago" if minutes < 60 else "an hour ago" if minutes < 120
               else f"{minutes // 60} hours ago" if minutes < 48 * 60 else f"{minutes // 1440} days ago")
        lines.append(f"{title} ({age})")
    return lines


def news(topic: str = "") -> str:
    """The newest headlines, about `topic` or overall, from Google News (India edition)."""
    edition = {"hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    url = ("https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": topic, **edition}) if topic
           else "https://news.google.com/rss?" + urllib.parse.urlencode(edition))
    return _short("\n".join(headlines(_fetch(url), datetime.now(timezone.utc)))) or "No news found"


def searxng_results(query: str, count: int = 5) -> list[str]:
    """The same from a SearXNG running on this laptop (scripts/searxng.sh starts one)."""
    data = json.loads(_fetch(f"{SEARXNG}/search?" + urllib.parse.urlencode({"q": query, "format": "json"})))
    return [f"{r['title']} — {r['url']}\n{(r.get('content') or '')[:220]}" for r in data.get("results", [])[:count]]


def _searxng(query: str) -> list[str]:
    try:
        return searxng_results(query)
    except (urllib.error.URLError, ConnectionError, json.JSONDecodeError):
        return []  # not running


def _brave(query: str) -> list[str]:
    global _brave_resting_until
    if time.monotonic() < _brave_resting_until:
        return []
    try:
        return brave_results(_fetch("https://search.brave.com/search?" + urllib.parse.urlencode({"q": query, "source": "web"})))
    except urllib.error.HTTPError as error:
        if error.code != 429:
            raise
        _brave_resting_until = time.monotonic() + BRAVE_REST
        return []


def web_search(query: str) -> str:
    """The top results for `query` with the start of the top page. From the local SearXNG when it
    runs, else Brave Search's web page, which turns scripts away after a few searches in a row.
    Then only news headlines and Wikipedia are left: DuckDuckGo, Mojeek and Ecosia show scripts a
    captcha, and Bing answers them with unrelated results, which is worse than none."""
    results = _searxng(query) or _brave(query)
    if not results:
        wiki = json.loads(_fetch("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": query, "srlimit": 2, "format": "json"})))
        about = [f"{hit['title']}: {_text(hit['snippet'])}" for hit in wiki["query"]["search"]]
        # Headlines rarely carry the number asked for, and she fills the gap from memory unless told not to.
        return _short("Web search is unavailable right now; all you have are these headlines and Wikipedia hits. "
                      "If they don't state the answer, tell the user you couldn't check it just now. Never give a "
                      "number from memory.\n\nHeadlines:\n" + news(query) + "\n\nWikipedia:\n" + "\n".join(about))
    # Snippets seldom hold the number asked for, and the model won't read a page by itself: it
    # searches again, then answers with a price it remembers from training. So read the top one.
    found = "\n\n".join(results[:3])
    for result in results[:2]:
        url = result.split(" — ", 1)[1].split("\n", 1)[0]
        try:
            page = _page_text(url)
        except Exception:  # paywalls, timeouts: try the next one
            continue
        return found[:500] + f"\n\nFrom the top page ({url}):\n{relevant(page, query, TOP_PAGE)}"
    return _short(found)


def weather(place: str) -> str:
    """The weather now, today and tomorrow in `place`, from Open-Meteo (free, no key)."""
    found = json.loads(_fetch("https://geocoding-api.open-meteo.com/v1/search?"
                              + urllib.parse.urlencode({"name": place, "count": 1, "language": "en"})))
    if not found.get("results"):
        return f"No place called {place!r} found"
    spot = found["results"][0]
    query = {"latitude": spot["latitude"], "longitude": spot["longitude"], "timezone": "auto", "forecast_days": 2,
             "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
             "daily": "temperature_2m_min,temperature_2m_max,precipitation_probability_max,weather_code"}
    data = json.loads(_fetch("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(query)))
    now, day = data["current"], data["daily"]
    where = ", ".join(filter(None, (spot["name"], spot.get("admin1"), spot.get("country"))))
    days = [f"{name} {SKY.get(day['weather_code'][i], 'mixed')}, {day['temperature_2m_min'][i]:.0f} to "
            f"{day['temperature_2m_max'][i]:.0f}°C, {day['precipitation_probability_max'][i]}% chance of rain"
            for i, name in enumerate(("Today", "Tomorrow"))]
    return (f"{where}: now {now['temperature_2m']:.0f}°C, feels like {now['apparent_temperature']:.0f}°C, "
            f"{SKY.get(now['weather_code'], 'mixed')}, humidity {now['relative_humidity_2m']}%, "
            f"wind {now['wind_speed_10m']:.0f} km/h. " + ". ".join(days) + ".")


def _page_text(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    markup = _fetch(url)
    # The article itself, without menus and footers, when the page marks it.
    for tag in ("article", "main"):
        if found := re.search(rf"(?is)<{tag}\b.*?</{tag}>", markup):
            markup = found.group(0)
            break
    return _text(re.sub(r"(?is)<(nav|header|footer|aside|form)\b.*?</\1>", " ", markup))


def relevant(text: str, query: str, size: int) -> str:
    """The `size` characters of a page that match `query` best: the most of its words, and
    amounts, since the question is usually a price or a score. Most pages start with menus."""
    words = set(re.findall(r"[a-z]{4,}", query.lower()))
    best, best_score = 0, -1
    for start in range(0, max(1, len(text) - size + 1), size // 3):
        window = text[start:start + size].lower()
        amounts = len(re.findall(r"(?:₹|rs\.?|\$|inr|usd)\s?\d|\d[\d,.]{2,}", window))
        score = sum(word in window for word in words) + 2 * min(3, amounts)
        if score > best_score:
            best, best_score = start, score
    return text[best:best + size]


def read_webpage(url: str) -> str:
    return _short(_page_text(url))


# --- what the model sees -------------------------------------------------------------------------

def _tool(function, description: str, **parameters: str) -> dict:
    required = [name for name in parameters if not name.endswith("?")]
    properties = {name.rstrip("?"): {"type": "string", "description": text} for name, text in parameters.items()}
    return {"type": "function", "function": {
        "name": function.__name__, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


FUNCTIONS = {f.__name__: f for f in (list_files, find_files, read_file, web_search, news, weather, read_webpage)}
SCHEMAS = [
    _tool(list_files, "List what is in one of the user's folders.", **{"path?": "folder, e.g. Downloads or ~/Documents"}),
    _tool(find_files, "Find the user's files by name.",
          name="part of the name, or a pattern like *.pdf", **{"folder?": "where to look, default the home folder"}),
    _tool(read_file, "Read one of the user's text files.", path="the file"),
    _tool(web_search, "Search the web for current information: news, scores, prices, facts.", query="what to search for"),
    _tool(news, "The newest news headlines, overall or about a topic: also for match scores and results.",
          **{"topic?": "e.g. India cricket, gold price; leave out for top headlines"}),
    _tool(weather, "The weather now, today and tomorrow in a city.", place="the city, e.g. Delhi"),
    _tool(read_webpage, "Read the text of a web page.", url="the page's address"),
]


def call(name: str, arguments: dict) -> str:
    """Run one tool call; problems come back as text for the model to tell the user."""
    function = FUNCTIONS.get(name)
    if function is None:
        return f"Error: there is no tool called {name}"
    try:
        return function(**arguments)
    except TypeError as error:
        return f"Error: wrong arguments for {name}: {error}"
    except Exception as error:  # a missing file, a refused path, a dead link: she can say so
        return f"Error: {error}"


def describe(name: str, arguments: dict) -> str:
    """One line for the chat page on what she looked at."""
    return f"{name.replace('_', ' ')}: " + ", ".join(str(value)[:60] for value in arguments.values())


if __name__ == "__main__":
    print(call(sys.argv[1], json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))
