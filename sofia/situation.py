"""What Sofia notices about the moment: the time, how long the user was away, where they are
talking from, the battery of the laptop she runs on, and what they talked about last time.

Two kinds of output, kept apart on purpose (see Memory.background):

- `background()`: plain facts placed before the conversation, for her to use when they fit
  ("what time is it", "good morning", asking how yesterday's presentation went).
- `notes()`: what is worth reacting to now, on the newest message, each said only once:
  the user is back after hours or days, it is the middle of the night, the battery is dying.
"""
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from sofia.memory import day_part

AWAY = timedelta(hours=3)  # a shorter break is still the same conversation
NIGHT_HOURS = range(0, 5)  # when "why are you still up?" fits
LOW_BATTERY = 15  # percent
# The previous conversation stays in view for the first few exchanges of a new one, if recent.
LAST_TIME_TURNS = 3
LAST_TIME_WITHIN = timedelta(days=7)
QUOTE_CHARS = 160

# Days with a fixed date. The ones that move (Diwali, Holi, Eid) would need a calendar.
OCCASIONS = {
    (1, 1): "New Year's Day", (1, 26): "Republic Day", (2, 14): "Valentine's Day",
    (8, 15): "Independence Day", (10, 31): "Halloween", (12, 25): "Christmas", (12, 31): "New Year's Eve",
}

POWER_SUPPLY = Path("/sys/class/power_supply")


def ago(gap: timedelta) -> str:
    """A gap the way people say it: '4 days', 'a day', '5 hours', 'an hour'."""
    if gap >= timedelta(hours=20):
        days = round(gap / timedelta(days=1))
        return "a day" if days <= 1 else f"{days} days"
    hours = round(gap / timedelta(hours=1))
    return f"{hours} hours" if hours >= 2 else "an hour"


def time_of_day(moment: datetime) -> str:
    return f"{moment.hour % 12 or 12}:{moment:%M} {'AM' if moment.hour < 12 else 'PM'}"


def clock(moment: datetime) -> str:
    occasion = OCCASIONS.get((moment.month, moment.day))
    said = f"It is {moment:%A}, {moment.day} {moment:%B}, {time_of_day(moment)}, {day_part(moment)}"
    return said + (f", and it's {occasion}." if occasion else ".")


def battery() -> tuple[int, bool] | None:
    """(percent, plugged in) for the laptop's battery; None when it has none."""
    for supply in sorted(POWER_SUPPLY.glob("BAT*")):
        try:
            percent = int((supply / "capacity").read_text())
            status = (supply / "status").read_text().strip()
        except (OSError, ValueError):
            continue
        return percent, status != "Discharging"
    return None


def _day(moment: datetime, now: datetime) -> str:
    days = (now.date() - moment.date()).days
    return "earlier today" if days == 0 else "yesterday" if days == 1 else f"on {moment:%A}"


def _quote(text: str) -> str:
    return text if len(text) <= QUOTE_CHARS else text[:QUOTE_CHARS].rsplit(" ", 1)[0] + "..."


class Situation:
    """The moment around one conversation. `last_turns` are the latest logged exchanges before it."""

    def __init__(self, last_turns: list[dict], now: Callable[[], datetime] = datetime.now):
        self.now = now
        self.last_turns = last_turns
        self.last_seen = datetime.fromisoformat(last_turns[-1]["time"]) if last_turns else None
        self.device = "laptop"
        self.turns = 0
        self._noted: set = set()  # the nights and battery warnings already mentioned

    def background(self) -> list[str]:
        now = self.now()
        lines = [clock(now)]
        if self.device == "phone":
            lines.append("The user is talking to you from their phone, so they may be out.")
        if self.last_turns and self.turns < LAST_TIME_TURNS:
            when = datetime.fromisoformat(self.last_turns[-1]["time"])
            if now - when <= LAST_TIME_WITHIN:
                said = " ".join(f'They said: "{_quote(turn["user"])}" You said: "{_quote(turn["sofia"])}"'
                                for turn in self.last_turns)
                lines.append(f"Your last conversation was {_day(when, now)}, {day_part(when)}. {said}")
        return lines

    def notes(self) -> list[str]:
        now = self.now()
        notes = []
        if self.last_seen and now - self.last_seen >= AWAY:
            notes.append(f"[The user is back after {ago(now - self.last_seen)} away from you.]")
        if now.hour in NIGHT_HOURS and now.date() not in self._noted:
            self._noted.add(now.date())
            notes.append(f"[It is {time_of_day(now)}, the middle of the night.]")
        level = battery()
        if level and level[1]:
            self._noted.discard("battery")  # charging again: warn again next time it runs low
        elif level and level[0] <= LOW_BATTERY and "battery" not in self._noted:
            self._noted.add("battery")
            notes.append(f"[The laptop you run on is down to {level[0]}% battery: the user should plug in the charger.]")
        return notes

    def seen(self) -> None:
        """Call after every exchange."""
        self.last_seen = self.now()
        self.turns += 1
