import json
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from sofia import situation
from sofia.log import ConversationLog
from sofia.situation import Situation, ago, clock

YESTERDAY = [{"time": "2026-10-18T20:31:00", "user": "Kal presentation hai", "sofia": "You've got this, sir."}]


class Clock:
    def __init__(self, moment: str):
        self.moment = datetime.fromisoformat(moment)

    def __call__(self) -> datetime:
        return self.moment


class SituationTest(unittest.TestCase):
    def setUp(self):
        self.battery = (77, True)
        self._real_battery = situation.battery
        situation.battery = lambda: self.battery

    def tearDown(self):
        situation.battery = self._real_battery

    def test_clock_and_gaps_read_naturally(self):
        self.assertEqual(clock(datetime(2026, 10, 19, 16, 5)), "It is Monday, 19 October, 4:05 PM, in the afternoon.")
        self.assertIn("and it's Valentine's Day.", clock(datetime(2027, 2, 14, 0, 30)))
        self.assertEqual(ago(timedelta(days=4, hours=2)), "4 days")
        self.assertEqual(ago(timedelta(days=1, hours=20)), "2 days")
        self.assertEqual(ago(timedelta(days=1, hours=2)), "a day")
        self.assertEqual(ago(timedelta(hours=5, minutes=50)), "6 hours")
        self.assertEqual(ago(timedelta(hours=3)), "3 hours")

    def test_back_after_a_long_gap(self):
        now = Clock("2026-10-22T19:00:00")
        here = Situation(YESTERDAY, now)
        self.assertEqual(here.notes(), ["[The user is back after 4 days away from you.]"])
        here.seen()
        now.moment += timedelta(minutes=2)
        self.assertEqual(here.notes(), [])

    def test_the_night_is_noted_once(self):
        now = Clock("2026-10-19T02:40:00")
        here = Situation([], now)
        self.assertEqual(here.notes(), ["[It is 2:40 AM, the middle of the night.]"])
        now.moment += timedelta(minutes=10)
        self.assertEqual(here.notes(), [])

    def test_low_battery_is_noted_until_charged(self):
        here = Situation([], Clock("2026-10-19T15:00:00"))
        self.battery = (9, False)
        self.assertEqual(len(here.notes()), 1)
        self.assertEqual(here.notes(), [])
        self.battery = (12, True)
        here.notes()
        self.battery = (8, False)
        self.assertIn("8% battery", here.notes()[0])

    def test_last_conversation_shows_early_and_when_recent(self):
        now = Clock("2026-10-19T09:10:00")
        here = Situation(YESTERDAY, now)
        background = " ".join(here.background())
        self.assertIn("Your last conversation was yesterday, in the evening.", background)
        self.assertIn('"Kal presentation hai"', background)
        for _ in range(3):
            here.seen()
        self.assertNotIn("last conversation", " ".join(here.background()))
        self.assertNotIn("last conversation", " ".join(Situation(YESTERDAY, Clock("2026-11-01T09:00:00")).background()))

    def test_phone_is_mentioned(self):
        here = Situation([], Clock("2026-10-19T13:00:00"))
        self.assertEqual(len(here.background()), 1)
        here.device = "phone"
        self.assertIn("phone", here.background()[1])


class LastTurnsTest(unittest.TestCase):
    def test_latest_turns_oldest_first(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            log = ConversationLog(path)
            self.assertEqual(log.last_turns(3), [])
            for text in ("one", "two", "three", "four"):
                log.turn(text, "ok", "english")
            log.feedback("x", "good")
            self.assertEqual([turn["user"] for turn in log.last_turns(3)], ["two", "three", "four"])
            self.assertEqual(json.loads(path.read_text().splitlines()[-1])["type"], "feedback")


if __name__ == "__main__":
    unittest.main()
