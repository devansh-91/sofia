import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sofia import settings
from sofia.log import ConversationLog


class SettingsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patch = mock.patch.object(settings, "PATH", Path(directory.name) / "settings.json")
        patch.start()
        self.addCleanup(patch.stop)

    def test_defaults_then_saved_choices(self):
        self.assertEqual(settings.load(), {"voice": None, "brain": "original"})
        settings.save(voice="kokoro:af_sky")
        settings.save(brain="tuned", unknown="ignored")
        self.assertEqual(settings.load(), {"voice": "kokoro:af_sky", "brain": "tuned"})

    def test_a_broken_file_gives_the_defaults(self):
        settings.PATH.write_text("{not json")
        self.assertEqual(settings.load()["brain"], "original")


class HistoryTest(unittest.TestCase):
    def test_history_is_newest_first_and_can_be_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ConversationLog(Path(directory) / "log.jsonl")
            first = log.turn("hi", "Hey, sir.", "english")
            log.turn("bye", "Bye, sir.", "english")
            log.feedback(first, "good")
            self.assertEqual([t["user"] for t in log.history()], ["bye", "hi"])
            log.clear()
            self.assertEqual(log.history(), [])


if __name__ == "__main__":
    unittest.main()
