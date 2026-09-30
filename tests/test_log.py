import json
import stat
import tempfile
import unittest
from pathlib import Path

from sofia.log import BAD, GOOD, ConversationLog, training_examples


class ConversationLogTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "conversations.jsonl"
        self.log = ConversationLog(self.path)

    def test_turns_are_appended_one_json_object_per_line(self):
        first = self.log.turn("Kya haal hai?", "Sab badhiya yaar.", "hinglish")
        second = self.log.turn("आज मन उदास है", "Kya hua?", "hinglish")
        records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["id"] for r in records], [first, second])
        self.assertNotEqual(first, second)
        self.assertEqual(records[1]["user"], "आज मन उदास है")

    def test_log_is_private(self):
        self.log.turn("hi there friend", "hey", "english")
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_training_examples_apply_feedback(self):
        liked = self.log.turn("Roast me.", "You are slow.", "english")
        disliked = self.log.turn("Joke suna.", "nonsense", "hinglish")
        corrected = self.log.turn("Tu kaun hai?", "wrong", "hinglish")
        self.log.turn("plain", "plain reply", "english")
        self.log.feedback(liked, GOOD)
        self.log.feedback(disliked, BAD)
        self.log.feedback(corrected, BAD)
        self.log.correction(corrected, "Main Sofia hoon.")
        examples = training_examples(self.path)
        self.assertEqual(
            [(e["messages"][0]["content"], e["messages"][1]["content"], e["rating"]) for e in examples],
            [("Roast me.", "You are slow.", GOOD), ("Tu kaun hai?", "Main Sofia hoon.", GOOD), ("plain", "plain reply", None)],
        )

    def test_no_log_yet(self):
        self.assertEqual(training_examples(self.path), [])


if __name__ == "__main__":
    unittest.main()
