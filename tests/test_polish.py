import unittest

from sofia.polish import polish


class PolishTest(unittest.TestCase):
    def test_strips_markdown_emoji_and_leaked_instruction(self):
        self.assertEqual(polish("Watch *3 Idiots* tonight 🎉"), "Watch 3 Idiots tonight")
        self.assertEqual(polish("Done, sir! [Reply only in casual English.]"), "Done, sir!")

    def test_plain_text_is_not_rewritten(self):
        self.assertEqual(polish("My bhai, the unga tribe."), "My bhai, the unga tribe.")


if __name__ == "__main__":
    unittest.main()
