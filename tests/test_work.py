import unittest

from sofia.work import ENTER, LEAVE, TOOL_NAMES, spoken, text_calls


class SwitchTest(unittest.TestCase):
    def test_only_the_whole_message_switches(self):
        for said in ("kaam mode", "Kaam mode.", "/kaam", "Sofia, work mode on", "Come mode!"):
            self.assertTrue(ENTER.match(said), said)
        for said in ("normal mode", "Kaam mode off.", "/normal", "kaam khatam", "Hey Sofia, chill mode"):
            self.assertTrue(LEAVE.match(said), said)
        self.assertFalse(ENTER.match("what is kaam mode?"))
        self.assertFalse(ENTER.match("kaam mode off"))
        self.assertFalse(LEAVE.match("I want a normal mode for my app"))


class TextCallsTest(unittest.TestCase):
    def test_calls_written_as_text_are_found(self):
        alone = '{"name": "read_file", "arguments": {"path": "~/x.py"}}'
        self.assertEqual(text_calls(alone, TOOL_NAMES), [{"function": {"name": "read_file", "arguments": {"path": "~/x.py"}}}])
        tagged = '<tool_call>\n{"name": "make_folder", "arguments": {"path": "a"}}\n</tool_call>\n' * 2
        self.assertEqual(len(text_calls(tagged, TOOL_NAMES)), 2)
        after_words = 'I\'ll propose this:\n```json\n{"name": "write_file", "arguments": {"path": "a.py", "content": "x"}}\n```\nAllow it?'
        self.assertEqual(text_calls(after_words, TOOL_NAMES)[0]["function"]["name"], "write_file")

    def test_other_json_is_not_a_call(self):
        self.assertEqual(text_calls('```json\n{"name": "Dev", "age": 25}\n```', TOOL_NAMES), [])
        self.assertEqual(text_calls("Plain words.", TOOL_NAMES), [])


class SpokenTest(unittest.TestCase):
    def test_she_says_the_words_not_the_code(self):
        answer = "Here's the fix, sir:\n```python\nprint('hi')\n```\nRun it with `python3 hi.py`."
        self.assertEqual(spoken(answer), "Here's the fix, sir: Run it with python3 hi.py.")
        self.assertEqual(spoken("```bash\nls\n```"), "Here you go, sir.")

    def test_long_words_are_cut_at_a_sentence(self):
        said = spoken("This is one sentence about the code. " * 20)
        self.assertLessEqual(len(said), 350)
        self.assertTrue(said.endswith("."))


if __name__ == "__main__":
    unittest.main()
