import unittest

from sofia.care import CRISIS, notes, spelled


class CareTest(unittest.TestCase):
    def test_self_harm_words_bring_only_the_crisis_note(self):
        for said in ("I want to kill myself", "sometimes I think about suicide", "jeene ka mann nahi karta",
                     "mujhe marna hai", "I've been cutting myself again"):
            self.assertEqual(notes(said), ([CRISIS.note], True), said)

    def test_feelings_bring_their_modules(self):
        found, crisis = notes("Bahut anxiety ho rahi hai, neend nahi aa rahi")
        self.assertFalse(crisis)
        self.assertTrue(found and "mental health" in found[0])
        self.assertIn("relationships", notes("My girlfriend ghosted me")[0][0])
        self.assertIn("emotional intelligence", notes("I get so jealous of my friends")[0][0])

    def test_helpline_name_is_spelled_right(self):
        self.assertEqual(spelled("Call Tele-MANANS or tele manan"), "Call Tele-MANAS or Tele-MANAS")

    def test_ordinary_messages_bring_none(self):
        self.assertEqual(notes("What's the weather in Gorakhpur?"), ([], False))
        self.assertEqual(notes("I killed it at the gym today"), ([], False))


if __name__ == "__main__":
    unittest.main()
