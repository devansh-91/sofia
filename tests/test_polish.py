import unittest

from sofia.polish import fit_sounds, polish, without_sounds


class PolishTest(unittest.TestCase):
    def test_strips_markdown_emoji_and_leaked_instruction(self):
        self.assertEqual(polish("Watch *3 Idiots* tonight 🎉"), "Watch 3 Idiots tonight")
        self.assertEqual(polish("Done, sir! [Reply only in casual English.]"), "Done, sir!")

    def test_plain_text_is_not_rewritten(self):
        self.assertEqual(polish("My bhai, the unga tribe."), "My bhai, the unga tribe.")

    def test_keeps_sound_tags_and_turns_stage_directions_into_them(self):
        self.assertEqual(polish("[laugh] Oh, sir, stop it!"), "[laugh] Oh, sir, stop it!")
        self.assertEqual(polish("*laughs* Oh, sir!"), "[laugh] Oh, sir!")
        self.assertEqual(polish("Long day, sir. (sighs)"), "Long day, sir. [sigh]")
        self.assertEqual(polish("[Chuckling] Fine, sir."), "[chuckle] Fine, sir.")
        self.assertEqual(polish("[smiles] Fine, sir."), "Fine, sir.")

    def test_without_sounds(self):
        self.assertEqual(without_sounds("[laugh] Oh, sir, stop it! [sigh]"), "Oh, sir, stop it!")

    def test_fit_sounds_drops_sounds_that_do_not_fit_the_mood(self):
        self.assertEqual(fit_sounds("[gasp] Oh no, sir. I'm so sorry.", "sad"), "Oh no, sir. I'm so sorry.")
        self.assertEqual(fit_sounds("Sir, [laugh] calm down.", "angry"), "Sir, calm down.")
        self.assertEqual(fit_sounds("[sigh] Long day, sir.", "tired"), "[sigh] Long day, sir.")
        self.assertEqual(fit_sounds("[laugh] Long day, sir.", "tired"), "Long day, sir.")
        self.assertEqual(fit_sounds("[gasp] No way, sir!", "happy"), "[gasp] No way, sir!")

    def test_fit_sounds_keeps_one_sound(self):
        self.assertEqual(fit_sounds("[laugh] Stop it, sir! [chuckle]", "neutral"), "[laugh] Stop it, sir!")


if __name__ == "__main__":
    unittest.main()
