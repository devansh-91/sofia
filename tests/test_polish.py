import unittest

from sofia.lang import ENGLISH, HINGLISH
from sofia.polish import polish


class PolishTest(unittest.TestCase):
    def check(self, raw, expected, language=HINGLISH):
        self.assertEqual(polish(raw, language), expected)

    def test_bhai_as_address_becomes_yaar(self):
        self.check("Bhai, abhi kahan hai?", "Yaar, abhi kahan hai?")
        self.check("Bhai tera boss ekdum gaddar hai.", "Yaar tera boss ekdum gaddar hai.")
        self.check("Arre bhai, tension mat le.", "Arre yaar, tension mat le.")
        self.check("Chill kar. Bhaiya, sab theek hoga.", "Chill kar. Yaar, sab theek hoga.")
        self.check("Kya baat kar raha hai bhai, chhod de.", "Kya baat kar raha hai yaar, chhod de.")

    def test_real_brother_is_kept(self):
        self.check("Tera bhai kahan hai aaj?", "Tera bhai kahan hai aaj?")
        self.check("Apne bhai ke saath movie dekh le.", "Apne bhai ke saath movie dekh le.")

    def test_sofia_speaks_as_a_woman(self):
        self.check("Chal, main leke aata hoon.", "Chal, main leke aati hoon.")
        self.check("Main sun raha hoon.", "Main sun rahi hoon.")
        self.check("Main kal aaunga, pakka.", "Main kal aaungi, pakka.")
        self.check("Main bhi thak gaya hoon.", "Main bhi thak gayi hoon.")

    def test_users_own_forms_are_untouched(self):
        self.check("Tu bahut kaam karta hai yaar.", "Tu bahut kaam karta hai yaar.")
        self.check("Tu kya soch raha hai?", "Tu kya soch raha hai?")

    def test_quoted_speech_is_untouched(self):
        self.check('Usko bol "main nahi aaunga, bhai."', 'Usko bol "main nahi aaunga, bhai."')

    def test_strips_markdown_emoji_and_leaked_instruction(self):
        self.check("Watch *3 Idiots* tonight 🎉", "Watch 3 Idiots tonight", ENGLISH)
        self.check("Haan yaar! [Reply in simple Roman-script Hinglish.]", "Haan yaar!")

    def test_english_is_not_rewritten(self):
        self.check("My bhai, the unga tribe.", "My bhai, the unga tribe.", ENGLISH)


if __name__ == "__main__":
    unittest.main()
