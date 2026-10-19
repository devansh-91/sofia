"""python3 -m unittest discover tests"""
import unittest

from sofia.lang import ENGLISH, HINGLISH, detect


class DetectTest(unittest.TestCase):
    def test_hinglish(self):
        for text in [
            "Yaar aaj office mein bahut bura din tha, boss ne sabke saamne daant diya.",
            "Mummy se aaj phir ladai ho gayi.",
            "Tu kaun hai?",
            "Kal movie dekhne chalein?",
            "Weekend pe kya karu, bahut bore ho raha hoon.",
            "Ek ganda joke suna.",
            "Mujhe roast kar, full savage.",
            "Boss ne meeting mein daant diya",
        ]:
            self.assertEqual(detect(text, previous=ENGLISH), HINGLISH, text)

    def test_devanagari_counts_as_hinglish(self):
        self.assertEqual(detect("आज मेरा मन बहुत उदास है।", previous=ENGLISH), HINGLISH)

    def test_english(self):
        for text in [
            "I had a terrible day at work, my boss yelled at me in front of everyone.",
            "What do you do all day?",
            "Tell me a dirty joke.",
            "bro I'm so done with this job",
            "The sun is out, let me go to the main road.",
            "Roast me, don't hold back.",
            "I want biryani so bad right now.",
            "ok cool, talk to you later then",
        ]:
            self.assertEqual(detect(text, previous=HINGLISH), ENGLISH, text)

    def test_too_short_keeps_previous_language(self):
        for text in ["haha ok", "hmm", "lol", "?"]:
            self.assertEqual(detect(text, previous=ENGLISH), ENGLISH, text)
            self.assertEqual(detect(text, previous=HINGLISH), HINGLISH, text)


if __name__ == "__main__":
    unittest.main()
