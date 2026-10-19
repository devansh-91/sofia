import unittest

from sofia.chat import Streamer


def streamed(text: str, mood: str = "neutral", crisis: bool = False, step: int = 3) -> tuple[list[str], Streamer]:
    spoken = []
    stream = Streamer(spoken.append, mood, crisis)
    for start in range(0, len(text), step):  # tokens arrive a few characters at a time
        stream.feed(text[start:start + step])
    stream.finish()
    return spoken, stream


class StreamerTest(unittest.TestCase):
    def test_pieces_are_cut_at_sentence_ends_short_first(self):
        spoken, stream = streamed("Oh, sir, that's huge! I knew you'd get it. We're celebrating tonight, no excuses at all, "
                                  "and you are paying for everything. Deal?")
        self.assertEqual(spoken[0], "Oh, sir, that's huge! I knew you'd get it.")  # 30+ characters
        self.assertEqual(" ".join(spoken), stream.text)
        self.assertTrue(stream.answered)
        self.assertTrue(stream.text.endswith("Deal?"))

    def test_at_most_one_sound_and_none_in_a_crisis(self):
        spoken, _ = streamed("[laugh] Oh, sir, stop it right now! [laugh] You're killing me with this one.")
        self.assertEqual(sum(piece.count("[laugh]") for piece in spoken), 1)
        spoken, _ = streamed("[sigh] I'm here with you, sir, please stay. Call 14416 tonight.", crisis=True)
        self.assertFalse(any("[" in piece for piece in spoken))

    def test_sounds_that_dont_fit_the_mood_go(self):
        spoken, _ = streamed("[laugh] Oh no, sir, I'm so sorry to hear that. Tell me what happened.", mood="sad")
        self.assertNotIn("[laugh]", " ".join(spoken))

    def test_markdown_is_cleaned_and_a_lone_sound_is_no_piece(self):
        spoken, _ = streamed("Watch *Interstellar* tonight, sir, it's perfect. [chuckle]")
        self.assertEqual(spoken, ["Watch Interstellar tonight, sir, it's perfect. [chuckle]"])

    def test_hinglish_first_piece_stops_the_stream(self):
        spoken, stream = streamed("Arre sir, aaj toh bahut mazaa aaya, chalo party karte hai! Kal milte hai.")
        self.assertEqual(spoken, [])
        self.assertFalse(stream.answered)


if __name__ == "__main__":
    unittest.main()
