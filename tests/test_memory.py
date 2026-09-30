import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from sofia.lang import ENGLISH, HINGLISH
from sofia.memory import _STANDING_RULE, BEHAVIOUR, EXAMPLES, FACTS, LIMITS, PREFERENCES, TAUGHT, Memory, has_real_dates, is_grounded


class MemoryTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "nested" / "memory.json"
        self.memory = Memory(self.path)

    def test_survives_a_restart(self):
        self.memory.add(FACTS, "User has a dog named Bruno.")
        self.memory.add(PREFERENCES, "Keep replies short.")
        self.memory.save()
        reloaded = Memory(self.path)
        self.assertEqual(reloaded.numbered(), ["User has a dog named Bruno.", "Keep replies short."])

    def test_near_duplicates_are_not_stored_twice(self):
        self.assertTrue(self.memory.add(FACTS, "User lives in Pune."))
        self.assertFalse(self.memory.add(FACTS, "user lives in Pune"))
        self.assertTrue(self.memory.add(FACTS, "User lives in Delhi with two flatmates."))
        self.assertEqual(len(self.memory.items[FACTS]), 2)

    def test_update_replaces_the_outdated_fact(self):
        self.memory.add(FACTS, "User is 24 years old.")
        self.memory.add(FACTS, "User plays guitar.")
        self.memory.add(FACTS, "User is 25 years old.", replaces=0)
        self.assertEqual(self.memory.items[FACTS], ["User plays guitar.", "User is 25 years old."])

    def test_oldest_facts_drop_out_past_the_limit(self):
        for number in range(LIMITS[FACTS] + 5):
            self.memory.add(FACTS, f"User knows a person called friend number {number:03d} from school days.")
        self.assertEqual(len(self.memory.items[FACTS]), LIMITS[FACTS])
        self.assertIn("005", self.memory.items[FACTS][0])

    def test_forget_uses_the_numbering_shown_to_the_user(self):
        self.memory.add(FACTS, "User has a dog named Bruno.")
        self.memory.add(PREFERENCES, "Keep replies short.")
        self.assertEqual(self.memory.forget(2), "Keep replies short.")
        self.assertIsNone(self.memory.forget(2))
        self.assertEqual(Memory(self.path).numbered(), ["User has a dog named Bruno."])

    def test_nothing_is_added_to_the_prompt_when_empty(self):
        self.assertEqual(self.memory.background(), [])
        self.assertEqual(self.memory.preference_note(), "")

    def test_facts_and_preferences_reach_the_prompt(self):
        self.memory.add(FACTS, "User has a dog named Bruno.")
        self.memory.add(PREFERENCES, "Keep replies short.")
        background = self.memory.background()
        self.assertEqual([m["role"] for m in background], ["user", "assistant"])
        self.assertIn("User has a dog named Bruno.", background[0]["content"])
        self.assertIn("Keep replies short.", self.memory.preference_note())


class ExamplesTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "memory.json"
        self.memory = Memory(self.path)

    def test_liked_reply_is_saved_and_survives_a_restart(self):
        self.memory.add_example("Mujhe roast kar.", "Tu alarm se bhi late uthta hai.", HINGLISH)
        self.assertEqual(Memory(self.path).items[EXAMPLES],
                         [{"user": "Mujhe roast kar.", "sofia": "Tu alarm se bhi late uthta hai.", "language": HINGLISH}])

    def test_correcting_the_same_message_replaces_the_old_example(self):
        self.memory.add_example("Roast me.", "You are slow.", ENGLISH)
        self.memory.add_example("roast me", "You have the speed of a Monday morning.", ENGLISH)
        self.assertEqual([e["sofia"] for e in self.memory.items[EXAMPLES]], ["You have the speed of a Monday morning."])

    def test_only_examples_for_similar_messages_are_used_closest_last(self):
        self.memory.add_example("Boss ne daant diya aaj.", "english boss reply", ENGLISH)
        self.memory.add_example("Mujhe roast kar.", "roast reply", HINGLISH)
        self.memory.add_example("Boss ne aaj office mein daant diya.", "office reply", HINGLISH)
        self.memory.add_example("Aaj boss ne daant diya.", "boss reply", HINGLISH)
        chosen = self.memory.examples_for("Aaj phir boss ne daant diya", HINGLISH)
        self.assertEqual([e["sofia"] for e in chosen], ["office reply", "boss reply"])
        self.assertEqual(self.memory.examples_for("Mujhe neend nahi aa rahi.", HINGLISH), [])

    def test_everything_is_listed_and_forgettable_by_number(self):
        self.memory.add(FACTS, "User has a dog named Bruno.")
        self.memory.add(TAUGHT, "In Mumbai, 'bantai' means friend.")
        self.memory.add(PREFERENCES, "Keep replies short.")
        self.memory.add_example("Roast me.", "You are slow.", ENGLISH)
        listed = self.memory.numbered()
        self.assertEqual(listed[:3], ["User has a dog named Bruno.", "In Mumbai, 'bantai' means friend.", "Keep replies short."])
        self.assertIn("You are slow.", listed[3])
        self.assertEqual(self.memory.forget(4), listed[3])
        self.assertEqual(self.memory.forget(2), "In Mumbai, 'bantai' means friend.")
        self.assertEqual(Memory(self.path).numbered(), ["User has a dog named Bruno.", "Keep replies short."])

    def test_taught_items_are_part_of_the_background(self):
        self.memory.add(TAUGHT, "In Mumbai, 'bantai' means friend.")
        self.assertIn("'bantai' means friend", self.memory.background()[0]["content"])


class BehaviourTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "memory.json"
        self.memory = Memory(self.path)

    def record(self, count, mood, hour, language=HINGLISH):
        for number in range(count):
            self.memory.record_mood(mood, language, datetime(2026, 9, 1 + number % 5, hour, 15))

    def test_one_long_chat_is_not_a_habit(self):
        for _ in range(30):
            self.memory.record_mood("stressed", HINGLISH, datetime(2026, 9, 30, 23, 15))
        self.assertEqual(self.memory.habits(), [])

    def test_no_habits_from_a_handful_of_messages(self):
        self.record(9, "stressed", 23)
        self.assertEqual(self.memory.habits(), [])

    def test_habits_are_counted_from_messages(self):
        self.record(7, "stressed", 23)
        self.record(3, "neutral", 1)
        self.record(2, "happy", 14, ENGLISH)
        self.assertEqual(self.memory.habits(), [
            "User usually chats late at night.",
            "User mostly writes in Hinglish.",
            "Lately the user has often been stressed.",
        ])

    def test_mixed_messages_give_no_habits(self):
        for hour, mood, language in [(8, "happy", ENGLISH), (13, "sad", HINGLISH), (19, "bored", ENGLISH), (23, "tired", HINGLISH)] * 3:
            self.memory.record_mood(mood, language, datetime(2026, 9, 1 + hour % 4, hour, 0))
        self.assertEqual(self.memory.habits(), [])

    def test_behaviour_and_habits_reach_the_prompt_and_survive_a_restart(self):
        self.memory.add(BEHAVIOUR, "User jokes when stressed.")
        self.record(10, "neutral", 23)
        self.memory.save()
        background = Memory(self.path).background()[0]["content"]
        self.assertIn("User jokes when stressed.", background)
        self.assertIn("User usually chats late at night.", background)

    def test_behaviour_is_listed_and_forgettable(self):
        self.memory.add(FACTS, "User has a dog named Bruno.")
        self.memory.add(BEHAVIOUR, "User jokes when stressed.")
        self.assertEqual(self.memory.forget(2), "User jokes when stressed.")

    def test_a_changed_rule_replaces_the_old_one(self):
        self.assertTrue(self.memory._add_preference("Call the user Sam."))
        self.assertTrue(self.memory._add_preference("Call the user Dev."))
        self.assertFalse(self.memory._add_preference("call the user dev"))
        self.assertTrue(self.memory._add_preference("Keep replies short."))
        self.assertEqual(self.memory.items[PREFERENCES], ["Call the user Dev.", "Keep replies short."])

    def test_forget_all_also_wipes_moods(self):
        self.record(12, "sad", 23)
        self.memory.clear()
        self.assertEqual(Memory(self.path).habits(), [])


class GroundingTest(unittest.TestCase):
    def test_names_must_come_from_the_message(self):
        message = "Tu mujhe yaar mat bol, mera naam le - Devansh."
        self.assertTrue(is_grounded("User's name is Devansh.", message))
        self.assertTrue(is_grounded("Do not call the user 'yaar'.", message))
        self.assertFalse(is_grounded("Do not call the user 'Dev'.", message))
        self.assertFalse(is_grounded("User lives in Pune.", message))

    def test_worked_out_dates_pass(self):
        self.assertTrue(is_grounded("User has an interview at Google on 1 October 2026.", "Kal mera interview hai Google mein."))

    def test_impossible_dates_are_rejected(self):
        self.assertFalse(has_real_dates("User's sister is getting married on 31 September 2026."))
        self.assertTrue(has_real_dates("User has an interview at Google on 1 October 2026."))
        self.assertTrue(has_real_dates("User plays guitar on weekends."))

    def test_only_messages_that_word_a_rule_can_set_a_preference(self):
        for message in ["Mujhe roast kar.", "Mere boss ko gaali de.", "Tell me a joke.", "Hey Sofia, kya chal raha hai?"]:
            self.assertIsNone(_STANDING_RULE.search(message), message)
        for message in ["Stop giving me advice, just listen.", "Mujhe bhai mat bola kar.", "Mujhe Sam bula.",
                        "Thoda zyada gaali de kar baat kiya kar.", "Tu har baar paani peene ko kyun bolti hai? Band kar ye.",
                        "From now on, roast me whenever I act lazy.", "Hamesha Hinglish mein baat kiya kar."]:
            self.assertIsNotNone(_STANDING_RULE.search(message), message)

    def test_plain_sentences_pass(self):
        self.assertTrue(is_grounded("User works as a software engineer.", "Main ek software engineer hoon."))
        self.assertTrue(is_grounded("Keep replies short.", "itna lamba jawab mat diya kar"))


if __name__ == "__main__":
    unittest.main()
