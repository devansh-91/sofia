import unittest

from sofia.stt import _HER_NAME


class HerNameTest(unittest.TestCase):
    def test_her_name_is_spelled_her_way(self):
        self.assertEqual(_HER_NAME.sub("Sofia", "Hey Sophia, are you there? sophie?"), "Hey Sofia, are you there? Sofia?")
        self.assertEqual(_HER_NAME.sub("Sofia", "Sophisticated philosophy"), "Sophisticated philosophy")


if __name__ == "__main__":
    unittest.main()
