import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sofia import docs


def fake_embed(texts, kind):
    """Vectors from word counts over a tiny vocabulary: enough to rank passages by topic."""
    words = ["gym", "protein", "exam", "dbms", "python"]
    return [[text.lower().count(word) + 0.01 for word in words] for text in texts]


class DocsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for patch in (mock.patch.object(docs, "FOLDER", root / "docs"), mock.patch.object(docs, "INDEX", root / "index.json"),
                      mock.patch.object(docs, "_embed", side_effect=fake_embed), mock.patch.object(docs, "refresh_soon")):
            patch.start()
            self.addCleanup(patch.stop)

    def test_passages_overlap_and_cover_the_text(self):
        text = " ".join(f"word{i}" for i in range(600))
        found = docs.passages(text)
        self.assertGreater(len(found), 1)
        self.assertTrue(all(len(p) <= docs.PASSAGE for p in found))
        self.assertIn("word599", found[-1])

    def test_search_finds_the_right_file_and_follows_the_folder(self):
        docs.add("fitness.md", b"My gym plan: protein every day, gym four times a week.")
        docs.add("exams.txt", b"DBMS exam topics: normalisation, SQL joins, transactions. The exam is on Monday.")
        self.assertTrue(docs.search("when is the dbms exam")[0]["file"].endswith("exams.txt"))
        self.assertEqual({d["name"] for d in docs.listing()}, {"fitness.md", "exams.txt"})
        docs.remove("exams.txt")
        self.assertTrue(all(hit["file"].endswith("fitness.md") for hit in docs.search("dbms exam")))

    def test_only_readable_kinds_and_plain_names_are_added(self):
        with self.assertRaises(ValueError):
            docs.add("setup.exe", b"MZ")
        self.assertEqual(docs.add("../../evil.txt", b"x").name, "evil.txt")
        self.assertEqual(docs.add("../../evil.txt", b"x").parent, docs.FOLDER)
        with self.assertRaises(FileNotFoundError):
            docs.remove("../index.json")


if __name__ == "__main__":
    unittest.main()
