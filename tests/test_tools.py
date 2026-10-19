import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from sofia import tools

BRAVE_PAGE = (
    '<div class="snippet svelte-x" data-pos="0" data-type="web"><a href="https://example.com/gold">'
    '<div class="site-name">Example</div><div class="title search-snippet-title line-clamp-1" '
    'title="Gold Rate Today">Gold Rate Today</div></a><div class="content">24K gold is &#8377;14,804 a gram.</div></div>'
    '<div class="snippet svelte-x" data-pos="1" data-type="web"><span>an ad without a link</span></div>'
)
FEED = (
    "<rss><channel><item><title>Old news - Paper</title><pubDate>Fri, 16 Oct 2026 10:00:00 GMT</pubDate></item>"
    "<item><title>India win by 5 wickets - Cricbuzz</title><pubDate>Mon, 19 Oct 2026 09:30:00 GMT</pubDate></item>"
    "</channel></rss>"
)


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.home = Path(self.directory.name).resolve()
        self._real_home, tools.HOME = tools.HOME, self.home
        (self.home / "Downloads").mkdir()
        (self.home / "Downloads" / "big.pdf").write_bytes(b"%PDF" + b"x" * 5000)
        (self.home / "Downloads" / "note.txt").write_text("buy milk")
        (self.home / ".ssh").mkdir()
        (self.home / ".ssh" / "id_rsa").write_text("secret")

    def tearDown(self):
        tools.HOME = self._real_home
        self.directory.cleanup()

    def test_paths_resolve_in_home_loosely(self):
        self.assertEqual(tools.resolve("downloads/note.txt"), self.home / "Downloads" / "note.txt")
        self.assertEqual(tools.resolve("~"), self.home)

    def test_hidden_and_outside_paths_are_refused(self):
        self.assertIn("off limits", tools.call("read_file", {"path": "~/.ssh/id_rsa"}))
        self.assertIn("off limits", tools.call("read_file", {"path": "/etc/passwd"}))
        self.assertIn("off limits", tools.call("list_files", {"path": "Downloads/../../"}))
        os.symlink(self.home / ".ssh" / "id_rsa", self.home / "Downloads" / "innocent.txt")
        self.assertIn("off limits", tools.call("read_file", {"path": "Downloads/innocent.txt"}))

    def test_folder_summary_and_search(self):
        listing = tools.call("list_files", {"path": "Downloads"})
        self.assertIn("0 folders, 2 files (1 pdf, 1 txt)", listing)
        self.assertIn("Biggest: big.pdf", listing)
        self.assertEqual(tools.call("find_files", {"name": "*.PDF"}), "1 found:\n~/Downloads/big.pdf")
        self.assertEqual(tools.call("read_file", {"path": "Downloads/note.txt"}), "buy milk")

    def test_bad_calls_come_back_as_text(self):
        self.assertIn("no tool called", tools.call("delete", {"path": "Downloads"}))
        self.assertIn("wrong arguments", tools.call("read_file", {"file": "x"}))


class WebTest(unittest.TestCase):
    def test_brave_results(self):
        self.assertEqual(tools.brave_results(BRAVE_PAGE),
                         ["Gold Rate Today — https://example.com/gold\n24K gold is ₹14,804 a gram."])

    def test_relevant_skips_the_menus(self):
        page = "Home News Sports Business Login " * 20 + "Petrol price in Delhi today is Rs 94.77 per litre." + " Footer" * 50
        self.assertIn("94.77", tools.relevant(page, "petrol price delhi today", 120))

    def test_headlines_newest_first_with_age(self):
        now = datetime(2026, 10, 19, 11, 0, tzinfo=timezone.utc)
        self.assertEqual(tools.headlines(FEED, now),
                         ["India win by 5 wickets - Cricbuzz (an hour ago)", "Old news - Paper (3 days ago)"])

    def test_only_looking_tools_are_offered(self):
        offered = {schema["function"]["name"] for schema in tools.SCHEMAS}
        self.assertEqual(offered, set(tools.FUNCTIONS))
        self.assertEqual(offered, {"list_files", "find_files", "read_file", "search_docs", "web_search", "news", "weather",
                                   "read_webpage"})


if __name__ == "__main__":
    unittest.main()
