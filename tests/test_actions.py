import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sofia import actions, tools
from sofia.actions import Pending


class ActionsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name).resolve()
        (self.home / "Documents").mkdir()
        (self.home / ".ssh").mkdir()
        for patch in (mock.patch.object(tools, "HOME", self.home),
                      mock.patch.object(actions, "WORKSPACE", self.home / "sofia-workspace")):
            patch.start()
            self.addCleanup(patch.stop)
        self.pending = Pending()

    def test_nothing_runs_until_allowed(self):
        action, told = self.pending.add("write_file", {"path": "hello.py", "content": "print('hi')\n"})
        self.assertIn("#1", told)
        self.assertEqual(action.title, "Create ~/sofia-workspace/hello.py (12 B)")
        self.assertFalse((self.home / "sofia-workspace" / "hello.py").exists())
        self.assertEqual(self.pending.decide(1, allowed=False)[1], "The user declined it.")
        self.assertFalse((self.home / "sofia-workspace" / "hello.py").exists())
        self.assertIsNone(self.pending.decide(1, allowed=True))  # a decided proposal is gone

    def test_allowed_write_and_command(self):
        self.pending.add("write_file", {"path": "hello.py", "content": "print('hi')\n"})
        self.assertIn("Wrote", self.pending.decide(1, allowed=True)[1])
        self.pending.add("run_command", {"command": "python3 hello.py"})
        self.assertEqual(self.pending.decide(2, allowed=True)[1], "Exit code 0\nhi")

    def test_paths_stay_in_the_home_folder_and_out_of_hidden_ones(self):
        for arguments in ({"path": "/etc/passwd", "content": "x"}, {"path": "~/.ssh/authorized_keys", "content": "x"}):
            action, told = self.pending.add("write_file", arguments)
            self.assertIsNone(action)
            self.assertTrue(told.startswith("Not proposed"))
        action, _ = self.pending.add("make_folder", {"path": "Documents/notes"})
        self.assertEqual(action.title, "Make the folder ~/Documents/notes")

    def test_no_root_commands(self):
        for command in ("sudo rm -rf /tmp/x", "ls; sudo reboot", "echo hi | su"):
            self.assertIsNone(self.pending.add("run_command", {"command": command})[0], command)

    def test_delete_goes_to_the_trash_and_says_so(self):
        (self.home / "Documents" / "old.txt").write_text("bye")
        action, _ = self.pending.add("delete", {"path": "~/Documents/old.txt"})
        self.assertEqual(action.title, "Move ~/Documents/old.txt (file, 3 B) to the Trash")
        self.assertIsNone(self.pending.add("delete", {"path": "~"})[0])


if __name__ == "__main__":
    unittest.main()
