import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from sofia.memory import MOOD_NAMES
from sofia.server import Handler
from sofia.tts import STYLES


class FakeSofia:
    """Stands in for the model: the server's routing and access control are what is tested."""

    speech = False

    def __init__(self, audio_dir: Path):
        self.audio_dir = audio_dir
        self.replies = [audio_dir / "reply.wav"]
        self.replies[0].write_bytes(b"RIFFfake")
        (audio_dir / "secret.txt").write_text("private")

    def chat(self, text, sounds="neutral"):
        return {"answer": f"echo: {text}", "learned": [], "mood": "neutral", "audio": None}


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        Handler.key = "right-key"
        Handler.sofia = FakeSofia(Path(cls.directory.name))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.directory.cleanup()

    def request(self, path, key=None, data=None):
        headers = {"X-Sofia-Key": key} if key else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(self.url + path, data=data, headers=headers)) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def test_page_is_served_without_a_key(self):
        status, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"<title>Sofia</title>", body)

    def test_api_refuses_a_missing_or_wrong_key(self):
        self.assertEqual(self.request("/api/status")[0], 403)
        self.assertEqual(self.request("/api/status", key="wrong")[0], 403)
        self.assertEqual(self.request("/api/chat", data=b'{"text": "hi"}')[0], 403)
        self.assertEqual(self.request("/audio/reply.wav")[0], 403)

    def test_key_works_in_header_or_link(self):
        self.assertEqual(self.request("/api/status", key="right-key")[0], 200)
        self.assertEqual(self.request("/api/status?key=right-key")[0], 200)

    def test_chat_round_trip(self):
        status, body = self.request("/api/chat", key="right-key", data=json.dumps({"text": "Kya haal hai?"}).encode())
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["answer"], "echo: Kya haal hai?")

    def test_only_spoken_replies_can_be_fetched(self):
        self.assertEqual(self.request("/audio/reply.wav", key="right-key"), (200, b"RIFFfake"))
        self.assertEqual(self.request("/audio/secret.txt", key="right-key")[0], 404)
        self.assertEqual(self.request("/audio/..%2fsecret.txt", key="right-key")[0], 404)

    def test_voice_is_unavailable_without_speech(self):
        self.assertEqual(self.request("/api/voice", key="right-key", data=b"audio")[0], 404)


class VoiceStyleTest(unittest.TestCase):
    def test_every_mood_has_a_voice_style(self):
        self.assertEqual(set(STYLES), set(MOOD_NAMES))


if __name__ == "__main__":
    unittest.main()
