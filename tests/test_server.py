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
from unittest import mock

from sofia import tts
from sofia.tts import STYLES, pieces


class FakeSofia:
    """Stands in for the model: the server's routing and access control are what is tested."""

    speech = False

    def __init__(self, audio_dir: Path):
        self.audio_dir = audio_dir
        (audio_dir / "reply.wav").write_bytes(b"RIFFfake")
        (audio_dir / "secret.txt").write_text("private")
        self.known = ["User likes chai."]

    fillers = {}
    mode = "normal"

    def filler(self, name):
        return None

    def audio(self, name):
        wav = self.audio_dir / name
        return wav if name == "reply.wav" else None

    def chat(self, text, sounds="neutral", device="laptop"):
        return {"answer": f"echo: {text}", "learned": [], "mood": "neutral", "audio": None}

    def hello(self, device):
        return {"answer": f"hello, {device}", "audio": None}

    def settings(self):
        return {"voice": "turbo:af_heart", "memory": list(self.known)}

    def change(self, voice=None, brain=None):
        self.changed = (voice, brain)
        return self.settings()

    def memory(self, forget=None, add=""):
        if forget == "all":
            self.known = []
        elif forget:
            self.known.pop(forget - 1)
        if add:
            self.known.append(add)
        return self.settings()

    def history(self):
        return [{"time": "2026-10-19T18:00:00", "user": "hi", "sofia": "Hey, sir."}]

    def forget_history(self):
        return {"ok": True}

    def voice(self, upload, device="laptop", amend=False):
        return {"heard": "amended" if amend else "said"}


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

    def request(self, path, key=None, data=None, agent=None):
        headers = {"X-Sofia-Key": key} if key else {}
        if agent:
            headers["User-Agent"] = agent
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

    def test_hello_knows_the_device(self):
        phone = "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/131.0 Mobile Safari/537.36"
        self.assertEqual(self.request("/api/hello", data=b"{}")[0], 403)
        status, body = self.request("/api/hello", key="right-key", data=b"{}", agent=phone)
        self.assertEqual((status, json.loads(body)["answer"]), (200, "hello, phone"))
        status, body = self.request("/api/hello", key="right-key", data=b"{}", agent="Mozilla/5.0 (X11; Linux x86_64)")
        self.assertEqual(json.loads(body)["answer"], "hello, laptop")

    def test_only_spoken_replies_can_be_fetched(self):
        self.assertEqual(self.request("/audio/reply.wav", key="right-key"), (200, b"RIFFfake"))
        self.assertEqual(self.request("/audio/secret.txt", key="right-key")[0], 404)
        self.assertEqual(self.request("/audio/..%2fsecret.txt", key="right-key")[0], 404)

    def test_settings_memory_and_history_need_the_key(self):
        for path in ("/api/settings", "/api/history", "/api/docs"):
            self.assertEqual(self.request(path)[0], 403)
        for path in ("/api/settings", "/api/memory", "/api/history/clear", "/api/data/clear", "/api/docs/remove"):
            self.assertEqual(self.request(path, data=b"{}")[0], 403)

    def test_settings_and_memory_round_trip(self):
        status, body = self.request("/api/settings", key="right-key", data=json.dumps({"voice": "kokoro:af_sky"}).encode())
        self.assertEqual((status, Handler.sofia.changed), (200, ("kokoro:af_sky", None)))
        body = json.loads(self.request("/api/memory", key="right-key", data=b'{"add": "User runs 5k."}')[1])
        self.assertEqual(body["memory"], ["User likes chai.", "User runs 5k."])
        body = json.loads(self.request("/api/memory", key="right-key", data=b'{"forget": 1}')[1])
        self.assertEqual(body["memory"], ["User runs 5k."])
        turns = json.loads(self.request("/api/history", key="right-key")[1])["turns"]
        self.assertEqual(turns[0]["sofia"], "Hey, sir.")

    def test_voice_is_unavailable_without_speech(self):
        self.assertEqual(self.request("/api/voice", key="right-key", data=b"audio")[0], 404)


class VoiceStyleTest(unittest.TestCase):
    def test_every_mood_has_a_voice_style(self):
        self.assertEqual(set(STYLES), set(MOOD_NAMES))

    def test_changing_a_style_makes_a_new_clone(self):
        before = tts.turbo_voice(style="soft")
        line, pace, pitch, tone, gain = tts.TURBO_STYLES["soft"]
        with mock.patch.dict(tts.TURBO_STYLES, {"soft": (line, pace, pitch - 1, tone, gain)}):
            self.assertNotEqual(tts.turbo_voice(style="soft"), before)
        self.assertEqual(tts.turbo_voice(style="warm").name, f"turbo-{tts.VOICE}.pt")

    def test_reply_is_spoken_in_pieces_at_sentence_ends(self):
        self.assertEqual(pieces("Hi, sir!"), ["Hi, sir!"])
        self.assertEqual(pieces("Long day, sir. [sigh]"), ["Long day, sir. [sigh]"])
        long = "[laugh] Oh, sir, stop it! You always do this. Anyway, how was the meeting? Did they approve it? I hope so."
        self.assertEqual(pieces(long), ["[laugh] Oh, sir, stop it! You always do this.",
                                        "Anyway, how was the meeting? Did they approve it? I hope so."])


if __name__ == "__main__":
    unittest.main()
