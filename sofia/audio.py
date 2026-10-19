"""Microphone, speaker and audio conversion, via PipeWire (pw-record / pw-play) and ffmpeg."""
import signal
import subprocess
import tempfile
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 16000  # what Whisper and the emotion model expect


def temporary_wav() -> Path:
    file = tempfile.NamedTemporaryFile(prefix="sofia-", suffix=".wav", delete=False)
    file.close()
    return Path(file.name)


class Recording:
    """Push-to-talk: start() begins capturing from the default microphone, stop() returns the WAV."""

    def __init__(self):
        self.path = temporary_wav()
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            ["pw-record", "--rate", str(SAMPLE_RATE), "--channels", "1", "--format", "s16", str(self.path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> Path:
        # SIGINT lets pw-record finish the WAV header; SIGTERM would leave it unreadable.
        self.process.send_signal(signal.SIGINT)
        self.process.wait(timeout=5)
        return self.path


def clipped_share(wav: Path) -> float:
    """Share of samples at full scale. More than a few percent means the input level is too
    high and speech recognition will fail."""
    with wave.open(str(wav), "rb") as file:
        samples = array("h", file.readframes(file.getnframes()))
    return sum(1 for sample in samples if abs(sample) >= 32700) / len(samples) if samples else 0.0


def to_wav(source: Path) -> Path:
    """Convert any audio ffmpeg understands (e.g. a phone's webm/ogg) to 16 kHz mono WAV."""
    target = temporary_wav()
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", "-ar", str(SAMPLE_RATE), str(target)],
        check=True,
    )
    return target


def play(path: Path) -> None:
    subprocess.run(["pw-play", str(path)], check=False)
