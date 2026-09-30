"""Talk to Sofia out loud in the terminal (push-to-talk).

    ./scripts/sofia.sh voice

Press Enter, speak, press Enter again. She hears what you said (speech-to-text)
and how you said it (emotion from your voice), answers, and speaks the answer.
Typing a message instead of pressing Enter works too.
"""
import time
from pathlib import Path

from sofia import audio, emotion, stt, tts
from sofia.chat import HELP, Session
from sofia.log import ConversationLog
from sofia.memory import NEUTRAL, Memory


def hear(wav: Path) -> tuple[str, str]:
    """(what was said, how it sounded) for a recording."""
    text = stt.transcribe(wav)
    mood, _ = emotion.detect(wav) if text else (NEUTRAL, 0.0)
    return text, mood


def respond(session: Session, text: str, sounds: str = NEUTRAL) -> tuple[str, list[str], Path, str]:
    """Answer `text` and render the answer as speech: (answer, learned, wav, voice style)."""
    answer, learned = session.say(text, sounds)
    wav, style = tts.speak(answer, session.language, session.mood)
    return answer, learned, wav, style


def main() -> None:
    session = Session(Memory(), ConversationLog())
    print("Loading speech models...", flush=True)
    stt._load(), emotion._load(), tts._load()
    print(f"Sofia — press Enter, speak, press Enter again. Or type a message.\n{HELP}\n")
    while True:
        try:
            text = input("[Enter to talk] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        sounds = NEUTRAL
        if not text:
            recording = audio.Recording()
            recording.start()
            input("  listening... press Enter when done ")
            started = time.time()
            wav = recording.stop()
            if audio.clipped_share(wav) > 0.05:
                print("  (mic is clipping — lower it: wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.2)")
            text, sounds = hear(wav)
            wav.unlink(missing_ok=True)
            if not text:
                print("  (didn't catch that)\n")
                continue
            print(f"  you said: {text}   [voice: {sounds}, {time.time() - started:.1f}s]")
        if text.lower() in {"bye", "/bye", "exit", "quit"}:
            break
        result = session.command(text)
        if result is not None:
            print(f"  {result}\n")
            continue
        answer, learned, wav, style = respond(session, text, sounds)
        print(f"sofia ({style})> {answer}")
        for item in learned:
            print(f"  (yaad rakha: {item})")
        print()
        audio.play(wav)
        wav.unlink(missing_ok=True)
    for item in session.reflect():
        print(f"  (yaad rakha: {item})")


if __name__ == "__main__":
    main()
