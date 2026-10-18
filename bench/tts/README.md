# TTS bench

Picks Sofia's next voice by ear. Every engine speaks the same 10 Hinglish lines (`lines.json`:
neutral, happy, laughing, sad, calm, firm, an agent report, a code report, a sigh, a playful
question). You rate each clip blind on a listening page.

| Engine | What it is | Voice |
|---|---|---|
| `piper` | Sofia today: Piper Hindi + pitch/pace reshaping | its own |
| `svara` | svara-TTS v1, 3B Orpheus-style Indic model, Q4 GGUF via Ollama; `<happy>`/`<sad>` tags | "Hindi (Female)" |
| `omnivoice` | k2-fsa/OmniVoice (the model behind VoiceStudio); `[laughter]`, `[sigh]` | clones the reference |
| `chatterbox` | Resemble Chatterbox, Hindi finetune; mood → exaggeration | clones the reference |
| `indicf5` | AI4Bharat IndicF5 + hinglish-tts duration patch (gated on Hugging Face) | clones the reference |

The reference voice is svara's "Hindi (Female)" saying `common.REF_TEXT`, so the cloning engines
all copy one timbre and the comparison comes down to delivery. To try a real voice instead, put a
clean 5–10 s clip at `$SOFIA_BENCH_DIR/ref/ref.wav` with its transcript in `ref.txt`, then rerun.

## Run

```sh
bench/tts/setup.sh                 # one venv per engine under ~/.local/share/sofia/tts-bench
bench/tts/run.sh                   # all engines (or name some: run.sh omnivoice chatterbox)
.venv/bin/python bench/tts/check.py   # Whisper intelligibility check (CER)
python3 bench/tts/serve.py         # http://127.0.0.1:8765  (--host 0.0.0.0 for the phone)
```

`run.sh` unloads every Ollama model first, so each GPU engine starts from an empty GPU. It then
repeats three lines on the CPU, because next to Sofia's LLM only ~1–2 GB of VRAM is left.
Ratings save to `$SOFIA_BENCH_DIR/ratings.json`.
