# TTS bench

Picks Sofia's voice by ear. Every voice speaks the same 10 English lines (`lines.json`: a greeting,
happy, laughing, sad, calm, firm, an agent report, a code report, a sigh, a playful question), and
you rate each clip blind on a listening page.

| Engine | What it is | Runs on |
|---|---|---|
| `kokoro-*` | Kokoro-82M (ONNX), exactly as Sofia speaks today (`sofia/tts.py`); voices `af_heart`, `af_bella`, `bf_emma` | CPU |
| `pocket-*` | Kyutai Pocket TTS (100M); built-in voices `jane`, `eve` | CPU |
| `chatterbox-turbo` | Resemble Chatterbox-Turbo (350M), built-in voice; `[laugh]`, `[sigh]` and other tags | GPU (2.9 GB) |
| `chatterbox-nano` | Its smaller sibling, same tags | GPU (1.9 GB) |

Measured on this laptop (RTF = time to speak ÷ length of the audio):

| | GPU | CPU | Next to Sofia's LLM |
|---|---|---|---|
| Kokoro | – | 0.25–0.3 | fits (CPU) |
| Pocket | – | 0.25 | fits (CPU) |
| Chatterbox-Turbo | 0.41 | 2.2–2.4 | out of GPU memory |
| Chatterbox-Nano | 0.20 | 1.1–1.9 | out of GPU memory |

With the LLM loaded about 1 GB of VRAM is left, so Chatterbox only works if the LLM leaves the GPU.

## Run

```sh
./scripts/sofia.sh setup              # Sofia's .venv and the Kokoro model (Kokoro runs there)
bench/tts/setup.sh                    # venvs for pocket and chatterbox under ~/.local/share/sofia/tts-bench
bench/tts/run.sh                      # every voice (or name engines: run.sh kokoro pocket)
bench/tts/with-sofia.sh               # can the Chatterbox variants run next to the LLM?
.venv/bin/python bench/tts/check.py   # Whisper intelligibility check (CER)
python3 bench/tts/serve.py            # http://127.0.0.1:8766  (--host 0.0.0.0 for the phone)
```

`run.sh` unloads every Ollama model before Chatterbox, so it starts from an empty GPU, then repeats
three lines on the CPU. Ratings save to `$SOFIA_BENCH_DIR/ratings.json`. To switch Sofia to another
Kokoro voice, set `SOFIA_VOICE` (e.g. `SOFIA_VOICE=bf_emma ./scripts/sofia.sh`).
