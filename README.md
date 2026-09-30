# Sofia

A voice-assisted, **Hinglish** AI agent with **emotional intelligence** — it senses your tone,
responds empathetically, and modulates its own voice (pitch/emotion) to fit the moment. Runs locally
on your laptop, reachable from your phone.

> 🚧 Early scaffolding — no application code yet. This repo is set up and ready for development.

## Vision

Sofia is a personal, emotionally-aware AI agent built around the **Qwen3.5-4B-Uncensored** model,
running entirely on your own laptop. You talk to it by voice in Hinglish (mixed Hindi + English).
It listens for *how* you say things, not just *what* you say, and it talks back with matching feeling.

- 🎙️ **Voice-first** — speak to Sofia, hear it reply
- ❤️ **High EQ** — recognizes your emotion (angry / friendly / sad / …) and responds with empathy
- 🎭 **Expressive voice** — controls its own pitch & tone (warm, firm, playful, …) per situation
- 🌐 **Hinglish** — understands and speaks natural mixed Hindi-English
- 💻 **Local-first** — everything runs on your laptop; no cloud bills, full privacy
- 📱 **Cross-device** — laptop is the server; phone connects over a private tunnel

## The affective loop

```
  🎙️ your voice
      │
      ├──► STT  ─────────────► text (what you said)
      │
      └──► SER  ─────────────► your emotion (angry / friendly / sad / ...)
                                    │
                text + your-emotion ▼
                          ┌───────────────────┐
                          │  Qwen3.5-4B (EQ)   │  empathetic reply +
                          │  system-prompted   │  a style tag for its OWN voice
                          └─────────┬──────────┘
                    reply + style   │  e.g. {emotion:"warm", pitch:"soft"}
                                    ▼
                          ┌───────────────────┐
                          │  Emotional TTS     │  renders speech with that
                          │  (pitch/emotion)   │  pitch & emotion
                          └─────────┬──────────┘
                                    ▼
                          🔊 Sofia's voice (matched to the moment)
```

Three distinct capabilities:
1. **Recognize your tone** — Speech Emotion Recognition (SER) reads prosody/pitch/energy from your audio.
2. **Respond with EQ** — the LLM is prompted with your detected emotion; replies empathetically and emits a style tag for its own voice.
3. **Control its own voice** — expressive TTS renders the reply with the chosen pitch/emotion.

## Hardware target (this laptop)

Sofia is tuned for the specific machine it runs on:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3050 **6 GB** Laptop (GA107) |
| CPU | Intel i5-12450HX, 8C / 12T (12th gen) |
| RAM | 15 GB |
| Disk | `/home` on ext4 with **214 GB free** — models, venvs & HF cache all live here |
| OS | Kali GNU/Linux Rolling |

> Note: a second 931 GB NVMe drive exists but is all **NTFS** — avoid it for models/venvs
> (NTFS breaks symlinks & POSIX perms that HuggingFace caches and venvs rely on). Everything
> stays under `/home`.

### VRAM budget (6 GB is the constraint)

Everything competes for 6 GB, so the split is deliberate:

| Component | Runs on | Footprint |
|-----------|---------|-----------|
| Qwen3.5-4B — **Q4_K_M** quant | GPU | ~3.0–3.5 GB |
| STT (faster-whisper) | **CPU** (frees VRAM) | 0 GB VRAM |
| Emotional TTS (small model) | GPU | ~1.5–2 GB |
| KV cache / context | GPU | remainder |

Q4 is the right quant — Q5+ won't leave room for TTS. Expect ~25–35 tok/s once CUDA is working.

### ⚠️ Prerequisite: NVIDIA driver + CUDA

`nvidia-smi` is currently missing — the GPU driver/CUDA is **not installed**. Until it is, inference
falls back to CPU (slow). **Installing the NVIDIA driver + CUDA on Kali is step 0.**

## Decisions so far

| Area | Choice |
|------|--------|
| Model | Qwen3.5-4B-Uncensored (Q4_K_M) |
| Hosting | Local on this laptop (no AWS — free tier can't run a 4B model) |
| Phone access | Private tunnel (e.g. Tailscale) to the laptop |
| Interface | Voice-assisted, Hinglish, emotionally aware |

## Still to decide

- **LLM runtime:** Ollama vs. llama.cpp vs. vLLM
- **STT:** faster-whisper model size for Hinglish (base / small / medium)
- **SER:** which speech-emotion model (wav2vec2 / SpeechBrain), and its emotion label set
- **Emotional TTS:** the crux — XTTS-v2 (multilingual, incl. Hindi) vs. StyleTTS2 / Parler-TTS (expressive, English-first). Getting expressive *Hinglish* out is the hardest problem.
- **Phone client:** web app (PWA) vs. native Android vs. chat-bot

### Known-hard problems (be realistic)
- **Sarcasm** — hardest to both detect (it's context, not just prosody) and to synthesize. Treat as a stretch goal, not v1.
- **Expressive code-switched Hinglish TTS** — the frontier; most expressive TTS is English-only.
- **Live loop on 6 GB VRAM** — needs careful sequencing so SER + STT + LLM + TTS don't peak at once. (Disk/RAM are fine; VRAM is the only tight resource.)

## Getting Started

_TBD — setup lands once the runtime and voice stack are chosen. Step 0 will be NVIDIA/CUDA setup._

## License

See [LICENSE](LICENSE).
