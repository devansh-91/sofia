# Sofia

A voice-assisted, **Hinglish** AI agent with **emotional intelligence** — it senses your tone,
responds empathetically, and modulates its own voice (pitch/emotion) to fit the moment. Runs locally
on your laptop, reachable from your phone.

> 🚧 Early days — text chat with Sofia works (`python3 -m sofia.chat`); voice (STT / emotion / TTS) and the phone client are not built yet.

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
| Qwen3.5-4B — **Q4_K_M** quant | GPU | ~3.9 GB measured (planned 3.0–3.5) |
| STT (faster-whisper) | **CPU** (frees VRAM) | 0 GB VRAM |
| Emotional TTS (small model) | GPU | ~1.5–2 GB |
| KV cache / context | GPU | remainder |

Q4 is the right quant — Q5+ won't leave room for TTS. Measured speed is ~43–48 tok/s.

### ⚠️ Prerequisite: NVIDIA driver + CUDA

Inference needs the proprietary NVIDIA driver; on `nouveau` there is no CUDA and everything falls
back to CPU (slow). **Step 0 below installs it** — on this laptop it is done: driver 550.163.01
(CUDA 12.4), with kernel modesetting on so GNOME/Wayland runs on the NVIDIA card.

## Decisions so far

| Area | Choice |
|------|--------|
| Model | [HauhauCS Qwen3.5-4B-Uncensored "Aggressive"](https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive) (Q4_K_M), wrapped as the `sofia` Ollama model |
| LLM runtime | Ollama (systemd service on `127.0.0.1:11434`, CUDA backend) |
| Hosting | Local on this laptop (no AWS — free tier can't run a 4B model) |
| Phone access | Private tunnel (e.g. Tailscale) to the laptop |
| Interface | Voice-assisted, Hinglish, emotionally aware |

## Still to decide

- **STT:** faster-whisper model size for Hinglish (base / small / medium)
- **SER:** which speech-emotion model (wav2vec2 / SpeechBrain), and its emotion label set
- **Emotional TTS:** the crux — XTTS-v2 (multilingual, incl. Hindi) vs. StyleTTS2 / Parler-TTS (expressive, English-first). Getting expressive *Hinglish* out is the hardest problem.
- **Phone client:** web app (PWA) vs. native Android vs. chat-bot

### Known-hard problems (be realistic)
- **Sarcasm** — hardest to both detect (it's context, not just prosody) and to synthesize. Treat as a stretch goal, not v1.
- **Expressive code-switched Hinglish TTS** — the frontier; most expressive TTS is English-only.
- **Live loop on 6 GB VRAM** — needs careful sequencing so SER + STT + LLM + TTS don't peak at once. (Disk/RAM are fine; VRAM is the only tight resource.)

## Getting Started

### Step 0 — GPU driver (do this first)

Kali ships the open `nouveau` driver (no CUDA). Switch to NVIDIA's proprietary driver so Sofia can
use the RTX 3050. Secure Boot is off and matching kernel headers are available, so this is a clean
install.

```bash
sudo ./scripts/setup-gpu.sh   # installs headers + dkms + NVIDIA 550, blacklists nouveau, enables KMS
sudo reboot
./scripts/verify-gpu.sh       # after reboot: confirms driver, KMS, nvidia-smi + free VRAM
```

This laptop is **dGPU-only** (no Intel iGPU on the PCI bus), so the NVIDIA card drives the panel and
GNOME/Wayland needs `nvidia-drm modeset=1`. `setup-gpu.sh` sets it; if the driver was installed
without it and GDM loops on a black screen, boot the recovery entry and run:

```bash
sudo ./scripts/fix-kms.sh     # writes /etc/modprobe.d/nvidia-kms.conf, rebuilds initramfs
sudo reboot
```

### Performance / power tuning

Sofia can push the laptop hard while it's running. Use max performance on AC, ease off on battery:

```bash
sudo ./scripts/performance-mode.sh on      # force max performance (CPU + GPU)
sudo ./scripts/performance-mode.sh auto     # performance on AC, balanced on battery
sudo ./scripts/performance-mode.sh off      # back to balanced
     ./scripts/performance-mode.sh status   # inspect current state (no sudo)
```

The agent will call `performance-mode.sh auto` on start and `off` on stop. No power-profiles
daemon is running, so these settings apply directly.

### Step 1 — LLM runtime (Ollama)

```bash
curl -fsSL https://ollama.com/install.sh | sh   # run in a real terminal (needs sudo)
journalctl -u ollama -b | grep 'inference compute'   # must say library=CUDA, not cpu
```

If the log shows `library=cpu` / `total_vram="0 B"` even though `nvidia-smi` works, the Ollama
install is stale or incomplete (its `cuda_v12` libs missing or unreadable) — re-run the installer.

### Step 2 — Sofia's model

```bash
ollama pull hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M   # 3.4 GB
ollama create sofia -f models/Sofia.Modelfile    # adds persona, examples + sampling params
python3 -m sofia.chat                            # chat in the terminal; "bye" to exit
```

`models/Sofia.Modelfile` holds the persona (short spoken replies, unfiltered) plus a set of example
exchanges; re-run `ollama create` after editing it. **Always run with thinking off** — with it on,
the model loops over drafts until the context is full and never answers. Via the API that is
`"think": false` (`sofia/chat.py` sets it); with plain `ollama run sofia` pass `--think=false`.

**Language:** Sofia replies in English to English and in Roman-script Hinglish to Hindi/Hinglish
(Roman or Devanagari). The 4B model can't pick the language reliably by itself, so `sofia/lang.py`
detects it per message and `sofia/chat.py` appends an explicit reply instruction — plain
`ollama run sofia` skips that step and mostly answers in Hinglish. Very short messages ("haha ok")
keep the previous language.

**Hinglish quality** comes from three things in the Modelfile: temperature 0.3, *no* repeat penalty
(it punishes the hai/ka/ko words Hindi repeats constantly), and the example exchanges. On top of
that, `sofia/polish.py` cleans each reply in code: "bhai" used as an address becomes "yaar",
masculine first-person verbs become feminine ("aata hoon" → "aati hoon"), and markdown, emojis and
leaked bracket instructions are stripped. Replies are coherent most of the time, but a 4B model
still produces the odd phrase that doesn't make sense, and its jokes are weak.

**Active learning:** Sofia learns while you talk and carries it into every later conversation.
The model's weights do not change (yet) — after each of your messages a second small prompt
(`sofia/memory.py`) works out what to keep, saves it to `~/.local/share/sofia/memory.json`, and
`sofia/chat.py` feeds it back into later prompts. It costs about 0.3–1 s per message and no extra
VRAM (same weights). She prints `(yaad rakha: ...)` whenever she stores something.

| She learns | From | How it is used |
|---|---|---|
| Facts about you | "Mera ek kutta hai, Bruno." | Background for every reply; a newer fact replaces the one it outdates |
| How you behave | A reflection over every 6 exchanges (and at exit) | "User copes with work stress by asking for roasts." shapes how she treats you |
| Your moods and habits | Mood of each message, time of day, language — counted in code | "User usually chats late at night.", "Lately the user has often been stressed." (needs 10+ messages over 3+ days) |
| Things you teach her | "Mumbai mein 'bantai' ka matlab dost hota hai." | Background for every reply |
| Standing rules | "Mujhe bhai mat bola kar.", "Stop giving advice." | Attached to every message you send |
| Replies you liked | You praise a reply ("haha mast tha"), or type `/good` | Saved as an example, shown to her again when you say something similar |
| Replies you corrected | `/say <the reply she should have given>` | Same — your version replaces hers |

- While she knows fewer than 12 facts about you, she asks a getting-to-know-you question every few turns.
- Moods, one-off requests ("roast me") and things that happened today are not stored; upcoming
  events are stored with a real date.
- Behaviour notes are the model's reading of you from a few messages, so some will be off or
  overstated — delete the wrong ones.
- `/memory` lists everything she has learned, `/forget 3` removes item 3, `/forget all` wipes it.
  A wrong standing rule bends every reply, so check `/memory` if she starts acting odd.
- Limits: 40 facts, 12 behaviour notes, 20 taught items, 15 rules, 40 examples; oldest dropped first. Dates beyond
  "tomorrow" are worked out by the model and can be wrong, and she does not yet notice when a dated
  event has passed.

**Conversation log (for real fine-tuning later):** every exchange is appended to
`~/.local/share/sofia/conversations.jsonl` (owner-readable only), with your praise, criticism and
corrections recorded against the turn. Nothing trains on it yet.

```bash
python3 -m sofia.log                     # how many exchanges have been collected
python3 -m sofia.log export train.jsonl  # chat-format examples: corrections applied, disliked replies dropped
```

```bash
python3 -m unittest discover tests               # language, clean-up, memory + log tests
```

Measured on this laptop (4096-token context): ~43–48 tok/s, 100% GPU, ~4.0 GB VRAM in use with the
model loaded, ~1.9 GB free for TTS. That is above the 3.0–3.5 GB budgeted above, so TTS has to fit
in ~1.9 GB or run on CPU.

Tried and rejected: stock `qwen3.5:4b` (refuses), Dolphin 3.0 Llama3.2-3B (fits easily but doesn't
understand Hinglish). Ruled out on size without testing: Dolphin 3 8B (4.9 GB download).

### Remaining setup

_TBD — voice stack (STT/SER/TTS), the agent loop, and phone client land next._

## License

See [LICENSE](LICENSE).
