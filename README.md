# Sofia

A voice-assisted AI companion with **emotional intelligence** — it senses your tone, responds
empathetically, and shapes its own voice to fit the moment. Talk to it in English or Hinglish; it
answers in natural spoken English and calls you "sir". Runs locally on your laptop, reachable from
your phone.

> 🚧 Working end to end, roughly: `./scripts/sofia.sh` starts text + voice chat in the browser (laptop or phone). Her voice is Chatterbox-Turbo speaking a clone of Kokoro's `af_heart`: she really laughs, chuckles, sighs and gasps. Kokoro on the CPU is the fallback.

## Vision

Sofia is a personal, emotionally-aware AI agent built around the **Qwen3.5-4B-Uncensored** model,
running entirely on your own laptop. You talk to it by voice, in English or Hinglish; it answers in English.
It listens for *how* you say things, not just *what* you say, and it talks back with matching feeling.

- 🎙️ **Voice-first** — speak to Sofia, hear it reply
- ❤️ **High EQ** — recognizes your emotion (angry / friendly / sad / …) and responds with empathy
- 🎭 **Expressive voice** — controls its own pitch & tone (warm, firm, playful, …) per situation
- 🌐 **Understands Hinglish** — write or speak mixed Hindi-English; Sofia always answers in English
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
| Qwen3.5-4B — **Q4_K_M** quant | GPU | ~2.9 GB measured, without the vision projector it ships with (~0.9 GB) |
| STT (faster-whisper) | **CPU** (frees VRAM) | 0 GB VRAM |
| SER (wav2vec2) | **CPU** | 0 GB VRAM |
| TTS (Chatterbox-Turbo, half precision, split GPU / CPU) | GPU | ~1.4 GB (the whole model in full precision: 3.3 GB) |
| TTS fallback (Kokoro-82M, ONNX) | **CPU** | 0 GB VRAM |
| KV cache / context | GPU | included above |

With the desktop's ~1.3 GB, everything together uses ~5.6 of the 6 GB.
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
| STT | faster-whisper `small`, int8, on CPU; chooses between Hindi and English per utterance |
| SER | `superb/wav2vec2-base-superb-er` on CPU (neutral / happy / angry / sad) |
| TTS | Chatterbox-Turbo on the GPU, cloned from Kokoro's `af_heart` (`SOFIA_VOICE`); Kokoro-82M on CPU as fallback (`SOFIA_TTS=kokoro` to use only it) |
| Phone client | Web app served by `sofia/server.py`, protected by an access key |
| Phone access | Private tunnel (e.g. Tailscale) to the laptop |
| Interface | Voice-assisted, emotionally aware; understands English + Hinglish, answers in English, calls the user "sir" |

## Still to decide

- **Emotional TTS:** Turbo makes real sounds but has no emotion or pace control, so the tone comes
  from her words and sounds alone (the mood still sets Kokoro's pace). `bench/tts` compares voices by ear.
- **SER for Hinglish:** the current model is trained on acted English and is unverified on real
  Hinglish speech; emotion2vec is the likely replacement.
- **STT speed:** ~3–4 s per utterance on CPU. Moving Whisper to the GPU would cut that, at ~0.6 GB VRAM.

### Known-hard problems (be realistic)
- **Sarcasm** — hardest to both detect (it's context, not just prosody) and to synthesize. Treat as a stretch goal, not v1.
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
./scripts/sofia.sh setup                         # pulls the weights (3.4 GB), builds sofia-base + sofia
python3 -m sofia.chat                            # chat in the terminal; "bye" to exit
```

`models/Sofia.Modelfile` holds the persona (short spoken replies, unfiltered) plus a set of example
exchanges; re-run `ollama create` after editing it. **Always run with thinking off** — with it on,
the model loops over drafts until the context is full and never answers. Via the API that is
`"think": false` (`sofia/chat.py` sets it); with plain `ollama run sofia` pass `--think=false`.

**Language:** Sofia always answers in English and calls the user "sir", whether you write in
English or Hinglish (Roman or Devanagari). Left to itself the 4B model answers in whatever language
it was spoken to, so `sofia/chat.py` appends an explicit reply instruction to every message; if a
reply still comes back in Hinglish (`sofia/lang.py` spots it) she is asked again, and as a last
resort it is translated. Plain `ollama run sofia` skips all of that.

**Reply quality** comes from the persona and the example exchanges in the Modelfile — several are
Hinglish questions with English answers, which is what keeps her in English — with Qwen's
recommended non-thinking sampling. `sofia/polish.py` strips markdown, emojis and leaked bracket
instructions, since every reply is spoken aloud. A 4B model still produces the odd reply that misses
the mood, and its jokes are weak.

**Sounds:** she may write `[laugh]`, `[chuckle]`, `[sigh]` or `[gasp]` where the sound belongs, and
her voice makes it. The Modelfile asks for them rarely and never when you are hurt; in a test 5 of
14 replies had one, mostly well placed. `sofia/polish.py` turns stage directions like `*laughs*`
into tags and drops any other bracketed text.

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

Measured on this laptop (4096-token context): ~43–48 tok/s, 100% GPU. The source model also loads a
vision projector Sofia never uses, so `scripts/sofia.sh` builds `sofia-base` from the same weights
without it (~2.9 GB instead of ~3.8 GB in VRAM) and `sofia` on top of that. The memory prompts run
on `sofia-base`, which Ollama serves from the same loaded copy.

Tried and rejected: stock `qwen3.5:4b` (refuses), Dolphin 3.0 Llama3.2-3B (fits easily but doesn't
understand Hinglish). Ruled out on size without testing: Dolphin 3 8B (4.9 GB download).

### Step 3 — Voice and phone

```bash
./scripts/sofia.sh setup     # .venv + speech stack, Turbo + Kokoro voices, STT + emotion models, sofia model
./scripts/sofia.sh           # start everything: web app with text + voice
./scripts/sofia.sh install   # add a Sofia icon to the app menu and Desktop (click it to open her)
./scripts/sofia.sh app       # what the icon runs: her own window, server started and stopped with it
./scripts/sofia.sh voice     # or talk in the terminal: Enter, speak, Enter
./scripts/sofia.sh chat      # text only, in the terminal
./scripts/sofia.sh test      # unit tests
```

One voice turn runs the whole affective loop:

```
mic → faster-whisper (what you said) ─┐
    → wav2vec2 SER  (how you sounded) ─┴→ Sofia (LLM + memory) → reply
reply → Chatterbox-Turbo, sounds included (or Kokoro, pace for the mood) → speaker
```

- **What you said:** Hindi comes back in Devanagari and English in English; the chat handles both.
- **How you sounded:** the mood heard in your voice is passed to her as `[The user sounds sad.]` and
  overrides the mood guessed from the words. Below 70 % confidence it is treated as neutral.
- **Her voice:** Chatterbox-Turbo (`sofia/chatterbox_worker.py`, its own venv and process, ~1.4 GB
  VRAM, about 4x faster than real time) laughs or sighs where her reply has a sound tag. When it
  is not installed or fails, Kokoro speaks instead: soft (slower) when you are sad, lonely or
  tired; calm when you are stressed, anxious or angry; bright (quicker) when you are happy;
  playful when you are bored.
- **Timing:** about 9 s from the end of your speech to her reply on this laptop (STT ~4 s).
- **Microphone level:** at the default input volume this laptop's mic clips on room noise alone and
  nothing is recognised. `wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.2` fixes it; the voice loop
  warns when a recording is clipped.

**Phone:** the web app listens on `127.0.0.1:8765` and prints a link containing an access key
(stored in `~/.local/share/sofia/access-key`). Every API call needs that key — this is an
unfiltered model with your personal memory behind it. To reach it from the phone, put both devices
on Tailscale and let it provide https, which the phone's browser needs before it allows the mic:

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up   # once, on the laptop
sudo tailscale serve --bg 8765        # https://<laptop>.<tailnet>.ts.net → the web app
```

Then open `https://<laptop>.<tailnet>.ts.net/?key=<access key>` on the phone. Without https
(e.g. `./scripts/sofia.sh serve --host 0.0.0.0` on home Wi-Fi) text chat and her voice still work,
but the mic button stays hidden.

## License

See [LICENSE](LICENSE).
