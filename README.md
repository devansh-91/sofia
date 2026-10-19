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
| Qwen3.5-4B — **Q4_K_M** quant, 8192-token context | GPU | ~3.0 GB measured, without the vision projector it ships with (~0.9 GB) |
| STT (faster-whisper) | **CPU** (frees VRAM) | 0 GB VRAM |
| SER (wav2vec2) | **CPU** | 0 GB VRAM |
| TTS (Chatterbox-Turbo, half precision, split GPU / CPU) | GPU | ~1.4 GB (the whole model in full precision: 3.3 GB) |
| TTS fallback (Kokoro-82M, ONNX) | **CPU** | 0 GB VRAM |
| KV cache / context | GPU | included above |

With the desktop's ~1.3 GB, everything together uses ~5.7 of the 6 GB.

System RAM matters too: loading Turbo briefly takes ~3 GB of RAM. When the laptop is short of RAM
(many browser tabs, swap full), the kernel kills Turbo's worker, and Sofia says so and speaks with
Kokoro until she is restarted.
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

- **Doing in normal mode:** changes (files, commands) are only offered in work mode, behind an Allow
  button. Her 4B model in normal mode only looks.
- **Emotional TTS:** Turbo has no emotion setting. Her tone for your mood comes from five clones of
  her voice whose references differ in pace, pitch and tone (see "Her voice" below), plus her words
  and sounds. Whether they sound sad or happy enough is for the ear: `bench/tts` has the old and new
  clones side by side (sofia-moods-old / sofia-moods-new).
- **Fine-tuning:** the pipeline works up to the data (see "Fine-tuning" below); whether the tuned
  model beats the prompted one is still to be measured.
- **SER for Hinglish:** the current model is trained on acted English and is unverified on real
  Hinglish speech; emotion2vec is the likely replacement.

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
exchanges; `./scripts/sofia.sh` rebuilds `sofia` whenever it changes. **Always run with thinking off** — with it on,
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
into tags and drops any other bracketed text. Once your mood is read, it also drops the sounds that
don't fit it (none when you are sad, lonely, anxious or angry; only a sigh when you are stressed
or tired) and keeps at most one per reply.

**Situational awareness:** `sofia/situation.py` tells her about the moment:
- the day, date and time, and fixed-date occasions such as Valentine's Day;
- whether you are on your phone;
- for the first few exchanges, the end of your last conversation, so she can ask how yesterday's
  presentation went.

Each of these is noted only once, on the newest message, when it is worth reacting to:
- you are back after hours or days;
- it is the middle of the night;
- the laptop she runs on is about to die.

When you open the web app, or `chat` or `voice`, she says hello first, fitted to all of that. She
skips the hello if you were there in the last 10 minutes.

**A little possessive:** she likes being your favourite. If you praise ChatGPT, talk to someone else
all night, or vanish for days, she pouts for one line, then helps as usual. The Modelfile keeps this
a running joke. She never guilt-trips you, never objects to your friends, family or plans, and never
does it when you are down. In a test run:
- She asked "did you pass?" about the previous day's driving test as her hello.
- She replied "ChatGPT got lucky, sir, but I'm still here when you need to talk."
- She stayed purely supportive about a breakup.

**Looking things up:** she can look, never touch. `sofia/tools.py` gives her these tools, through
Ollama's tool calling:
- list a folder (with counts per file type, and the biggest and newest file), find files by name, and
  read text files;
- search the web, read a page, get the news (Google News headlines, also for scores and results),
  and get the weather (Open-Meteo).

Two limits hold whatever the model asks for:
- Only the home folder is reachable, never hidden files or folders (keys, configs, browser data,
  her memory), and never through a symlink pointing out of bounds.
- No tool changes, moves or deletes anything, opens anything or runs a program. Asked to, she says
  she can't.

Three things keep the 4B model from making answers up:
- A question about the news, prices, the weather or your files gets a note telling her to look first.
  If she answers without looking anyway, she is sent back once.
- A web search also reads the top result's page, because snippets seldom hold the number asked for.
  It passes on the part of that page that best matches the question, since most pages start with menus.
- If her answer has a number that is in nothing she looked up, she gets one more try and is told
  which numbers those are.

In tests she then said "I couldn't find the exact rate just now" instead of inventing a petrol price.
A lookup adds ~2–6 s to a reply.

**Work mode ("kaam mode"):** say or type "kaam mode" (or flip the *kaam* switch) for code and real
work; "normal mode" brings her back. Her 4B model and Turbo leave the GPU to Qwen2.5-Coder 7B
(`SOFIA_WORK_MODEL`; ~29 tokens/s here, against ~15 for Qwen3.5 9B, which spills 37 % onto the CPU).
Switching takes ~9 s. She then answers with whole code blocks (shown with a copy button), reads your
files and the web to get it right, and says only the words around the code, in Kokoro's voice.
Nothing from work mode goes into her memory or conversation log.

In work mode she can also **change things**: write a file, make a folder, move, delete, run a
command. She only proposes them (`sofia/actions.py`); each shows up as a card with exactly what it
will do, the file's new content or the command, and happens only when you press **Allow**:
- paths follow the lookup rules (home folder only, nothing hidden); bare names land in
  `~/sofia-workspace`;
- delete means the Trash (`gio trash`), so it can be undone;
- commands run in a folder of yours for at most a minute, never with sudo, su or pkexec;
- a proposal runs once: a second Allow for it is refused.

**The agent** (`sofia/agent.py`): in work mode, "agent, make a todo app and test it" hands the whole
task to [OpenCode](https://opencode.ai) (MIT), a coding agent that reads the project, edits files and
runs commands until it is done. Sofia runs it headless in `~/sofia-workspace` on her own local weights
(sofia-base with a 16k context, thinking off), so nothing leaves the laptop. Every edit and command it
wants becomes a card with the diff or the command, and goes ahead only on Allow. A two-step task
(write a script, run it) took ~25 s. Qwen2.5-Coder can't drive it (it writes tool calls as text), and
Qwen3.5 with thinking on spent five minutes on one file.

The 7B model has habits the code works around: it writes tool calls as JSON text (parsed as calls),
says "I've proposed it" without proposing (sent back once to do it), and retells a command's output
from memory (her answer shows the real output instead).

**Your documents** (`sofia/docs.py`): notes, PDFs, Word files and code in `~/sofia-docs` (or added on
the settings page) are cut into passages with an embedding each, from nomic-embed-text on the CPU so
the GPU stays with her model and voice. Her `search_docs` tool finds the passages closest in meaning
to a question, and a question about notes or documents tells her to search them first. The index
follows the folder: new and changed files are read again, deleted ones dropped.

**Feelings, relationships, a crisis** (`sofia/care.py`): when a message touches mental health,
relationships or emotional intelligence, a note with a counsellor's tools rides on it: hear the
feeling first, then one tool that fits (slow breathing, grounding, checking the worst thought,
"I feel ... when ...", naming the feeling under anger). She is a friend with those skills, not a
psychiatrist: no diagnoses, no medicine, and a counsellor or Tele-MANAS when it has gone on long.
Words of self-harm or suicide override everything: no jokes, sounds or possessiveness, she asks
whether you are safe, and the page shows Tele-MANAS (14416, free, 24x7, Hindi and English) and 112.

**Web search** goes to a private SearXNG on this laptop when one is running. Start it with
`./scripts/searxng.sh`; it needs Docker and pulls a few hundred MB once. Through it, Google and Brave
answer in under a second; she then said "West Indies won by 5 wickets" and "Petrol in Delhi is ₹102.12
per litre", both matching the source pages. Without SearXNG she uses
Brave Search's web page, which blocks scripts after a few searches in a row. While Brave is blocking,
she falls back to news headlines and Wikipedia, and says when they don't hold the answer. DuckDuckGo,
Mojeek and Ecosia show scripts a captcha, and Bing returns unrelated results to them, so they aren't
used.

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

Measured on this laptop: ~43–48 tok/s, 100% GPU. The context is 8192 tokens: this model keeps a KV
cache in only a few of its layers, so that costs ~0.14 GB more VRAM than 4096, and her prompt with
persona, examples and tools is already ~2.7k. The source model also loads a
vision projector Sofia never uses, so `scripts/sofia.sh` builds `sofia-base` from the same weights
without it (~2.9 GB instead of ~3.8 GB in VRAM) and `sofia` on top of that. The memory prompts run
on `sofia-base`, which Ollama serves from the same loaded copy.

Tried and rejected: stock `qwen3.5:4b` (refuses), Dolphin 3.0 Llama3.2-3B (fits easily but doesn't
understand Hinglish). Ruled out on size without testing: Dolphin 3 8B (4.9 GB download).

### Step 3 — Voice and phone

```bash
./scripts/sofia.sh setup     # .venv + speech stack, Turbo + Kokoro voices, STT + emotion models, sofia model
./scripts/sofia.sh           # start everything: web app with text + voice, and open her window
./scripts/sofia.sh install   # add a Sofia icon to the app menu and Desktop (click it to open her)
./scripts/sofia.sh app       # what the icon runs: her own window, server started and stopped with it
# Her window opens in Google Chrome (then Chromium, then Brave), whatever the default browser is;
# SOFIA_BROWSER=chromium picks another.
./scripts/sofia.sh voice     # or talk in the terminal: Enter, speak, Enter
./scripts/sofia.sh chat      # text only, in the terminal
./scripts/sofia.sh test      # unit tests
./scripts/searxng.sh         # optional: private web search for her lookups (Docker)
```

One voice turn runs the whole affective loop:

```
mic → faster-whisper (what you said) ─┐
    → wav2vec2 SER  (how you sounded) ─┴→ Sofia (LLM + memory) → reply
reply → Chatterbox-Turbo in the style for your mood, sounds included (or Kokoro) → speaker
```

- **What you said:** Hindi comes back in Devanagari and English in English; the chat handles both.
- **How you sounded:** the mood heard in your voice is passed to her as a guess that is often wrong
  (the model often hears "happy" in a flat or disappointed voice, and she then said "glad to hear" to
  a wasted weekend). The mood read from your words wins. Below 70 % confidence it is not passed on.
- **Her voice:** Chatterbox-Turbo (`sofia/chatterbox_worker.py`, its own venv and process, ~1.4 GB
  VRAM, about 4x faster than real time) laughs or sighs where her reply has a sound tag. Its style
  follows your mood: soft when you are sad, lonely or tired (~12% slower); calm when you are
  stressed, anxious or angry (~6% slower); bright when you are happy (~5% quicker, livelier pitch);
  playful when you are bored. Turbo has no style setting but copies the pace, pitch and tone of
  the voice it clones, so each style is its own clone, of Kokoro reading a fitting line at a fitting
  pace, pitch-shifted and with its top end raised or cut (`TURBO_STYLES` in `sofia/tts.py`). Pace
  alone gave five clones with the same pitch and timbre; now soft is ~1 semitone lower and darker,
  bright and playful ~2 higher and brighter, all 0.87-0.94 similar to warm. When Turbo is not
  installed or fails, Kokoro speaks instead, in the same styles by pace alone.
- **Start-up:** the LLM loads while the speech models do, so her first reply takes ~5 s instead of
  ~20 s. It stays in VRAM for an hour after her last reply (`SOFIA_KEEP_ALIVE`; Ollama's default
  of 5 minutes made the next reply after a pause wait for it) and is unloaded when she stops.
- **Speech recognition** runs on the GPU (~0.34 GB VRAM, ~0.25 s a sentence; the CPU took ~2 s),
  through the cuBLAS / cuDNN wheels in `.venv`. Without room on the GPU it runs on the CPU.
- **Timing:** her reply is spoken in pieces cut at sentence ends: the first plays while the rest
  render, and what she learns from your message is worked out while she talks. In a call she starts
  ~2.5 s after you stop talking.
- **Streaming:** her answer is spoken while it is written. Each sentence goes to her voice as soon as
  the model finishes it (cleaned, with sounds fitted to your mood), unless the answer needs checking
  first: one held to looking something up, or one after a lookup, whose numbers are checked against
  what was found. A first sentence in Hinglish stops the stream and the answer is made the usual way.
- **Call mode:** the green phone button. Talk hands-free: she notices when you stop, answers, and
  listens again; talk over her and she stops. A "hmm" or "okay, sir" in her voice fills a wait of
  more than ~1 s (rendered once into `~/.local/share/sofia/voices/fillers`). Go on talking while she
  is thinking and she drops that answer and answers both parts together. The speech threshold
  follows the room's noise (the quieter end of the last 3 s, with the browser's automatic gain off),
  the meter shows it as a mark, and tapping her picture starts and sends by hand. A mic that stays
  silent, like a muted Bluetooth headset, gets a hint on how to unmute it.
- **Settings** (the gear): her voice, from nine packages (Turbo clones of Kokoro's Heart, Bella,
  Nicole and Emma, Turbo's own voice, and four light Kokoro voices on the CPU; a new Turbo voice is
  cloned the first time it is picked, in about a minute); her brain (original or tuned); what she
  knows about you, to read, add to or forget item by item; your documents; the chat history; and
  deleting all of it.
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

## Fine-tuning

`./scripts/finetune.sh` tunes her on this laptop, in four steps:

1. **Data** (`training/build.py`): the dialogues in `training/dialogues/*.jsonl`, the ones about your
   own life in `~/.local/share/sofia/training/dialogues` (kept out of this public repo), and every
   exchange you liked (`/good`) or corrected (`/say`). Each becomes a chat in exactly the shape she
   sees at run time: her system prompt, what she knows about you from her memory, the bracketed
   notes and care notes. The few-shot examples are left out: the tuning replaces them.
2. **Training** (`training/train.py`): a QLoRA (4-bit, rank 16) on Qwen/Qwen3.5-4B, on the MLP and
   attention-output layers only, with the loss on her replies only. It needs the GPU, so Sofia must
   be stopped.
3. **Merge** (`training/merge.py`): HauhauCS's uncensored model, which Sofia runs on, is only
   published as GGUF, so the LoRA is added to its BF16 GGUF tensor by tensor. Every tensor is first
   checked against the official one it was trained on. Ollama then quantizes it to Q4_K_M as
   `sofia-tuned-base`, and `sofia-tuned` is her persona on top of it.
4. **Compare** (`training/compare.py`): `sofia` and `sofia-tuned` side by side on held-out messages.

Try the tuned one with `SOFIA_MODEL=sofia-tuned SOFIA_BASE_MODEL=sofia-tuned-base ./scripts/sofia.sh`.

A ChatGPT data export can feed both her memory and the training: `training/chatgpt.py export.zip`
pulls out the messages you wrote (ChatGPT's replies are left out, they are not how she talks).

## License

See [LICENSE](LICENSE).
