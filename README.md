# Sofia

A voice-assisted, **Hinglish** AI agent that runs locally on your laptop and is reachable from your phone.

> 🚧 Early scaffolding — no application code yet. This repo is set up and ready for development.

## Vision

Sofia is a personal AI agent built around the **Qwen3.5-4B-Uncensored** model, running entirely on
your own machine. You talk to it by voice in Hinglish (mixed Hindi + English), from your Linux laptop
or your phone.

- 🎙️ **Voice-first** — speak to Sofia, hear it reply
- 🌐 **Hinglish** — understands and speaks natural mixed Hindi-English
- 💻 **Local-first** — the model runs on your laptop; no cloud bills, full privacy
- 📱 **Cross-device** — laptop is the server; phone connects over a private tunnel

## Architecture (planned)

```
                 ┌──────────────────────────────────────────┐
                 │           Linux laptop (server)            │
                 │                                            │
  Phone ──tunnel─┤  ┌─────────┐   ┌─────────┐   ┌──────────┐  │
  (voice client) │  │  STT    │→  │  Sofia  │→  │   TTS    │  │
                 │  │(Hinglish)│   │  agent  │   │(Hinglish)│  │
                 │  └─────────┘   └────┬────┘   └──────────┘  │
                 │                     │                      │
                 │              ┌──────▼───────┐              │
                 │              │ Qwen3.5-4B    │              │
                 │              │ (local LLM)   │              │
                 │              └──────────────┘              │
                 └──────────────────────────────────────────┘
```

## Decisions so far

| Area | Choice |
|------|--------|
| Model | Qwen3.5-4B-Uncensored |
| Hosting | Local on the laptop (no AWS — free tier can't run a 4B model) |
| Phone access | Private tunnel (e.g. Tailscale) to the laptop |
| Interface | Voice-assisted, Hinglish |

## Still to decide

- **Phone client:** web app (PWA) vs. native Android vs. chat-bot
- **LLM runtime:** Ollama vs. llama.cpp vs. vLLM
- **Voice stack:** which STT + TTS handle Hinglish well

## Getting Started

_TBD — setup instructions land once the runtime and client are chosen._

## License

See [LICENSE](LICENSE).
