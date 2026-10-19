#!/usr/bin/env bash
#
# sofia.sh — one entry point for everything.
#
#   ./scripts/sofia.sh            start everything: web app for laptop + phone, with voice
#   ./scripts/sofia.sh setup      install the speech stack, voices and the `sofia` model
#   ./scripts/sofia.sh chat       text chat in this terminal
#   ./scripts/sofia.sh voice      talk out loud in this terminal (push-to-talk)
#   ./scripts/sofia.sh serve ...  web app only; extra arguments go to sofia.server
#   ./scripts/sofia.sh test       run the unit tests
#
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_MODEL="${SOFIA_BASE_MODEL:-hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M}"
VOICE_DIR="${SOFIA_VOICE_DIR:-$HOME/.local/share/sofia/voices}"
# Kokoro's model and voice pack, with the sizes and SHA-256 sums they were first fetched with.
KOKORO_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
KOKORO_FILES=(
  "kokoro-v1.0.onnx 7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
  "voices-v1.0.bin bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
)
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx \033[0m %s\n' "$*" >&2; exit 1; }

# The speech stack lives in .venv; without it, fall back to system Python (text only).
python() { if [ -x .venv/bin/python ]; then .venv/bin/python "$@"; else python3 "$@"; fi; }

ensure_model() {
  curl -fsS "$OLLAMA_URL/api/version" >/dev/null 2>&1 \
    || die "Ollama is not answering at $OLLAMA_URL — start it with: sudo systemctl start ollama"
  if ! ollama list | grep -q '^sofia:'; then
    ollama list | grep -qF "$BASE_MODEL" || { log "Downloading the base model (3.4 GB)..."; ollama pull "$BASE_MODEL"; }
    log "Building the sofia model..."
    ollama create sofia -f models/Sofia.Modelfile
  fi
}

setup() {
  command -v uv >/dev/null || die "uv is required: https://docs.astral.sh/uv/"
  [ -x .venv/bin/python ] || { log "Creating .venv..."; uv venv --python 3.12 .venv; }
  log "Installing the speech stack (about 1.4 GB on disk)..."
  uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements-voice.txt
  mkdir -p "$VOICE_DIR"
  local entry file sum
  for entry in "${KOKORO_FILES[@]}"; do
    read -r file sum <<<"$entry"
    [ -f "$VOICE_DIR/$file" ] && continue
    log "Downloading Kokoro $file..."
    curl -fL --retry 3 -o "$VOICE_DIR/$file.part" "$KOKORO_URL/$file"
    echo "$sum  $VOICE_DIR/$file.part" | sha256sum -c --quiet - || die "$file failed its checksum"
    mv "$VOICE_DIR/$file.part" "$VOICE_DIR/$file"
  done
  log "Fetching the speech-to-text and emotion models (about 850 MB, first time only)..."
  .venv/bin/python -c "from sofia import stt, emotion; stt._load(); emotion._load()"
  ensure_model
  log "Rebuilding the sofia model from models/Sofia.Modelfile..."
  ollama create sofia -f models/Sofia.Modelfile
  log "Setup done. Start with: ./scripts/sofia.sh"
}

command="${1:-start}"
[ $# -gt 0 ] && shift

case "$command" in
  setup) setup ;;
  chat)  ensure_model; exec python3 -m sofia.chat ;;
  voice) ensure_model; [ -x .venv/bin/python ] || die "Voice needs the speech stack: ./scripts/sofia.sh setup"
         exec .venv/bin/python -m sofia.voice ;;
  start|serve)
         ensure_model
         [ -x .venv/bin/python ] || log "Speech stack not installed — text only. Run ./scripts/sofia.sh setup for voice."
         exec "$(if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)" -m sofia.server "$@" ;;
  test)  exec python3 -m unittest discover tests ;;
  *)     sed -n '3,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
