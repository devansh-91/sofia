#!/usr/bin/env bash
#
# sofia.sh — one entry point for everything.
#
#   ./scripts/sofia.sh            start everything: web app for laptop + phone, with voice; opens her window
#   ./scripts/sofia.sh app        open Sofia in her own window (starts her in the background if needed)
#   ./scripts/sofia.sh install    add a Sofia icon to the app menu and the Desktop
#   ./scripts/sofia.sh setup      install the speech stack, voices and the `sofia` model
#   ./scripts/sofia.sh chat       text chat in this terminal
#   ./scripts/sofia.sh voice      talk out loud in this terminal (push-to-talk)
#   ./scripts/sofia.sh serve ...  web app only; extra arguments go to sofia.server
#   ./scripts/sofia.sh test       run the unit tests
#
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE_MODEL="${SOFIA_SOURCE_MODEL:-hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M}"
# The source model without its vision projector, which Sofia never uses: ~0.9 GB of VRAM
# more for her voice. The sofia model is built FROM it (models/Sofia.Modelfile).
BASE_MODEL=sofia-base
# Work mode ("kaam mode"): the bigger model for code, swapped in for sofia and her laughing voice.
WORK_MODEL="${SOFIA_WORK_MODEL:-qwen2.5-coder:7b}"
# Her context in tokens, as in models/Sofia.Modelfile. 8192 costs only ~0.14 GB more VRAM than 4096
# (most of the model's layers keep no KV cache) and leaves room for what she looks up.
CONTEXT=8192
VOICE_DIR="${SOFIA_VOICE_DIR:-$HOME/.local/share/sofia/voices}"
# Kokoro's model and voice pack, with the sizes and SHA-256 sums they were first fetched with.
KOKORO_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
KOKORO_FILES=(
  "kokoro-v1.0.onnx 7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
  "voices-v1.0.bin bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"
)
# Chatterbox-Turbo, the voice that can laugh, lives in its own venv: its torch build clashes with .venv.
TURBO_VENV="${SOFIA_TURBO_VENV:-$HOME/.local/share/sofia/chatterbox}"
# cu126 wheels run on the 550 driver (CUDA 12.4) through minor-version compatibility.
CUDA_INDEX=https://download.pytorch.org/whl/cu126
CHATTERBOX_REV=5de7a54aa4e5e2baadb0182dde554908b48b85c2
OLLAMA_URL="${OLLAMA_HOST:-http://127.0.0.1:11434}"
PORT="${SOFIA_PORT:-8765}"
KEY_FILE="${SOFIA_KEY_FILE:-$HOME/.local/share/sofia/access-key}"
STATE_DIR="$HOME/.local/state/sofia"
ICON="$PWD/sofia/web/icon.svg"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
# Started from the app icon there is no terminal to read, so problems also become notifications.
die()  {
  printf '\033[1;31mxx \033[0m %s\n' "$*" >&2
  [ -t 2 ] || notify-send -i "$ICON" "Sofia" "$*" 2>/dev/null || true
  exit 1
}

# The speech stack lives in .venv; without it, fall back to system Python (text only).
python() { if [ -x .venv/bin/python ]; then .venv/bin/python "$@"; else python3 "$@"; fi; }

ensure_base() {
  # An older sofia-base had a 4096-token context; rebuild it with the one models/Sofia.Modelfile uses.
  ollama list | grep -q "^$BASE_MODEL:" && ollama show --parameters "$BASE_MODEL" | grep -qE "num_ctx +$CONTEXT\b" && return
  ollama list | grep -qF "$SOURCE_MODEL" || { log "Downloading the base model (3.4 GB)..."; ollama pull "$SOURCE_MODEL"; }
  # Its Modelfile has two FROM lines: the language model first, then the vision projector.
  local weights
  weights="$(ollama show --modelfile "$SOURCE_MODEL" | awk '/^FROM /{ print $2; exit }')"
  log "Building $BASE_MODEL (the model without its vision projector)..."
  # num_ctx matches models/Sofia.Modelfile, so Ollama runs both on one loaded copy.
  curl -fsS "$OLLAMA_URL/api/create" -o /dev/null -d "{\"model\": \"$BASE_MODEL\", \"stream\": false,
    \"files\": {\"model.gguf\": \"sha256:${weights##*sha256-}\"}, \"parameters\": {\"num_ctx\": $CONTEXT}}" \
    || die "Could not build $BASE_MODEL"
  ! ollama show "$BASE_MODEL" | grep -qw vision || die "$BASE_MODEL still has the vision projector"
}

ensure_model() {
  curl -fsS "$OLLAMA_URL/api/version" >/dev/null 2>&1 \
    || die "Ollama is not answering at $OLLAMA_URL — start it with: sudo systemctl start ollama"
  ensure_base
  # Rebuild when models/Sofia.Modelfile changed since the last build, or when an older sofia
  # was built with the vision projector.
  local stamp="$STATE_DIR/modelfile.sha256" sum
  sum="$(sha256sum models/Sofia.Modelfile | cut -d' ' -f1)"
  if ! ollama list | grep -q '^sofia:' || ollama show sofia | grep -qw vision || [ "$(cat "$stamp" 2>/dev/null)" != "$sum" ]; then
    log "Building the sofia model..."
    ollama create sofia -f models/Sofia.Modelfile
    # The tuned brain (scripts/finetune.sh) carries the same persona, without the few-shot examples.
    if ollama show sofia-tuned-base >/dev/null 2>&1; then
      local tuned; tuned="$(mktemp)"
      python3 scripts/tuned-modelfile.py >"$tuned"
      ollama create sofia-tuned -f "$tuned"
      rm -f "$tuned"
    fi
    mkdir -p "$STATE_DIR"
    echo "$sum" >"$stamp"
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
  if nvidia-smi >/dev/null 2>&1; then
    log "Installing Chatterbox-Turbo, the voice that can laugh (GPU; about 6 GB on disk)..."
    uv venv --quiet --allow-existing --python 3.11 "$TURBO_VENV"
    uv pip install --python "$TURBO_VENV/bin/python" --index-url "$CUDA_INDEX" torch==2.6.0 torchaudio==2.6.0
    uv pip install --python "$TURBO_VENV/bin/python" soundfile \
      "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@$CHATTERBOX_REV"
  else
    log "No NVIDIA GPU: Sofia will speak with Kokoro on the CPU."
  fi
  log "Fetching the speech-to-text, emotion and voice models (about 5 GB with Turbo, first time only)..."
  .venv/bin/python -c "from sofia import stt, emotion, tts; stt._load(); emotion._load(); tts.clone()"
  ensure_model
  log "Rebuilding the sofia model from models/Sofia.Modelfile..."
  ollama create sofia -f models/Sofia.Modelfile
  ollama list | grep -qF "$WORK_MODEL" || { log "Downloading the work-mode model (4.7 GB)..."; ollama pull "$WORK_MODEL"; }
  ollama list | grep -q '^nomic-embed-text' || { log "Downloading the model that reads your documents (0.3 GB)..."; ollama pull nomic-embed-text; }
  log "Setup done. Start with: ./scripts/sofia.sh"
}

server_up() { curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; }

# A Chromium-family browser can show Sofia as an app: her own window, no tabs or address bar.
# Chrome first, whatever the system's default browser is; SOFIA_BROWSER picks another.
app_browser() {
  local browser
  for browser in ${SOFIA_BROWSER:-} google-chrome chromium brave-browser; do
    command -v "$browser" >/dev/null && { echo "$browser"; return; }
  done
}

# Opens her window and returns when it is closed.
open_window() {
  local url="http://127.0.0.1:$PORT/?key=$(cat "$KEY_FILE")" browser
  browser="$(app_browser)"
  if [ -z "$browser" ]; then
    xdg-open "$url"
    return
  fi
  # Her own browser profile keeps the window separate and remembers the microphone permission.
  # Autoplay lets her say hello out loud before the first click.
  "$browser" --user-data-dir="$HOME/.local/share/sofia/browser" --no-first-run --no-default-browser-check \
    --autoplay-policy=no-user-gesture-required --class=Sofia --app="$url" >/dev/null 2>&1 || true
}

app() {
  local started=""
  if ! server_up; then
    ensure_model
    mkdir -p "$STATE_DIR"
    notify-send -i "$ICON" "Sofia" "Waking up..." 2>/dev/null || true
    setsid "$PWD/scripts/sofia.sh" serve --port "$PORT" >"$STATE_DIR/server.log" 2>&1 &
    started=$!
    # The speech models take a while to load.
    for _ in $(seq 180); do
      server_up && break
      kill -0 "$started" 2>/dev/null || die "Sofia failed to start — see $STATE_DIR/server.log"
      sleep 1
    done
    server_up || die "Sofia is taking too long to start — see $STATE_DIR/server.log"
  fi
  open_window
  # Her window is closed now. If this click started her, stop her too so the model leaves the
  # GPU. SIGINT, not SIGTERM: on Ctrl-C she saves a last reflection first.
  if [ -n "$started" ]; then kill -INT "$started" 2>/dev/null || true; fi
}

install_launcher() {
  local entry="$HOME/.local/share/applications/sofia.desktop" desktop
  mkdir -p "$(dirname "$entry")"
  cat >"$entry" <<EOF
[Desktop Entry]
Type=Application
Name=Sofia
GenericName=AI companion
Comment=Talk to Sofia, your English-speaking companion
Exec="$PWD/scripts/sofia.sh" app
Icon=$ICON
Terminal=false
Categories=Utility;Chat;
StartupWMClass=Sofia
StartupNotify=true
EOF
  chmod +x "$entry"
  log "Added Sofia to the app menu."
  desktop="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
  if [ -d "$desktop" ]; then
    cp "$entry" "$desktop/sofia.desktop"
    gio set "$desktop/sofia.desktop" metadata::trusted true 2>/dev/null || true
    log "Added a Sofia icon to $desktop — double-click it to open her."
  fi
}

command="${1:-start}"
[ $# -gt 0 ] && shift

case "$command" in
  setup) setup ;;
  app)   app ;;
  install) install_launcher ;;
  chat)  ensure_model; exec python3 -m sofia.chat ;;
  voice) ensure_model; [ -x .venv/bin/python ] || die "Voice needs the speech stack: ./scripts/sofia.sh setup"
         exec .venv/bin/python -m sofia.voice ;;
  start|serve)
         ensure_model
         # Plain start opens her window (in Chrome) once she is up; the server stays in this terminal.
         if [ "$command" = start ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
           ( for _ in $(seq 180); do server_up && { open_window; exit; }; sleep 1; done ) &
         fi
         [ -x .venv/bin/python ] || log "Speech stack not installed — text only. Run ./scripts/sofia.sh setup for voice."
         exec "$(if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)" -m sofia.server "$@" ;;
  test)  exec python3 -m unittest discover tests ;;
  *)     sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
