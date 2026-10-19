#!/usr/bin/env bash
# The real constraint: can each GPU voice run while Sofia's LLM sits in VRAM, and how fast?
# Loads the `sofia` Ollama model, then runs three lines per variant, saved as <engine>@sofia.
#
#   bench/tts/with-sofia.sh [turbo nano]    (Chatterbox variants; the CPU engines don't need this)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${SOFIA_BENCH_DIR:-$HOME/.local/share/sofia/tts-bench}"
VENVS="$DATA/venvs"
LINES=l01,l04,l08

load_sofia() {
  ollama ps | awk 'NR > 1 { print $1 }' | xargs -r -n1 ollama stop
  curl -s http://127.0.0.1:11434/api/generate -d '{"model": "sofia", "prompt": "hi", "stream": false, "keep_alive": "30m", "think": false, "options": {"num_predict": 1}}' > /dev/null
  echo "sofia loaded: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader) in use"
}

variants=("$@")
[[ ${#variants[@]} -gt 0 ]] || variants=(turbo nano)
for variant in "${variants[@]}"; do
  load_sofia
  echo "== chatterbox-$variant next to sofia"
  "$VENVS/chatterbox/bin/python" "$HERE/engines/chatterbox_engine.py" --voice "$variant" --only "$LINES" --tag sofia \
    || echo "!! chatterbox-$variant failed next to sofia"
  ollama ps | grep -q '^sofia' || echo "note: sofia did not stay loaded next to chatterbox-$variant"
done
ollama stop sofia
