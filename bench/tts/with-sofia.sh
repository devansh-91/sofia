#!/usr/bin/env bash
# The real constraint: can each GPU engine run while Sofia's LLM sits in VRAM, and how fast?
# Loads the `sofia` Ollama model, then runs three lines per engine, saved as <engine>@sofia.
#
#   bench/tts/with-sofia.sh [svara omnivoice chatterbox indicf5]
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

engines=("$@")
[[ ${#engines[@]} -gt 0 ]] || engines=(svara omnivoice chatterbox indicf5)
for engine in "${engines[@]}"; do
  load_sofia
  echo "== $engine next to sofia"
  "$VENVS/$engine/bin/python" "$HERE/engines/${engine}_engine.py" --only "$LINES" --tag sofia \
    || echo "!! $engine failed next to sofia"
  # Ollama makes room for svara by unloading sofia, which then takes seconds to come back.
  if ! ollama ps | grep -q '^sofia'; then
    "$VENVS/svara/bin/python" - "$DATA/out/$engine@sofia/results.json" <<'PY'
import json, sys
path = sys.argv[1]
results = json.load(open(path))
results.setdefault("notes", {})["swap"] = "sofia was unloaded to make room; reloading it takes ~5 s"
json.dump(results, open(path, "w"), indent=2, ensure_ascii=False)
PY
    echo "note: sofia did not stay loaded next to $engine"
  fi
done
ollama stop sofia
