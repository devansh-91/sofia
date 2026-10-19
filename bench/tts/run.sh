#!/usr/bin/env bash
# Run the TTS comparison: every voice speaks every line, then listen with serve.py.
#
#   bench/tts/run.sh [kokoro pocket chatterbox]    (default: all)
#
# Kokoro and Pocket run on the CPU. Chatterbox starts from an empty GPU (all Ollama models
# are unloaded first), then repeats a few lines on the CPU, since in Sofia the LLM takes
# most of the 6 GB.
set -uo pipefail   # no -e: one engine failing must not stop the others

HERE="$(cd "$(dirname "$0")" && pwd)"
SOFIA="$(cd "$HERE/../.." && pwd)"
DATA="${SOFIA_BENCH_DIR:-$HOME/.local/share/sofia/tts-bench}"
VENVS="$DATA/venvs"
CPU_LINES=l01,l04,l08
KOKORO_VOICES=(af_heart af_bella bf_emma)
POCKET_VOICES=(jane eve)

free_gpu() { ollama ps | awk 'NR > 1 { print $1 }' | xargs -r -n1 ollama stop; }

bench() {
  local engine=$1 python=$2; shift 2
  echo "== $engine $*"
  "$python" "$HERE/engines/${engine}_engine.py" "$@" || echo "!! $engine $* failed (see above)"
}

engines=("$@")
[[ ${#engines[@]} -gt 0 ]] || engines=(kokoro pocket chatterbox)
for engine in "${engines[@]}"; do
  case $engine in
    kokoro)
      for voice in "${KOKORO_VOICES[@]}"; do bench kokoro "$SOFIA/.venv/bin/python" --voice "$voice"; done ;;
    pocket)
      for voice in "${POCKET_VOICES[@]}"; do bench pocket "$VENVS/pocket/bin/python" --voice "$voice"; done ;;
    chatterbox)
      for variant in turbo nano; do
        free_gpu
        bench chatterbox "$VENVS/chatterbox/bin/python" --voice "$variant"
        bench chatterbox "$VENVS/chatterbox/bin/python" --voice "$variant" --device cpu --only "$CPU_LINES"
      done ;;
    *)
      echo "unknown engine: $engine" ;;
  esac
done
echo "Listen: python3 $HERE/serve.py   (then open http://127.0.0.1:8766)"
