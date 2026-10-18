#!/usr/bin/env bash
# Run the TTS comparison: every engine speaks every line, then listen with serve.py.
#
#   bench/tts/run.sh [piper svara omnivoice chatterbox indicf5]    (default: all)
#
# Each GPU engine starts from an empty GPU (all Ollama models are unloaded first), then
# repeats a few lines on the CPU, since in Sofia the LLM takes most of the 6 GB.
set -uo pipefail   # no -e: one engine failing must not stop the others

HERE="$(cd "$(dirname "$0")" && pwd)"
SOFIA="$(cd "$HERE/../.." && pwd)"
DATA="${SOFIA_BENCH_DIR:-$HOME/.local/share/sofia/tts-bench}"
VENVS="$DATA/venvs"
CPU_LINES=l01,l04,l08

free_gpu() { ollama ps | awk 'NR > 1 { print $1 }' | xargs -r -n1 ollama stop; }

bench() {
  local engine=$1 python=$2; shift 2
  echo "== $engine $*"
  "$python" "$HERE/engines/${engine}_engine.py" "$@" || echo "!! $engine $* failed (see above)"
}

engines=("$@")
[[ ${#engines[@]} -gt 0 ]] || engines=(piper svara omnivoice chatterbox indicf5)
for engine in "${engines[@]}"; do
  case $engine in
    piper)
      bench piper "$SOFIA/.venv/bin/python" ;;
    svara)
      free_gpu
      [[ -f $DATA/ref/ref.wav ]] || "$VENVS/svara/bin/python" "$HERE/engines/svara_engine.py" --make-ref
      free_gpu   # so svara's VRAM is measured from an empty GPU
      bench svara "$VENVS/svara/bin/python"
      free_gpu ;;
    omnivoice | chatterbox | indicf5)
      free_gpu
      bench "$engine" "$VENVS/$engine/bin/python"
      bench "$engine" "$VENVS/$engine/bin/python" --device cpu --only "$CPU_LINES" ;;
    *)
      echo "unknown engine: $engine" ;;
  esac
done
echo "Listen: python3 $HERE/serve.py   (then open http://127.0.0.1:8765)"
