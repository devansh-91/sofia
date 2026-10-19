#!/usr/bin/env bash
#
# finetune.sh — fine-tune Sofia on her training dialogues (training/), all on this laptop.
#
#   ./scripts/finetune.sh setup     the training venv, the weights and llama-quantize (~20 GB, once)
#   ./scripts/finetune.sh build     training/dialogues + liked/corrected log exchanges -> training/data
#   ./scripts/finetune.sh train     the LoRA (needs the GPU: stop Sofia first; ~15 min)
#   ./scripts/finetune.sh merge     fold it into the uncensored GGUF, then build sofia-tuned in Ollama
#   ./scripts/finetune.sh compare   sofia and sofia-tuned side by side on the held-out messages
#   ./scripts/finetune.sh all       build, train, merge, compare
#
# Then try her with:  SOFIA_MODEL=sofia-tuned SOFIA_BASE_MODEL=sofia-tuned-base ./scripts/sofia.sh
#
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN="$HOME/.local/share/sofia/train"
PY="$TRAIN/venv/bin/python"
CUDA_INDEX=https://download.pytorch.org/whl/cu126
CONTEXT=8192  # as sofia-base (scripts/sofia.sh): the memory prompts run on the same loaded copy
# Ollama quantizes only safetensors imports, so llama.cpp's llama-quantize makes the Q4_K_M.
LLAMA_CPP="${SOFIA_LLAMA_CPP:-$TRAIN/llama.cpp}"
QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mxx \033[0m %s\n' "$*" >&2; exit 1; }

setup() {
  command -v uv >/dev/null || die "uv is required: https://docs.astral.sh/uv/"
  mkdir -p "$TRAIN"
  uv venv --quiet --allow-existing --python 3.11 "$TRAIN/venv"
  uv pip install --python "$PY" --index-url "$CUDA_INDEX" torch==2.6.0
  uv pip install --python "$PY" "transformers==5.17.*" peft trl accelerate bitsandbytes datasets huggingface_hub gguf safetensors
  if [ ! -x "$QUANTIZE" ]; then
    log "Building llama.cpp's llama-quantize (CPU only)..."
    uv pip install --python "$PY" cmake ninja
    [ -d "$LLAMA_CPP" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP"
    PATH="$TRAIN/venv/bin:$PATH" cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -G Ninja -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
    PATH="$TRAIN/venv/bin:$PATH" cmake --build "$LLAMA_CPP/build" --target llama-quantize -j "$(nproc)"
  fi
  log "Downloading Qwen/Qwen3.5-4B (the weights the LoRA trains on, ~9 GB)..."
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-4B', local_dir='$TRAIN/Qwen3.5-4B',
    allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', 'LICENSE'])"
  log "Downloading the uncensored model's BF16 GGUF (what the LoRA is merged into, ~8 GB)..."
  "$PY" -c "from huggingface_hub import hf_hub_download as h; h('HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive',
    'Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf', local_dir='$TRAIN/hauhau')"
}

build() { .venv/bin/python training/build.py; }

train() {
  [ -x "$PY" ] || die "Run ./scripts/finetune.sh setup first"
  if ps -eo args | grep -q '^[.]venv/bin/python -m sofia.server'; then
    die "Sofia is running and holds the GPU: close her window (or stop the server) first"
  fi
  curl -fsS http://127.0.0.1:11434/api/ps | grep -q '"name"' && {
    log "Unloading Ollama's models to free the GPU..."
    for model in $(curl -fsS http://127.0.0.1:11434/api/ps | "$PY" -c "import json,sys; print(' '.join(m['name'] for m in json.load(sys.stdin)['models']))"); do
      curl -fsS http://127.0.0.1:11434/api/generate -d "{\"model\": \"$model\", \"keep_alive\": 0}" >/dev/null
    done
  }
  "$PY" training/train.py "$@"
}

merge() {
  "$PY" training/merge.py
  local dir; dir="$(mktemp -d)"
  [ -x "$QUANTIZE" ] || die "llama-quantize is missing: run ./scripts/finetune.sh setup"
  log "Quantizing to Q4_K_M, as Sofia's model is..."
  "$QUANTIZE" "$TRAIN/sofia-tuned-BF16.gguf" "$TRAIN/sofia-tuned-Q4_K_M.gguf" Q4_K_M >"$TRAIN/quantize.log" 2>&1 \
    || die "Quantizing failed: see $TRAIN/quantize.log"
  printf 'FROM %s\nPARAMETER num_ctx %s\n' "$TRAIN/sofia-tuned-Q4_K_M.gguf" "$CONTEXT" >"$dir/Modelfile"
  log "Importing it into Ollama as sofia-tuned-base..."
  ollama create sofia-tuned-base -f "$dir/Modelfile"
  # Her persona without the few-shot examples: the tuning replaces them, and the prompt gets shorter.
  python3 scripts/tuned-modelfile.py >"$dir/Sofia-tuned.Modelfile"
  ollama create sofia-tuned -f "$dir/Sofia-tuned.Modelfile"
  rm -rf "$dir"
  log "Built sofia-tuned. Compare: ./scripts/finetune.sh compare"
}

compare() { .venv/bin/python training/compare.py "$@"; }

case "${1:-}" in
  setup)   setup ;;
  build)   build ;;
  train)   shift; train "$@" ;;
  merge)   merge ;;
  compare) shift; compare "$@" ;;
  all)     build && train && merge && compare ;;
  *)       sed -n '3,14p' "$0"; exit 1 ;;
esac
