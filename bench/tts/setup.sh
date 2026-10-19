#!/usr/bin/env bash
# Venvs for the TTS comparison bench (their torch pins clash, so one per engine).
#
#   bench/tts/setup.sh [pocket chatterbox]    (default: both)
#
# Everything lands under $SOFIA_BENCH_DIR (~/.local/share/sofia/tts-bench); model weights
# go to the normal Hugging Face cache. Kokoro needs nothing here: it is Sofia's own voice and
# runs in her .venv (./scripts/sofia.sh setup).
set -euo pipefail

DATA="${SOFIA_BENCH_DIR:-$HOME/.local/share/sofia/tts-bench}"
VENVS="$DATA/venvs"
# cu126 wheels run on the 550 driver (CUDA 12.4) through minor-version compatibility.
CUDA_INDEX=https://download.pytorch.org/whl/cu126
CPU_INDEX=https://download.pytorch.org/whl/cpu
# Pinned so a rerun installs exactly what was benchmarked.
CHATTERBOX_REV=5de7a54aa4e5e2baadb0182dde554908b48b85c2

venv() { uv venv --quiet --allow-existing --python "$2" "$VENVS/$1"; }
install() { local name=$1; shift; uv pip install --python "$VENVS/$name/bin/python" "$@"; }

setup_pocket() {
  venv pocket 3.12
  install pocket --index-url "$CPU_INDEX" torch==2.14.0
  install pocket pocket-tts==3.3.0 soundfile
}

setup_chatterbox() {
  venv chatterbox 3.11
  install chatterbox --index-url "$CUDA_INDEX" torch==2.6.0 torchaudio==2.6.0
  install chatterbox "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@$CHATTERBOX_REV" soundfile
}

mkdir -p "$VENVS"
engines=("$@")
[[ ${#engines[@]} -gt 0 ]] || engines=(pocket chatterbox)
for engine in "${engines[@]}"; do
  echo "== setting up $engine"
  "setup_$engine"
done
echo "done: venvs in $VENVS"
