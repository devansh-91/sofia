#!/usr/bin/env bash
# One venv per TTS engine for the comparison bench (their torch/transformers pins clash).
#
#   bench/tts/setup.sh [svara omnivoice chatterbox indicf5]    (default: all)
#
# Everything lands under $SOFIA_BENCH_DIR (~/.local/share/sofia/tts-bench); model weights
# go to the normal Hugging Face cache. Piper needs nothing: it uses Sofia's own .venv.
set -euo pipefail

DATA="${SOFIA_BENCH_DIR:-$HOME/.local/share/sofia/tts-bench}"
VENVS="$DATA/venvs"
PYTHON=3.11
# cu126 wheels run on the 550 driver (CUDA 12.4) through minor-version compatibility.
CUDA_INDEX=https://download.pytorch.org/whl/cu126
CPU_INDEX=https://download.pytorch.org/whl/cpu
SVARA_MODEL="${SVARA_MODEL:-hf.co/mradermacher/svara-tts-v1-GGUF:Q4_K_M}"
# Pinned so a rerun installs exactly what was benchmarked.
CHATTERBOX_REV=5de7a54aa4e5e2baadb0182dde554908b48b85c2
INDICF5_REV=13f7c4d627cc10111aea8fe9c0039462cacacdc7
HINGLISH_TTS_REV=1bce934b26f25edf2cf8a6af0488923b00a07067

venv() { uv venv --quiet --allow-existing --python "$PYTHON" "$VENVS/$1"; }
install() { local name=$1; shift; uv pip install --python "$VENVS/$name/bin/python" "$@"; }

setup_svara() {
  venv svara
  install svara --index-url "$CPU_INDEX" torch==2.8.0   # only the small SNAC decoder runs here
  install svara snac numpy soundfile
  ollama pull "$SVARA_MODEL"
}

setup_omnivoice() {
  venv omnivoice
  install omnivoice --index-url "$CUDA_INDEX" torch==2.8.0 torchaudio==2.8.0
  install omnivoice omnivoice==0.2.1 soundfile
}

setup_chatterbox() {
  venv chatterbox
  install chatterbox --index-url "$CUDA_INDEX" torch==2.6.0 torchaudio==2.6.0
  install chatterbox "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@$CHATTERBOX_REV" soundfile
}

setup_indicf5() {
  venv indicf5
  install indicf5 --index-url "$CUDA_INDEX" torch==2.8.0 torchaudio==2.8.0
  install indicf5 "f5-tts @ git+https://github.com/AI4Bharat/IndicF5.git@$INDICF5_REV" \
    "transformers==4.49.0" "accelerate==0.33.0" soundfile   # IndicF5 itself pins numpy<=1.26.4
  local src="$DATA/src/hinglish-tts"
  [[ -d $src/.git ]] || git clone --quiet https://github.com/harrrshall/hinglish-tts.git "$src"
  git -C "$src" fetch --quiet origin && git -C "$src" checkout --quiet "$HINGLISH_TTS_REV"
}

mkdir -p "$VENVS"
engines=("$@")
[[ ${#engines[@]} -gt 0 ]] || engines=(svara omnivoice chatterbox indicf5)
for engine in "${engines[@]}"; do
  echo "== setting up $engine"
  "setup_$engine"
done
echo "done: venvs in $VENVS"
