#!/usr/bin/env bash
#
# verify-gpu.sh — run after rebooting into the NVIDIA driver.
# Confirms the driver is loaded, CUDA is visible, and reports free VRAM.
#
#   ./scripts/verify-gpu.sh
#
set -uo pipefail

ok()   { printf '\033[1;32m ✔\033[0m %s\n' "$*"; }
bad()  { printf '\033[1;31m x\033[0m %s\n' "$*"; }
info() { printf '\033[1;36m •\033[0m %s\n' "$*"; }

echo "=== Sofia GPU check ==="

# 1) nouveau should be gone
if lsmod | grep -q '^nouveau'; then
  bad "nouveau is STILL loaded — the switch didn't take. Re-check setup-gpu.sh / reboot."
else
  ok "nouveau not loaded"
fi

# 2) nvidia module loaded
if lsmod | grep -q '^nvidia'; then
  ok "nvidia kernel module loaded"
else
  bad "nvidia module NOT loaded"
fi

# 3) nvidia-smi works
if command -v nvidia-smi >/dev/null; then
  ok "nvidia-smi present"
  echo "----------------------------------------------------------"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu \
             --format=csv 2>/dev/null || nvidia-smi
  echo "----------------------------------------------------------"
else
  bad "nvidia-smi not found — driver userspace missing"
fi

# 4) driver in use for the card
info "Kernel driver in use:"
lspci -k -s 01:00.0 2>/dev/null | grep -i "driver in use" || true

echo
echo "If you see the RTX 3050 with free VRAM above, the GPU is ready for Sofia. 🎉"
