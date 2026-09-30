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

# Module checks read /sys/module directly: `lsmod | grep -q` reports a false
# failure under pipefail (grep exits on first match, lsmod dies with SIGPIPE).

# 1) nouveau should be gone
if [ -d /sys/module/nouveau ]; then
  bad "nouveau is STILL loaded — the switch didn't take. Re-check setup-gpu.sh / reboot."
else
  ok "nouveau not loaded"
fi

# 2) nvidia module loaded
if [ -d /sys/module/nvidia ]; then
  ok "nvidia kernel module loaded"
else
  bad "nvidia module NOT loaded"
fi

# 2b) KMS enabled (GNOME/Wayland needs it — this laptop is dGPU-only).
# The modeset parameter is root-only, so without sudo fall back to checking
# that the NVIDIA card exposes display connectors (it only does with KMS on).
kms_on() {
  local modeset card
  if modeset=$(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null); then
    [ "$modeset" = "Y" ]
    return
  fi
  for card in /sys/class/drm/card[0-9]*-*; do
    [ -e "$card" ] || continue
    [ "$(basename "$(readlink -f "${card%%-*}/device/driver")")" = "nvidia" ] && return 0
  done
  return 1
}
if kms_on; then
  ok "nvidia-drm modeset enabled"
else
  bad "nvidia-drm modeset is OFF — Wayland desktop won't start on this driver"
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
