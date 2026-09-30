#!/usr/bin/env bash
#
# setup-gpu.sh — install the NVIDIA proprietary driver + CUDA runtime on Kali
# so Sofia can run Qwen on the RTX 3050. Run with sudo, then REBOOT.
#
#   sudo ./scripts/setup-gpu.sh
#
# Safe to re-run (idempotent). Secure Boot is OFF on this machine, so no MOK
# enrollment is needed. This does NOT install the full CUDA toolkit (nvcc) —
# the driver alone is enough for Ollama / llama.cpp CUDA builds that ship libs.
#
set -euo pipefail

log()  { printf '\033[1;36m[gpu-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gpu-setup]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[gpu-setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "Run with sudo: sudo $0"

KREL="$(uname -r)"
log "Running kernel: $KREL"

# --- Sanity: Secure Boot must be off (or module won't load unsigned) ----------
SBVAR="$(ls /sys/firmware/efi/efivars/SecureBoot-* 2>/dev/null | head -1 || true)"
if [ -n "$SBVAR" ]; then
  SB="$(od -An -t u1 "$SBVAR" 2>/dev/null | awk '{print $NF}')"
  if [ "$SB" = "1" ]; then
    die "Secure Boot is ON. The DKMS-built NVIDIA module won't load unsigned.
         Either disable Secure Boot in BIOS, or set up MOK signing first."
  fi
  log "Secure Boot: OFF (good)"
fi

# --- Install headers, build tools, and the driver -----------------------------
log "apt update..."
apt-get update -y

log "Installing kernel headers, dkms and build tools..."
apt-get install -y \
  "linux-headers-${KREL}" \
  linux-headers-amd64 \
  dkms build-essential

log "Installing NVIDIA driver (550) metapackage..."
# The Debian/Kali nvidia-driver package blacklists nouveau and triggers an
# initramfs rebuild automatically via nvidia-kernel-common.
apt-get install -y nvidia-driver nvidia-driver-libs

# --- Make sure nouveau is blacklisted and initramfs is fresh ------------------
if ! grep -rqi "blacklist nouveau" /etc/modprobe.d/ 2>/dev/null; then
  warn "nouveau blacklist not found; adding one."
  cat >/etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
fi

log "Rebuilding initramfs..."
update-initramfs -u

# --- Verify the DKMS module built --------------------------------------------
if command -v dkms >/dev/null; then
  log "DKMS status:"; dkms status || true
fi

cat <<'EOF'

============================================================
 ✅ Install finished.

 NEXT STEP — REBOOT, then verify:

     sudo reboot
     # after login:
     ./scripts/verify-gpu.sh

 If after reboot the screen is black or you land in a console:
   * log in on the TTY (Ctrl+Alt+F3),
   * run:  sudo apt-get install --reinstall xserver-xorg-video-nouveau
   *       sudo apt-get remove nvidia-driver
   * reboot to get your desktop back, then ping me.
============================================================
EOF
