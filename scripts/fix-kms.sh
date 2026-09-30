#!/usr/bin/env bash
#
# fix-kms.sh — enable NVIDIA kernel modesetting so GNOME/Wayland can start on
# the proprietary driver (this laptop is dGPU-only). Run with sudo, then reboot
# into the normal (first) GRUB entry.
#
#   sudo ./scripts/fix-kms.sh
#
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "Run with sudo: sudo $0" >&2; exit 1; }

KVER="7.1.5+kali-amd64"   # the kernel the NVIDIA DKMS module is built for

cat >/etc/modprobe.d/nvidia-kms.conf <<'EOF'
options nvidia-drm modeset=1 fbdev=1
options nvidia-current-drm modeset=1 fbdev=1
EOF
echo "wrote /etc/modprobe.d/nvidia-kms.conf"

update-initramfs -u -k "$KVER"

echo
echo "Done. Now: sudo reboot  (pick the normal first GRUB entry, not recovery)"
