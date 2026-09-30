#!/usr/bin/env bash
#
# performance-mode.sh — tune the laptop for Sofia.
#
#   sudo ./scripts/performance-mode.sh on      # force max performance
#   sudo ./scripts/performance-mode.sh off     # back to balanced/power-save
#   sudo ./scripts/performance-mode.sh auto    # performance on AC, balanced on battery
#        ./scripts/performance-mode.sh status  # show current state (no sudo needed)
#
# What it changes when ON (all reverted by OFF):
#   * CPU governor        -> performance      (via sysfs, no cpupower needed)
#   * CPU EPP hint        -> performance      (intel_pstate)
#   * ACPI platform profile -> performance    (if the laptop exposes one)
#   * NVIDIA persistence mode -> on, power limit -> max
#
# AC-aware: `auto` (and `on` while on battery) will WARN and use balanced on
# battery to protect runtime/heat, matching "reduce power when unplugged".
#
set -uo pipefail

log()  { printf '\033[1;36m[perf]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[perf]\033[0m %s\n' "$*"; }

need_root() { [ "$(id -u)" -eq 0 ] || { echo "Run with sudo for '$1'."; exit 1; }; }

on_ac() {
  # returns 0 if plugged in
  for p in /sys/class/power_supply/*/online; do
    [ -r "$p" ] || continue
    # only Mains-type supplies report AC state
    t="$(cat "${p%/online}/type" 2>/dev/null || true)"
    [ "$t" = "Mains" ] && [ "$(cat "$p" 2>/dev/null)" = "1" ] && return 0
  done
  return 1
}

set_cpu() { # $1 = performance | powersave ; $2 = EPP hint
  local gov="$1" epp="$2"
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -w "$g" ] && echo "$gov" > "$g" 2>/dev/null || true
  done
  for e in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
    [ -w "$e" ] && echo "$epp" > "$e" 2>/dev/null || true
  done
  log "CPU governor -> $gov, EPP -> $epp"
}

set_platform_profile() { # $1 = performance | balanced | low-power
  local want="$1" f=/sys/firmware/acpi/platform_profile
  [ -w "$f" ] || { warn "no ACPI platform_profile on this machine (skipping)"; return; }
  local choices; choices="$(cat /sys/firmware/acpi/platform_profile_choices 2>/dev/null || true)"
  if echo "$choices" | grep -qw "$want"; then
    echo "$want" > "$f" 2>/dev/null && log "platform_profile -> $want"
  else
    warn "platform_profile '$want' not offered (have: $choices)"
  fi
}

set_gpu() { # $1 = max | default
  command -v nvidia-smi >/dev/null || { warn "nvidia-smi not found (driver not installed yet?)"; return; }
  nvidia-smi -pm 1 >/dev/null 2>&1 && log "GPU persistence mode ON" || warn "could not set persistence mode"
  if [ "$1" = "max" ]; then
    # query max power limit and apply it
    local maxp
    maxp="$(nvidia-smi -q -d POWER 2>/dev/null | awk -F: '/Max Power Limit/{gsub(/[^0-9.]/,"",$2); print $2; exit}')"
    if [ -n "${maxp:-}" ]; then
      nvidia-smi -pl "${maxp%.*}" >/dev/null 2>&1 && log "GPU power limit -> ${maxp} W (max)" \
        || warn "GPU power limit not adjustable on this laptop (fine, ignoring)"
    fi
  else
    nvidia-smi -rgc >/dev/null 2>&1 || true   # reset clocks
    log "GPU clocks reset to default"
  fi
}

apply_on() {
  need_root on
  set_cpu performance performance
  set_platform_profile performance
  set_gpu max
  log "Performance mode ON."
}

apply_off() {
  need_root off
  set_cpu powersave balance_power
  set_platform_profile balanced
  set_gpu default
  log "Performance mode OFF (balanced)."
}

status() {
  echo "AC connected : $(on_ac && echo yes || echo no)"
  echo "CPU governor : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo '?')"
  echo "EPP          : $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo 'n/a')"
  echo "Platform prof: $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo 'n/a')"
  if command -v nvidia-smi >/dev/null; then
    echo "GPU          : $(nvidia-smi --query-gpu=name,power.limit,persistence_mode --format=csv,noheader 2>/dev/null || echo '?')"
  else
    echo "GPU          : driver not installed yet"
  fi
}

case "${1:-status}" in
  on)     apply_on ;;
  off)    apply_off ;;
  auto)
    if on_ac; then log "On AC power -> performance"; apply_on
    else warn "On battery -> staying balanced to save power/heat"; apply_off; fi ;;
  status) status ;;
  *) echo "usage: $0 {on|off|auto|status}"; exit 1 ;;
esac
