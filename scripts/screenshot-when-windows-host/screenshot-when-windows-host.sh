#!/usr/bin/env bash
# screenshot-when-windows-host.sh — host-side PNG of a live QEMU window on Windows.
#
# The counterpart to the KDE and macOS scripts, for the x86 Windows/WHPX rail.
# Until this existed there was no way to close the loop on a visual claim on that
# host, so a boot could only ever be reported as "not verified".
#
# Captures the host window, not a guest QMP screendump: the point is to see what
# the host's presentation path actually put on screen, which is exactly what a
# screendump cannot tell you.
#
# Usage:
#   scripts/screenshot-when-windows-host/screenshot-when-windows-host.sh [-o PATH] [--pid PID]
#
# Prints the PNG path on stdout. Errors on stderr, non-zero exit when no QEMU
# window exists, when several do and none was chosen, or when the window is
# minimised (a minimised window cannot be captured, and silently returning a
# blank PNG would be worse than failing).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=""
PID_FILTER=0

usage() {
  cat <<'EOF'
usage: scripts/screenshot-when-windows-host/screenshot-when-windows-host.sh [options]

Capture the host QEMU display window on Windows to a PNG.

Options:
  -o, --output PATH   where to write the PNG (default: $REIMS_VGPU_LOG_DIR or
                      the OS temp directory, as reims-vgpu-screen-<stamp>.png)
      --pid PID       pick one QEMU process when several windows are open
  -h, --help          this text

Output is downscaled to fit 1280x720 (aspect preserved, never upscaled), matching
the other screenshot scripts so an agent pays the same token cost either way.
EOF
}

die() { echo "screenshot-when-windows-host: $*" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    -o|--output) shift; OUT="${1:-}"; shift ;;
    --output=*)  OUT="${1#--output=}"; shift ;;
    --pid)       shift; PID_FILTER="${1:-0}"; shift ;;
    --pid=*)     PID_FILTER="${1#--pid=}"; shift ;;
    -h|--help)   usage; exit 0 ;;
    *)           echo "unknown argument '$1'" >&2; usage >&2; exit 64 ;;
  esac
done

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) ;;
  *) die "this script is for a Windows host; use the KDE or macOS one instead" ;;
esac

if [ -z "$OUT" ]; then
  # Same directory the crate's sinks resolve to, so a screenshot lands beside the
  # fail log it is evidence for.
  dir="${REIMS_VGPU_LOG_DIR:-${TEMP:-${TMP:-/tmp}}}"
  OUT="$dir/reims-vgpu-screen-$(date +%Y%m%d-%H%M%S).png"
fi

# PowerShell is a Windows program: hand it a Windows path, whatever the caller used.
win_path() {
  case "$1" in
    /*) cygpath -w "$1" 2>/dev/null || echo "$1" ;;
    *)  echo "$1" ;;
  esac
}

command -v powershell.exe >/dev/null 2>&1 || die "powershell.exe not found on PATH"

powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$(win_path "$SCRIPT_DIR/screenshot-when-windows-host.ps1")" \
  -Out "$(win_path "$OUT")" \
  -TargetPid "$PID_FILTER" >/dev/null || die "capture failed"

[ -s "$OUT" ] || die "capture produced no file at $OUT"
echo "$OUT"
