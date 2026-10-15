#!/bin/bash
# Builds the suite's runner (pgrun.m) against a dist tree (`cargo xtask
# dist`): its header, and its dylib, linked by its install name
# (@rpath/libreims_vgpu.dylib) with an LC_RPATH at the tree's lib, so the
# runner loads the library the tree holds when it runs, whichever build that
# is. Plain arm64 Objective-C with ARC, as a hypervisor is built, not with the
# library's flags.
#
#   build.sh DIST_DIR OUTPUT
set -euo pipefail
DIST="$(cd "$1" && pwd)"
OUT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$(dirname "$OUT")"
xcrun clang -arch arm64 -mmacosx-version-min=26.6 -fobjc-arc -O1 -g \
  -Wall -Wno-deprecated-declarations -Wno-unguarded-availability-new \
  -I "$DIST/include" "$DIST/lib/libreims_vgpu.dylib" -Wl,-rpath,"$DIST/lib" \
  -framework Foundation -framework Metal \
  "$HERE/pgrun.m" -o "$OUT.tmp"
# The runner needs no entitlement; it is signed ad hoc, as QEMU's build signs
# QEMU.
codesign -f -s - "$OUT.tmp" 2>/dev/null
mv "$OUT.tmp" "$OUT"
