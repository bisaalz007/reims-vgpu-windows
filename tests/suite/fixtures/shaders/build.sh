#!/bin/bash
# Compiles each .metal here into its own metallib, the form a function
# record's IR takes: the framework hands it to Metal's newLibraryWithData:
# unmodified. The metallibs are committed, so a scenario's input bytes do
# not drift with the toolchain; run this only to change a fixture, and bless
# the scenarios that use it again. two.metal holds two kernels on purpose,
# and fc.metal an unspecialized function constant.
set -euo pipefail
cd "$(dirname "$0")"
for m in *.metal; do
  n="${m%.metal}"
  xcrun -sdk macosx metal -c "$m" -o "$n.air"
  xcrun -sdk macosx metallib "$n.air" -o "$n.metallib"
  rm "$n.air"
done
head -c 1000 kernel.metallib > kernel_trunc.metallib
