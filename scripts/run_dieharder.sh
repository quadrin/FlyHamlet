#!/usr/bin/env bash
# Run a subset of dieharder on the raw ISI stream and on the whitened SHA-256 stream.
# The whitened file is only ~1.8 MB, so dieharder rewinds it (it prints a warning); the full
# battery needs gigabytes. -g 201 reads raw 32-bit little-endian words.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/entropy_exp2/dieharder.txt
: > "$OUT"
for f in raw whitened; do
  echo "### $f.bin" | tee -a "$OUT"
  for t in 100 101 102 15 1 2; do     # sts_monobit, sts_runs, sts_serial, rgb_bitdist, operm5, rank32x32
    dieharder -g 201 -f results/entropy_exp2/$f.bin -d $t 2>/dev/null | grep -E "^\s*(sts_|rgb_|diehard_)" | tee -a "$OUT"
  done
done
