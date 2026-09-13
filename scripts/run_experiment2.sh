#!/usr/bin/env bash
# Experiment 2: fly brain as entropy source. Runs the tapped flies, whitens, types, tests.
set -euo pipefail
cd "$(dirname "$0")/.."
PROCS=${PROCS:-4}
NUMBA_NUM_THREADS=1 python -m flyhamlet.run_flies --tap --log-dir results/arena_tap --procs "$PROCS" "$@"
mkdir -p results/entropy_exp2
python analysis/whiten.py results/entropy/isi_fly*.bin --out results/entropy_exp2/keys.txt --hash-bin results/entropy_exp2/whitened_sha256.bin
python analysis/typist.py results/entropy_exp2/keys.txt | tee results/entropy_exp2/typist.txt
python analysis/entropy_tests.py results/entropy/isi_fly*.bin --out results/entropy_exp2
