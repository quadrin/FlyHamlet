#!/usr/bin/env bash
# Experiment 1: typing entropy. Runs N flies in the typewriter arena and analyses the logs.
set -euo pipefail
cd "$(dirname "$0")/.."
PROCS=${PROCS:-4}
NUMBA_NUM_THREADS=1 python -m flyhamlet.run_flies --procs "$PROCS" "$@"
python analysis/entropy.py --out results/entropy_exp1
