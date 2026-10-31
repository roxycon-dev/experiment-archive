#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/HRDecoder_v3_20261008
cd "$ROOT/training_execution"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
PY=/root/miniconda3/bin/python
# Smoke uses a separate process and discards its single synthetic/probe update.
# The formal process reconstructs the seed-42 ImageNet baseline from scratch.
flock -n "$ROOT/reports/gpu.lock" /bin/bash -c '
  /root/miniconda3/bin/python -u train.py --smoke
  /root/miniconda3/bin/python -u train.py --train
'
