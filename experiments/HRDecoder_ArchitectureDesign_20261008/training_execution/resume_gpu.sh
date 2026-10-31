#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/HRDecoder_v3_20261008
cd "$ROOT/training_execution"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
flock -n "$ROOT/reports/gpu.lock" /root/miniconda3/bin/python -u train.py --train --resume
