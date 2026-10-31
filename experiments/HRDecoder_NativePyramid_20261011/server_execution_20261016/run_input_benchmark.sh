set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
cd "$BASE/code"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$BASE/venv/bin/python" "$BASE/benchmark_input.py"
