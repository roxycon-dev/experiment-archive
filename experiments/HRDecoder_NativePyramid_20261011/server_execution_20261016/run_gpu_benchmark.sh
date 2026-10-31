set -eu
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
cd "$BASE/code"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 "$BASE/venv/bin/python" "$BASE/benchmark_gpu.py"
