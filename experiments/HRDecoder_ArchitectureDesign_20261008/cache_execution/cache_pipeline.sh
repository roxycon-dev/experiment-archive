#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/HRDecoder_v3_20261008
LEGACY=/root/miniconda3/bin/python
MODERN=/root/miniconda3/envs/hrdecoder_q_20261003/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
cd "$ROOT"
mkdir -p reports
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "CACHE_PIPELINE_FAILED exit=%s time=%s\n" "$code" "$(date -Iseconds)" > reports/pipeline_failure.txt; fi' EXIT
printf 'CACHE_PIPELINE_STARTED %s\n' "$(date -Iseconds)"
"$MODERN" -u audit_inputs.py
for file in B_weights.pth D_student_2000.pth E_1500.pth E4_weights.pth H4_best.pth; do
  while [ ! -f "weights/$file" ]; do sleep 10; done
done
# Adopt an already running exact A export after an interrupted shell controller.
while pgrep -f '^/root/miniconda3/bin/python -u export_A[.]py$' > /dev/null; do sleep 10; done
flock reports/gpu.lock "$LEGACY" -u export_A.py
while [ ! -f weights/U0_inference.pth ]; do sleep 10; done
flock reports/gpu.lock "$LEGACY" -u export_U.py --sid 20170502204843387
while [ ! -f weights/S20_ema_inference.pth ] || [ ! -f references/S20_007-1774-100.npz ]; do sleep 10; done
flock reports/gpu.lock "$MODERN" -u export_S.py --split val --limit 1
flock reports/gpu.lock "$LEGACY" -u export_U.py
flock reports/gpu.lock "$MODERN" -u export_S.py
"$MODERN" -u cache_audit.py
printf 'CACHE_PIPELINE_COMPLETE %s\n' "$(date -Iseconds)" > reports/pipeline_complete.txt
if [ -f shutdown_after_cache.enabled ]; then
  printf 'SHUTDOWN_AUTHORIZED_AFTER_CACHE_PASS %s\n' "$(date -Iseconds)" > reports/shutdown_requested.txt
  sync
  sleep 45
  /bin/bash /usr/bin/shutdown
fi
