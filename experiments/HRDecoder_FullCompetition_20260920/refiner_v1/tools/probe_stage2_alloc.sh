#!/bin/bash
# Scratch probe (NOT part of the formal run): find a CUDA allocator setting
# under which the FROZEN stage-2 batch fits.
#
# Why this is needed
# ------------------
# `refiner.dataset.collate` zero-pads every patch in a batch up to that batch's
# maximum (h, w).  On the frozen stage-2 cache 223 / 10608 patches (2.1 %) have
# both edges >= 512 px, so ~31 % of shuffled batches of 16 are padded to
# 576x576.  Measured peak for a 16x576x576 forward+backward is 21.34 GiB on the
# 24 GiB RTX 3090 (23.57 GiB torch-visible), i.e. it fits with only ~2.2 GiB of
# headroom, and the default allocator lost the first formal run at step 52 to
# fragmentation (18.49 GiB live + 2.85 GiB requested, 21.29 GiB reserved).
#
# The probe replays the EXACT first N batches of the formal run: the trainer
# seeds torch with --seed 20260920 before building the DataLoader, so the
# shuffle order is deterministic.  Nothing here writes into the run root.
#
# usage:  STEPS=150 bash probe_stage2_alloc.sh
ROOT=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
PY=/root/miniconda3/bin/python
OUT=/root/autodl-tmp/s2probe
STEPS=${STEPS:-150}

rm -rf "$OUT"
mkdir -p "$OUT"
cd "$ROOT"

run() {
  name="$1"; conf="$2"
  d="$OUT/$name"
  mkdir -p "$d"
  if [ -n "$conf" ]; then export PYTORCH_CUDA_ALLOC_CONF="$conf"; else unset PYTORCH_CUDA_ALLOC_CONF; fi
  echo "### config=$name  PYTORCH_CUDA_ALLOC_CONF=${conf:-<unset>}  steps=$STEPS"
  "$PY" refiner_v1/train_refiner.py \
    --stage-config refiner_v1/configs/stage2.json \
    --cache-root refiner_v1/patch_cache \
    --config refiner_v1/configs/refiner_v1.json \
    --out-dir "$d" --overfit-steps "$STEPS" > "$d/log" 2>&1
  rc=$?
  last=$(tail -n 1 "$d/steps.csv" 2>/dev/null | cut -d, -f1)
  warm=$(grep -c '^\[train\] ep=' "$d/log" 2>/dev/null)
  if grep -q OutOfMemoryError "$d/log"; then
    echo "    RESULT=OOM  rc=$rc  last_step=${last:-0}  progress_prints=$warm"
    grep -m1 'Tried to allocate' "$d/log" | sed 's/^/    /'
  else
    echo "    RESULT=OK   rc=$rc  last_step=${last:-0}  progress_prints=$warm"
    grep -m1 'peak' "$d/log" | sed 's/^/    /'
    tail -n 2 "$d/log" | sed 's/^/    /'
  fi
  echo
}

run baseline ""
run maxsplit128 "max_split_size_mb:128"
run roundup4 "roundup_power2_divisions:4"
run roundup8gc "roundup_power2_divisions:8,garbage_collection_threshold:0.8"
echo "### probe done"
