# refiner_v1 — frozen-champion MA/HE candidate patch refiner

Status: **implemented, unit-tested, formal run supervised on the server.**
The formal run was launched detached on 2026-09-21 (see `RUN_REPORT.md`) and is
driven by `run_formal_v1.py`; progress is readable at any time from
`runs/formal_20260921/status.json`.

## What this is

The champion (`val mIoU 0.426694`, MA `.259672`, HE `.437583`) is frozen. This
directory contains a small, separately-trained refiner that is allowed to edit
**only** MA and HE probabilities, and **only inside a pre-declared candidate
region** derived from the champion's own low-threshold activation. EX and SE,
and every pixel outside the candidate region, stay bitwise identical to the
champion.

Mechanisms are borrowed, not re-implemented wholesale:

| Source | What is borrowed | Where |
|---|---|---|
| WSRFNet | fixed, invertible Haar multi-scale high-frequency representation | `refiner/haar.py` (0 parameters) |
| HACDR-Net | heterogeneous receptive-field aggregation (parallel depthwise convs, different kernels/dilations) | `HaarHeteroRefiner.HeteroDWBlock` |

There is no third or fourth module. The whole trainable network is a stem
convolution, two heterogeneous depthwise blocks and a zero-initialised 1x1
head: **25,394 trainable parameters**.

## Layout

```
refiner_v1/
  GO_NO_GO.md              <- decision, evidence, risks, budget  (READ THIS)
  README.md                <- this file
  configs/
    refiner_v1.json        frozen config (rule, model, loss, optim, guards)
    stage1.json            stage-1 override (formal sanity stage)
    stage2.json            stage-2 override + pre-registered eval nodes
  refiner/
    candidate.py           frozen candidate rule + tiling + fixed stitching
    haar.py                fixed invertible Haar HF features
    model.py               HaarHeteroRefiner
    losses.py              focal BCE + soft Dice (two terms, frozen weights)
    dataset.py             ragged patch-cache reader + frozen sampling
    assemble.py            candidate-only write-back, bitwise-preserving
    reference.py           frozen champion probability provider
    guards.py              no-test / frozen-reference / grad-isolation guards
  audit/
    probe_dims.py              GT geometry + connected-component statistics
    audit_candidate_rules.py   GO/NO-GO candidate-recall audit (GPU, val only)
    candidate_audit.json       audit output (rules, recalls, FP separability)
    input_stats.json           per-split GT statistics
  build_patch_cache.py     one-time GPU data build (frozen champion -> patches)
  train_refiner.py         training entry point (per-step CSV, guards, resume)
  eval_refiner.py          full-image val evaluation with contract self-check
  run_formal_v1.py         detached supervisor: all stages + status.json
  tests/                   CPU unit tests (32 + 15, all passing)
  tools/rc.py              local ssh helper for the single allowed server
  tools/start_formal_v1.sh detached launch of the formal run
  tools/stop_formal_v1.sh  graceful stop (writes the STOP sentinel)
  logs/ manifests/ runs/   outputs
```

## Commands

```bash
# --- on the server, root = /root/autodl-tmp/HRDecoder_FullCompetition_20260920
PY=/root/miniconda3/bin/python

# CPU unit tests (no GPU, no champion, no data)
$PY refiner_v1/tests/test_candidate_rules.py
$PY refiner_v1/tests/test_refiner_units.py

# GO/NO-GO candidate audit (val only, ~6 min, GPU)
$PY refiner_v1/audit/audit_candidate_rules.py

# champion baseline self-check on val (~6 min, GPU) -- must reproduce the
# frozen contract numbers exactly
$PY refiner_v1/eval_refiner.py --no-refine

# patch-cache data build (one-time, GPU)
$PY refiner_v1/build_patch_cache.py --split val          #  76 images
$PY refiner_v1/build_patch_cache.py --split train        # 454 images
$PY refiner_v1/build_patch_cache.py --split val --limit 3   # smoke

# learning sanity smoke on TRAIN patches (tiny, bounded)
$PY refiner_v1/train_refiner.py --stage-config refiner_v1/configs/stage1.json \
      --train-images 4 --overfit-steps 40 --out-dir refiner_v1/runs/smoke

# --- formal run: detached, self-monitoring, survives the launching session ---
bash refiner_v1/tools/start_formal_v1.sh        # launch + verify pid/status
cat  refiner_v1/runs/formal_20260921/status.json  # monitor (state, stages,
                                                  # exit codes, disk, alerts)
bash refiner_v1/tools/stop_formal_v1.sh         # graceful stop, deletes nothing
```

Stages executed in order (strictly one GPU job at a time):
`preflight_tests → cache_train → cache_val → stage1_train →
eval_stage1_ep200 → stage2_train → eval_stage2_ep{005,010,015,020,025,030} →
selection`. `selection.json` records the verdict: a node is promoted only if
**both** val MA IoU and val HE IoU beat the champion; otherwise the highest-mIoU
node is kept as `best_for_analysis.pth` and is not called a win.

## Invariants enforced by code (each has a unit test)

1. **Identity at init.** The head is zero-initialised and the residual is
   `p_refined = clip(p_champ + 0.5*tanh(r), 0, 1)`. With `r == 0`, `tanh(0) == 0`
   and `p + 0.0 == p` bitwise, so the refiner returns the champion exactly.
   `tanh'(0) == 1`, so the branch is not dead.
2. **EX/SE never change.** The refiner outputs 2 channels; `assemble()` copies
   the 4-channel champion and overwrites only MA/HE inside candidates.
3. **Outside the candidate region nothing changes**, bitwise, for all classes.
4. **Overlap uses a fixed Hann weighted average** (no learned blending); the
   weights are normalised so a constant input is reproduced exactly.
5. **Fixed threshold `p > 0.5`**, asserted in `assemble.binarize`.
6. **Frozen champion is input-only**: `assert_frozen_reference` and
   `assert_only_refiner_trainable` fail closed, and a unit test checks that a
   frozen reference receives `grad is None`.
7. **No test data**: every path is screened against a `test` component by
   `guards.assert_no_test_paths`; splits are restricted to `train`/`val`.
8. **Strict checkpoint loading** plus per-tensor bitwise verification.

## Two-term loss (frozen 2026-09-21)

`L = 30.0 * focal_bce(alpha=0.75, gamma=2.0) + 1.0 * soft_dice`, computed over
valid pixels only. No third term and no sum of several papers' losses: the
round decision forbids stacking unvalidated objectives.

The focal weight was raised from 1.0 to 30.0 **once, before the first formal
step**, because the learning smoke measured focal at only ≈ 0.5 % of the total
loss at 1.0 (i.e. the objective was effectively Dice-only). It is a
pre-registered design decision, **not** a val-tuned parameter:
`refiner.losses.assert_frozen_objective` fails closed if the configs drift, and
the per-step CSV records `focal_weighted`, `dice_weighted` and `focal_share` so
the claim can be re-checked from the log. It must never be revised afterwards,
and never by looking at val (or test) curves.

## Notes on two implementation choices

* **GroupNorm, not BatchNorm.** Candidate patches have different spatial sizes
  and a batch is zero-padded to its maximum. BatchNorm would fold padding into
  its running statistics; GroupNorm is per-sample and size-agnostic, so padding
  can never change a prediction.
* **Ragged patch storage.** Padding every patch in an image to the largest
  window inflated the first prototype to 3.67 MB/image; storing natural sizes
  brought it to 1.40 MB/image (measured), and the reader rebuilds offsets from
  a per-patch `shape` table.
