# baseline42 → champion → 47.20 protocol runner

## Status / honesty

* These files were generated on Windows and were **never executed on a real GPU**.
  The Linux target is legacy mmcv/mmseg on a single RTX 4090 (≈1.98 h paid
  window inside a 24 h wall clock; budget 50 RMB).
* **Real-GPU behaviour is unverified.** Expect to fix import/path issues on
  first contact with the legacy repo.
* **Missing assets fail closed**: every config and checkpoint must be listed in
  `expected_sha256` in the settings file and hash-match, otherwise the run
  refuses to start. There is no unhashed fallback of any kind.
* **No synthetic metrics are produced anywhere.** All reported numbers come
  from strict per-image 4-class TP/FP/FN accumulation (threshold 0.5, strict
  `>`, 255 ignored, background not counted; gt nearest-resized to ori shape —
  identical to the old evaluation). Test results are exploratory only; the
  winner is locked on val and never changed by test.

## Files

* `runtime.py` — repo injection, hashing, strict D2000 init (`strict_initialize`
  accepts full-D or student-only; the full reference checkpoint must be
  byte-consistent with B), `configure_trainable` (only `decode_head` and
  `backbone.stage4` trainable; BN and reference frozen), `freeze_bn`,
  `paired_geometry` (shared random geometry for image/label/reference cache,
  normalized CHW output + trace), `loss_pair` (A: student dice; B:
  0.5·student + 0.5·fused dice), `choose_steps`, `gate_first`,
  `choose_checkpoint`, native-resolution reference-probability cache with an
  atomic COMPLETE flag and manifest (checkpoint/config/image IDs and hashes,
  normalization, shapes, finiteness/range — complete only when all 454 train
  entries exist), strict evaluation, the exact champion ensemble (weighted
  members 3/2/1/1 + horizontal flip + per-class 7→16→16→1 correction heads with
  6·tanh delta; no member or head may be missing), and mix = 0.5·champion +
  0.5·new.
* `run_protocol.py` — protocol driver. Phases with caps: cache / starting-D
  val / champion val + timing pilot ≤ 3 h each; pair seed42 ≤ 10 h; pair seed43
  ≤ 6 h; final (incl. backup) 5 h — all inside the global deadline read from
  `ledger.json` (ISO-8601 or epoch seconds; anything else rejected). Every
  train/eval/cache iteration checks the deadline and exits gracefully via the
  shutdown guard. The pilot (50 updates per arm, reinit each) measures the max
  train-step time and a full val eval, then `choose_steps` picks T ∈
  [1000, 4000], multiple of 4, fitting 2·T·step + 8·eval + 600 s overhead in
  21600 s. Main training does not start if T < 1000 or if the final 5 h cannot
  be reserved.
* Gate after seed42: B−A ≥ 0.003 AND ≥2/4 paired classes B>A AND best-B ≥
  starting-D val, otherwise stop and retain the champion (no pointless test).
  Seed43 reversal (B≤A) rejects adoption. Champion vs new vs mix compared on
  **val only**; ties → champion. A `selection_lock.json` (config hashes +
  weights) is written **before** test. `TEST_STARTED` / `TEST_DONE` flags
  prevent automatic re-running of an interrupted test; an interrupted test is
  reported as blocked.
* `run_guarded.sh` (supplied by the environment, not generated here) launches
  `python3 run_protocol.py --settings S.json --run-dir R` under an independent
  watchdog. Add `--dry-run` for local config/split validation with no GPU work
  and no shutdown.

## Data layout (audited, fail closed)

```
data_root/
  images/{train,val,test}/   *.jpg        454 / 76 / 227
  labels/{train,val}/        *.png
  train.txt val.txt test.txt
```

Stems are unique with no intersection (JPEG images vs PNG labels); the split
audit builds `split_assignment.json`-equivalent mapping and refuses to run on
any mismatch. **Test labels are never read during training.**

## Settings

Copy `settings.example.json`, fill real absolute paths and compute the
`expected_sha256` values (`python3 -c "import runtime;print(runtime.sha256(p))"`).
`ledger.json` in the run dir must contain `"deadline": "<ISO or epoch>"`.

## Checkpointing / resume

Checkpoints store the full model, optimizer, python/numpy/torch/cuda RNG and
sampler RNG, iteration, seed, T, arm and config/split hashes. Resume validates
all of them. `last.pth` is saved every 100 iterations; quarter checkpoints are
evaluated on val with save/restore of all RNG states, and the best val mIoU
(ties → earlier step) is chosen via `choose_checkpoint`.
