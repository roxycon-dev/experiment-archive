# RESUME_REPORT — refiner_v1 formal run (runs/formal_20260921)

Author: fault-repair turn. Date: 2026-09-21. Server: `root@connect.nmb2.seetacloud.com:48301`
(RTX 3090, 24 GiB). Run root: `/root/autodl-tmp/HRDecoder_FullCompetition_20260920/refiner_v1/runs/formal_20260921`.

---

## 1. Bottom line

The formal run is **RUNNING again and past both previous crash points**.

| | |
|---|---|
| supervisor PID | **7463** (`runs/formal_20260921/supervisor.pid`) |
| status | **RUNNING**, `current_stage=stage2_train`, `stages_done=5/12`, `alerts=[]` |
| resumed | `true`, `started_at=2026-09-21T08:25:25` |
| GPU | only this task: trainer PID 8710, ~21.5 GiB, 91 % util |
| disk free | 13.92 GB (guard floor 4.0 GB) |
| stage2_train progress | **step 320+ and climbing**, 0.70 s/step |
| realistic finish | **~2026-09-21 13:00–13:15** server time (`eta_end_rough` says 11:16 — see §7) |

Two independent defects were fixed. Neither changed the model, the candidate rule,
the loss weights, the epoch budget or the selection protocol. No cache and no
Stage-1 training was deleted or redone.

---

## 2. Defect A — ragged candidate patches (the reported crash)

### Symptom
`eval_stage1_ep200` FAILED (exit 1, 21.2 s, 2026-09-21 01:07) with

```
ValueError: all input arrays must have the same shape
  File ".../eval_refiner.py", line 171, in main
    refined = A.assemble(p4, tiles, np.stack(patches))
```

Full pre-fix traceback preserved at `logs/formal_20260921_remote/eval_stage1_ep200.PRE_FIX_FAILURE.log`.

### Exact size source (measured, not assumed)
From the frozen val cache (`logs/diag_patch_sizes_val.log`, `audit/diag_patch_sizes.py`):

* 1599 candidate patches over the 76 val images; **only 34 (2.1 %) are full 512x512**;
* **892 distinct candidate window shapes**;
* **every one of the 76 images mixes 2..51 distinct shapes**.

Two independent clipping steps create this: `candidate.tile_patch` clips tiles to their
8-connected cluster bbox (a lesion blob is usually ~120 px, not 512), and
`candidate.window_for` clips the extraction window again at the image border. So the
patch list is ragged *by construction*; `np.stack` could never work.

### Correct semantics
The window is `tile grow by context(32)`, clipped at the border and capped at
`max_patch(768)`, so the tile sits at offset `(x0-ex0, y0-ey0)` inside it — **not** at
its centre. `build_patch_cache.cut` already encodes exactly that offset:

```python
valid[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = 1     # build_patch_cache.py:81
```

That is the geometry the refiner was **trained** with. The new inference path uses the
same offset, so inference geometry now equals training geometry.

### Fix (keeps every required invariant)
* `refiner/tiled_infer.py::refine_tiles` — new single geometry owner. Buckets windows by
  exact `(h, w)` (`candidate.bucket_windows_by_shape`), forwards batches of
  **identical-size tensors cut from real image pixels** (`candidate.extract_exact_batch`,
  which fails closed on mixed shapes instead of padding), then writes each output back at
  its **true** tile offset and returns a **ragged per-tile list** `(2, th, tw)`.
* `refiner/assemble.py::_patch_list` — accepts a ragged sequence **or** a stacked array
  when every tile happens to share one size.
* `refiner/candidate.py` — adds `bucket_windows_by_shape`, `extract_exact`,
  `extract_exact_batch`.
* `eval_refiner.py` — calls `TI.refine_tiles(...)` and passes the ragged list to
  `assemble`; no `np.stack` anywhere.

Invariants preserved: **no padding at all** (GroupNorm statistics can never see a pad
pixel), per-patch spatial alignment, Hann-weighted overlap averaging, fixed `p > 0.5`,
EX/SE bitwise untouched, everything outside a candidate tile bitwise unchanged.

> Side note: the old path centre-cropped with `(ph-th)//2`, which is only correct for
> interior tiles. For a left/top-border tile the window has context on one side only, so
> that crop was off by up to 16 px. The new exact-offset crop removes that latent bug.
> `assemble`'s centre-crop stays as a safety net for callers that still pass
> window-sized patches; in the eval path it is a no-op because `tiled_infer` pre-crops.

### Regression tests (`tests/test_refiner_units.py`, section 3b)
One 260x340 image containing an **interior tile, all four border tiles and all four
corner tiles**, all of different sizes (`_ragged_tiles`, >= 6 distinct shapes):

| check | asserts |
|---|---|
| `ragged_windows_exact` | window clips at the border but never loses the tile; extracted patch is bitwise the real pixels of that window |
| `ragged_buckets_no_padding` | buckets are shape-homogeneous and padding-free; **mixed-shape batch is rejected, not padded** |
| `ragged_bucketed_equals_per_patch` | batched (bucket) output == one-patch-at-a-time reference (tol 1e-5) for every tile; `np.stack` on the result still raises, i.e. the crash shape is genuinely reproduced; write-back and tile-order independence |
| `ragged_e2e_identity_bitwise` | zero-init refiner over ragged patches returns the champion **bitwise**; EX/SE changed = 0; outside-candidate changed = 0; all finite |
| `ragged_assemble_placement` | each ragged patch lands on its own tile; stacked form still accepted; wrong arity rejected |
| `ragged_overlap_weighted` | overlapping ragged tiles: exact outside, fixed Hann weighted mean inside |

---

## 3. Defect B — CUDA OOM in `stage2_train` (found after fixing A)

### Symptom
With A fixed, `stage2_train` started and then died (exit 1, 41.5 s, 08:06):

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.85 GiB
(GPU 0; 23.57 GiB total capacity; 18.49 GiB already allocated; 1.95 GiB free;
 21.29 GiB reserved in total by PyTorch)
```

### Exact cause (measured)
`refiner.dataset.collate` zero-pads **every patch in a batch up to that batch's maximum
`(h, w)`** (it is deliberately shape-agnostic, unlike the inference path). On the frozen
train cache:

* **223 / 10608 patches (2.1 %) have both edges >= 512 px**; 195 are full 576x576;
* therefore **~31 % of shuffled batches of 16 get padded to 576x576**;
* measured peak for `16 x 576 x 576` (forward + backward + AdamW step) = **21.34 GiB**
  of the 23.57 GiB torch-visible capacity.

So the frozen stage-2 batch *does* fit, but with only ~2.2 GiB of headroom, and the
default allocator stranded 2.80 GiB of its reservation (`21.29 reserved` vs
`18.49 allocated`) over the preceding 51 mixed-size batches — hence OOM at exactly
`18.49 + 2.85 = 21.34 GiB`.

This is a capacity bug that Stage 1 never exposed (8 images, small patches).

### Fix
`run_formal_v1.py` now hands every GPU stage `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
(constant `CUDA_ALLOC_CONF`, applied by the pure helper `stage_env()` used by `_spawn`;
an explicit operator value in the environment still wins).

`max_split_size_mb:128` stops the allocator from carving blocks larger than 128 MiB into
smaller pieces — precisely what stranded the reservation. It is a **pure allocation hint:
it cannot change any number the trainer computes.**

`torch` here is **2.0.1+cu117**, which predates `expandable_segments` (2.1+); that option
is rejected outright, hence this route.

### Evidence
Deterministic replay harness `tools/probe_stage2_alloc.sh` (replays the *exact* batch
sequence — the trainer seeds `torch.manual_seed(20260920)` before building the
DataLoader, so the shuffle is reproducible). Nothing is written into the run root.

| allocator config | result |
|---|---|
| default | **OOM at step 51**, byte-identical numbers to the formal failure |
| `max_split_size_mb:128` | **OK, 700 consecutive steps** (a full epoch of 663 + 37), rc=0 |
| `roundup_power2_divisions:4` | OOM (bigger blocks, worse) |
| `roundup_power2_divisions:8,garbage_collection_threshold:0.8` | OOM |

The decisive check — the probe's `steps.csv` rows 1..51 are **byte-identical** to the
failed formal run's `steps.csv`:

```
bytes-identical on steps 1..51 : True
step 1  : loss 1.181321   focal 0.023817984  dice 0.46678147  focal_share 0.60486484  gnorm 95.891815
step 51 : loss 0.77726161 focal 0.008583542  dice 0.51975536  focal_share 0.33129934  gnorm 27.252325
```

and step time is unchanged (0.694 s/step with the hint vs 0.70 s/step without).
So the setting removes the OOM without perturbing the optimisation at all.

Regression guard added: `tests/test_refiner_units.py::gpu_stages_allocator_policy`
asserts the policy constant, that gpu/eval stages receive it, that cpu stages do not,
that an operator override wins, and that `stage_env` never mutates its input.

---

## 4. Stage-1 val numbers (node_ep200, `stage1_overfit`, step 2200)

Reproduced three times with identical results (standalone to temp JSON, plus twice
inside the supervisor):

| class | champion | refined | delta |
|---|---|---|---|
| EX | 0.45771525507464833 | 0.45771525507464833 | 0.0 (bitwise untouched) |
| HE | 0.43758256131196327 | 0.4105759612443474 | **-0.02700660006761585** |
| SE | 0.5518070087629399 | 0.5518070087629399 | 0.0 (bitwise untouched) |
| MA | 0.25967231303383037 | 0.2563502037396702 | **-0.003322109294160158** |
| **mIoU** | **0.42669428454584546** | **0.4191121072054015** | **-0.007582177340443974** |

**Stage-1 verdict: does NOT promote.** The frozen selection rule requires val MA IoU
*and* HE IoU both to beat the champion; both are lower. That is expected for an
8-image overfit sanity stage and is reported as-is — no protocol was touched.

### Verification gates (all pass)
```
contract_selfcheck.pass = True
  observed mIoU 0.42669428454584546  (abs_diff 0.0)
  observed MA   0.25967231303383037  (abs_diff 3.1303383035696797e-07)
  observed HE   0.43758256131196327  (abs_diff 4.3868803673019485e-07)
bitwise_checks: ex_se_changed=0  outside_cand_changed=0  nonfinite_refined=0
                n_pixels=401111888  n_candidate_px=19525679  images=76
strict checkpoint load: OK (load_state_dict(..., strict=True) succeeded)
accepted = True        elapsed 359.2 s (standalone) / 386.0 s (in supervisor)
```

Patch protocol actually exercised:
```
mode = "shape-bucketed exact windows (no padding)"
tiles=1599  images_with_tiles=76  buckets=1512  max_bucket=14
max_distinct_shapes_per_image=51  max_window=576  forwards=1514  batch_cap=8
```

Artefacts: `logs/formal_20260921_remote/eval_stage1_ep200_standalone.json` (temp run),
`logs/formal_20260921_remote/eval_stage1_ep200.official.json` (the run's own
`runs/formal_20260921/eval/stage1_ep200.json`).

---

## 5. Test suites — local and server, both green

| where | python / torch | result |
|---|---|---|
| local (Windows) | 3.13.11 / 2.6.0+cu124 | `test_candidate_rules.py` 15 passed 0 failed; `test_refiner_units.py` **39 passed 0 failed** |
| server | 3.8.10 / 2.0.1+cu117 | `test_candidate_rules.py` 15 passed 0 failed; `test_refiner_units.py` **39 passed 0 failed** |

Logs: `logs/tests_fix_local_full_20260921.log`,
`logs/formal_20260921_remote/tests_fix_remote_full_20260921.log`.

---

## 6. How the run was resumed (nothing rebuilt, nothing deleted)

`tools/resume_formal_v1.sh` refuses if a supervisor is alive, archives the previous
`status.json` / `supervisor.out` as `*.before_resume_<ts>.*`, clears a stale `STOP`
sentinel, prints the artefacts that must cause a skip, then launches
`run_formal_v1.py --run-root <RUN> --resume` under `setsid nohup`.

`--resume` recomputes every skip rule from artefacts (`build_stages`): caches by
`index_*.json complete=true`, training stages by `done.json` (or, if only `last.pth`
exists, the trainer is re-invoked **with `--resume`**). Observed in `supervisor.out`:

```
[sup] RESUME ...: previous state=FAILED stages_done=4; stages whose artefacts exist
      are SKIPPED (caches by index_*.json complete=true, training by done.json),
      nothing is deleted or rebuilt
[sup] DONE preflight_tests rc=0 21.5s
[sup] skip cache_train (already present: index_train.json complete=true)
[sup] skip cache_val (already present: index_val.json complete=true)
[sup] skip stage1_train (already present: done.json present)
[sup] == stage eval_stage1_ep200 ...
[sup] DONE eval_stage1_ep200 rc=0 386.0s refined mIoU=0.419112 MA=0.256350 HE=0.410576
      champ mIoU=0.426694 dMA=-0.003322 dHE=-0.027007 dIoU=-0.007582
[sup] == stage stage2_train ...
```

`stages_done` did not regress: 4 -> 5 as the eval completed, and the three artefact
skips are counted as done. `eval_stage1_ep200` has no artefact skip rule, so it was
re-run rather than trusted — it reproduced the identical numbers.

Sequence of this repair: two resumes were needed, because fixing A exposed B.
Attempt 1 (PID 2597) proved A fixed and exposed B; after the B fix, attempt 2 (PID 7463)
is the live run.

---

## 7. Expected completion

Measured throughput is 0.70 s/step, 663 steps/epoch, 30 epochs = 19 890 steps
=> **stage2_train ~13 900 s (~3.9 h)**, then 6 val evals at ~386 s each (~0.65 h),
then selection.

* stage2_train started ~08:32:30 -> ends ~**12:25**
* last eval (`eval_stage2_ep030`) -> ends ~**13:05**

**Caveat:** `status.json.eta_end_rough` (11:16) is optimistic because
`EST_S["stage2_train"] = 7200 s` understates the real ~13 900 s. Trust the numbers above,
or watch `stage2/epochs.csv` (one row per epoch, with `elapsed_s`).

Monitoring: read `runs/formal_20260921/status.json` (`state`, `current_stage`,
`stages_done`, `gpu_apps`, `alerts`, `heartbeat_age_s`). Safe stop is `touch <RUN>/STOP`
or `tools/stop_formal_v1.sh` — stop only, never deletes.

---

## 8. Files changed this turn

| file | change |
|---|---|
| `run_formal_v1.py` | `CUDA_ALLOC_CONF` constant + `stage_env()` helper; `_spawn` uses it |
| `tests/test_refiner_units.py` | new `gpu_stages_allocator_policy` check |
| `tools/probe_stage2_alloc.sh` | new deterministic replay harness (evidence) |

Carried in from the interrupted previous turn and verified byte-identical local <-> server
(md5 compared): `eval_refiner.py`, `refiner/tiled_infer.py`, `refiner/assemble.py`,
`refiner/candidate.py`, `tests/test_refiner_units.py`, `tools/verify_ragged_fix.py`,
`tools/resume_formal_v1.sh`, `audit/diag_patch_sizes.py`.

`tools/verify_ragged_fix.py` (previous turn, re-verified on the server against a real val
image) reports `verdict: true`: 72148/72148 elements exact vs the per-patch reference,
`ex_se_changed=0`, `outside_cand_changed=0`, `nonfinite_refined=0`,
bucketed-vs-per-patch max abs diff 0.0.

---

## 9. Explicitly NOT changed

* model architecture, weights, `residual_scale`
* candidate threshold / dilate / margin / tile / stride (`t=0.10, d=0, margin=24`)
* loss terms and weights (`L = 30.0*focal_bce(a=.75,g=2) + 1.0*soft_dice`)
* epoch budgets, lr schedule, optimiser, **batch size (16)**
* pre-registered `checkpoint_nodes` and the val MA **and** HE promotion rule
* decision threshold `p > 0.5`, Hann overlap averaging
* no cache rebuilt, no Stage-1 training redone, no `test` split touched (fail-closed
  guards `assert_val_only_inference` / `assert_no_test_paths` stayed active)
* nothing was shut down or powered off

---

## 10. Artefact inventory (synced to `refiner_v1/logs/formal_20260921_remote/`)

```
status.json                                   live supervisor status (PID 7463, RUNNING)
supervisor.pid                                7463
supervisor.out                                current supervisor stdout (after resume)
supervisor.before_resume_20260921_075851.out  archived FAILED-time evidence (defect A)
eval_stage1_ep200.PRE_FIX_FAILURE.log         verbatim pre-fix ValueError traceback
eval_stage1_ep200_standalone.json / .log      temp-JSON standalone re-run (gate §4)
eval_stage1_ep200.official.json               the run's own eval report
eval_stage1_ep200.after_fix.log               in-supervisor re-run log
stage2_train.OOM_FAILURE.log                  verbatim defect-B traceback
stage2_OOM_steps.csv / stage2_OOM_heartbeat.json   defect-B state at crash (step 51)
stage2_train.current.log                      live stage2 progress
stage1_done.json                              Stage-1 completion record
tests_fix_remote_full_20260921.log            server test suite (54 checks green)
probe/probe_baseline_OOM.log                  replay, default allocator -> OOM step 51
probe/probe_maxsplit128.log                   replay, 150 steps with the fix
probe/probe_epoch_full_maxsplit128.log        replay, 700 steps with the fix
probe/probe_epoch_full_steps.csv              per-step numbers (byte-identical 1..51)
tools_logs/verify_ragged_fix_007-1774-100.json  ragged-fix verification on a real val image
```

Local-side logs from this turn: `logs/tests_fix_local_full_20260921.log`,
`logs/diag_patch_sizes_val.log`, `logs/tests_fixed_local_20260921.log`.

Server-side replay scratch (evidence, harmless, outside the run root):
`/root/autodl-tmp/s2probe/` — a few MB, no checkpoints of value; safe to delete later.
