# GO / NO-GO — HRDecoder MA/HE candidate patch refiner (refiner_v1)

Date: 2026-09-20
Server: `root@connect.nmb2.seetacloud.com:48301` (RTX3090) — the ONLY server touched.
Root: `/root/autodl-tmp/HRDecoder_FullCompetition_20260920`
Local mirror: `D:\dataset\experiment\HRDecoder_FullCompetition_20260920\refiner_v1`

**Verdict: GO.**

The gate asked for MA GT-pixel recall ≥ 0.85 and HE ≥ 0.90 under a
pre-declared candidate rule, with candidate patches and disk controllable.
The frozen rule reaches **MA 0.9335 / HE 0.9302** at a candidate footprint of
**4.87 % of pixels**, and the measured cache cost extrapolates to **≈ 800 MB**.

**No formal training was started. Test was never read. Nothing was deleted.
The server was not shut down. 44795 / 4090 were never accessed.**

---

## Update 2026-09-21 — R1 closed, formal run launched (see `RUN_REPORT.md`)

The only open design question (R1, focal weight) was decided by the main agent
**before the first formal step** and is now frozen:

```
L = 30.0 * focal_bce(alpha=0.75, gamma=2.0) + 1.0 * soft_dice
```

*Reason:* at `w_focal = 1.0` the learning smoke measured focal at only
**≈ 0.5 %** of the total loss (`focal 0.0016–0.0071` against `Dice 0.30–0.65`),
i.e. the objective was effectively Dice-only and the hard-example term was
never exercised. Multiplying focal by 30 makes it a visible but still
non-dominant part of the early objective. **This is a one-time decision: it
must not be revised afterwards, and never on val (or test) curves.**

Enforcement, so the decision cannot silently drift:

* `refiner/losses.py` carries `FROZEN_TERMS`, `FROZEN_WEIGHTS`,
  `FROZEN_ALPHA/GAMMA` and `assert_frozen_objective()`; every training entry
  point calls it before any compute and fails closed on a mismatch.
* `configs/refiner_v1.json`, `configs/stage1.json` and `configs/stage2.json`
  all carry the identical block (checked by unit test
  `objective_30focal_1dice`); the trainer additionally proves that the stage
  override did not change the base config's loss.
* The per-step CSV records `focal_weighted`, `dice_weighted` and
  `focal_share`, so the R1 question can be re-checked from the log alone.

Formal run protocol (implemented in `run_formal_v1.py`, launched by
`tools/start_formal_v1.sh`): `preflight_tests → cache_train → cache_val →
stage1_train → eval_stage1_ep200 → stage2_train → eval_stage2_ep{005,010,015,
020,025,030} → selection`, strictly sequential, one GPU job at a time. The
validation nodes are pre-registered in the stage configs
(`checkpoint_nodes`) and evaluated only at fixed `p > 0.5`. Selection requires
val MA IoU **and** val HE IoU to both beat the champion; with no such node the
highest-mIoU node is kept for failure analysis and is **not** called a win.

---

## 0. Constraint compliance (facts, not intentions)

| Constraint | Evidence |
|---|---|
| test never read | every entry point screens its paths through `guards.assert_no_test_paths` (`FORBIDDEN_PATH_COMPONENTS=("test",)`); the mmseg `val` dataset is used for both splits (`FrozenChampion.dataset`); dataset counts asserted 76 (val) / 454 (train); unit test `no_test_paths` and `forbidden_path_guard` pass |
| no formal training | `train_refiner.py` was executed only as `--train-images 4 --overfit-steps 40` (a 40-step plumbing smoke on TRAIN patches). `configs/stage2.json` was never run |
| nothing deleted | `df` before/after; only files created under `refiner_v1/` |
| server left running | `nvidia-smi` after every job: process exited, GPU back to idle |
| 44795 / 4090 untouched | all SSH sessions target port 48301 only |
| no val re-scanning after freeze | the 12-rule grid below is the complete set of numbers ever computed; no rule was re-evaluated after the config was frozen |

---

## 1. Audit A1 — what the champion actually is

Source of truth: `analysis/expert_eval_settings.json`
(sha256 `c44992e8c0d9628689f6ee48a17c6e45e22f347608a8f3bfd1f3916d23872d6d`).

**FACT.** The champion is a probability-level ensemble of four frozen members,
then a single post-fusion MA correction head:

```
champion = weighted_mean(D×3, E×2, B×1, E4×1)      with hflip TTA (8 forwards/image)
MA_champion = sigmoid( logit(pf_MA) + H4(f7) )      H4: 7ch -> 16 -> 16 -> 1, Δ = 6·tanh
EX, HE, SE  = pf                                     (untouched by H4)
```

Weights on disk (`/root/autodl-tmp/hr_exec_20260919/assets/`):
`D_student_2000.pth`, `E_1500.pth`, `B_weights.pth`, `E4_weights.pth`,
`H4_best.pth` (15,447 bytes). Independent confirmation from the previous round:
the champion's EX/HE/SE are **bitwise** equal to `BASE_0`, and its MA differs on
72 of 76 val images (`EXECUTION_REPORT_20260920.md` §1.5).

**FACT — there is no reusable probability cache.** A filesystem search for
`*prob*` / `*logit*` under the project root returns only `tests/_probe.json`.
`expert_outputs/` contains **packed binary masks only** (1 bit/pixel,
`np.packbits`, 4 classes) for 6 HRPlus pipelines + U0/H0 + shared GT. The
champion probability therefore has to be **regenerated by inference**; there is
nothing to reuse.

**Consequence:** the patch cache must be produced by re-running the champion,
once, on train (454) and val (76).

---

## 2. Audit A2 — disk, sizes, cache cost

**FACT.** `/root/autodl-tmp`: 50 G total, 35 G used, **16 G available** (69 %).
GPU idle before and after every job this round. No asset was deleted to make
room, and none needed to be.

**FACT.** Data: `/root/autodl-tmp/hr_exec_20260919/data/DDR_split_613`,
**454 train / 76 val / 227 test**, 262 MB total. Image geometry
(measured, all train+val images): median **1728 × 2592** (4.48 MP), max
3456 × 5184 (17.9 MP).

Why full probability maps were rejected:

| Object | Size |
|---|---|
| 4-channel fp32 map, one median image | 4.48 MP × 16 B = **71.7 MB** |
| 4-channel fp32, 530 images | **≈ 38 GB** (infeasible) |
| MA/HE fp16, 530 images | **≈ 9.5 GB** (too much of the 16 GB free) |
| candidate patches, 530 images (measured, §4) | **≈ 0.8 GB** |

To get a per-image candidate recall number we need the MA/HE probabilities at
*original* resolution, so the audit ran the frozen champion on val. Measured
cost: **4.85 s/image** (audit, 76 images in 359 s) and **4.33 s/image** (eval,
76 images in 329 s). ⇒ the one-time cache build costs about **37 min** for the
454 train images and **6 min** for val.

---

## 3. Audit A3 — GT geometry, candidate recall, and false-positive separability

### 3.1 MA/HE connected-component statistics (measured)

Label PNGs carry values 0–4 only; there is **no 255 / ignore region** in either
split (so ignore handling is defensive, not load-bearing).

| split | class | pos images | total CC | CC/image p50 / p90 / max | CC area p50 / p90 / max | bbox w p50 / p90 |
|---|---|---|---|---|---|---|
| train | MA | 338 / 454 | 6 793 | 11 / 46 / 166 | **48 / 172 / 1 272** | **8 / 16** |
| train | HE | 366 / 454 | 8 295 | 11 / 52 / – | 216 / 1 494 / 769 272 | 18 / 56 |
| train | EX | 303 / 454 | 15 431 | 21 / 150 / – | 56 / 520 / 273 752 | 8 / 30 |
| train | SE | 143 / 454 | 779 | 3 / 11 / – | 852 / 4 485 / 17 988 | 36 / 96 |
| val | MA | 59 / 76 | 1 074 | 10 / 37 / 166 | 44 / 132 / 899 | 8 / 14 |
| val | HE | 63 / 76 | 1 555 | 12 / 63 / – | 192 / 1 314 / 75 240 | 18 / 52 |

**FACT that shapes the whole design.** MA lesions are tiny: median area 48 px,
median bounding box **8 × 8 px**, 51 % of train MA components are ≤ 50 px.
Six thousand eight hundred separate MA blobs across 338 images means the
refiner cannot be a global semantic segmenter; it has to be a *local
re-scoring* of pixels the champion already half-flagged.

### 3.2 Champion operating point on val (measured, reproduces the contract exactly)

| class | TP | FP | FN | IoU | recall at p>0.5 | precision |
|---|---|---|---|---|---|---|
| MA | 27 149 | 35 267 | 42 135 | 0.259672 | **0.3919** | 0.4350 |
| HE | 704 581 | 404 243 | 501 343 | 0.437583 | **0.5843** | 0.6354 |

Both IoUs equal the frozen contract values to six decimals — the reconstruction
of the champion inside the audit is therefore verified, not assumed.

The champion misses **61 % of MA pixels** and **42 % of HE pixels** at the
0.5 threshold. That is the headroom the candidate region is meant to expose.

### 3.3 Pre-declared candidate rule and the recall gate

```
seeds = (P_MA > t) | (P_HE > t)          P_* = frozen champion prob, original res
grown = dilate(seeds, rect(2*(d+24)+1))  24 px box margin + d px bridging
boxes = 8-connected component bboxes of grown
tiles = 512x512 grid, stride 384, clipped to each box
cand  = union of tiles                    <- the ONLY pixels a refiner may change
```

`t ∈ {0.05, 0.10, 0.20}` was declared as the single scanned axis, for the
GO/NO-GO decision only. Full grid, val n=76, 401 111 888 pixels:

**The table below was produced twice, independently:** once by the audit script
with its own inline rule engine, and once by the same script after it was
refactored to import the *shipping* rule module (`refiner/candidate.py`).
Comparing the two complete val runs field by field across all 12 rules:
`max abs diff = 0.0` (**bit-identical**), and the champion component counts and
every FP-separability feature matched exactly. There is no drift between the
audited rule and the rule that runs at inference.

| rule | MA recall (upper bound) | HE recall (upper bound) | candidate frac | clusters/img | tiles/img |
|---|---|---|---|---|---|
| t0.05_d0 | 0.9572 | 0.9390 | 0.0595 | 23.71 | 24.37 |
| t0.05_d8 | 0.9635 | 0.9610 | 0.0818 | 20.01 | 21.13 |
| t0.05_d16 | 0.9666 | 0.9751 | 0.1063 | 16.93 | 18.61 |
| t0.05_d32 | 0.9780 | 0.9829 | 0.1548 | 12.36 | 15.49 |
| **t0.10_d0** | **0.9335** | **0.9302** | **0.0487** | **20.58** | **21.04** |
| t0.10_d8 | 0.9424 | 0.9552 | 0.0673 | 17.42 | 18.25 |
| t0.10_d16 | 0.9458 | 0.9615 | 0.0868 | 15.04 | 16.25 |
| t0.10_d32 | 0.9635 | 0.9768 | 0.1296 | 11.24 | 13.66 |
| t0.20_d0 | 0.8815 | 0.9083 | 0.0393 | 17.93 | 18.26 |
| t0.20_d8 | 0.8976 | 0.9213 | 0.0536 | 15.32 | 15.83 |
| t0.20_d16 | 0.9103 | 0.9505 | 0.0703 | 13.26 | 14.14 |
| t0.20_d32 | 0.9341 | 0.9668 | 0.1077 | 10.13 | 11.95 |

### 3.4 Pre-declared selection rule and the frozen choice

> Among the audited grid, take the rule with the **smallest mean candidate
> area** whose MA and HE recall upper bounds both clear the gate
> (MA ≥ 0.85, HE ≥ 0.90) by **at least +0.01 absolute margin**.

Applied mechanically to the table above:

* `t0.20_d0` is smaller (0.0393) but its HE margin is **+0.0083** → excluded.
* `t0.10_d0` is next (0.0487): MA 0.9335 (**+0.0835**), HE 0.9302 (**+0.0302**) → **frozen**.

**FROZEN RULE: `t = 0.10`, `dilate = 0`, `margin = 24`, `tile = 512`,
`stride = 384`.** Written into `configs/refiner_v1.json`; the full grid is kept
in the config and in `audit/candidate_audit.json` so this choice can be checked
mechanically. No further threshold may be evaluated on val.

### 3.5 False-positive separability — the "vessel / dark-texture" hypothesis is NOT supported

For every champion component at p>0.5 (MA: 573 TP / 399 FP components;
HE: 781 TP / 335 FP), shape and green-channel texture features were computed and
scored by a rank AUC for TP-vs-FP separation (0.5 = no information):

| feature | MA AUC | HE AUC | reading |
|---|---|---|---|
| elongation (PCA √(λ1/λ2)) | **0.467** | 0.516 | no signal for MA |
| aspect ratio | 0.501 | 0.505 | no signal |
| fill ratio (area / bbox) | 0.508 | 0.407 | no signal |
| solidity (area / hull) | 0.490 | 0.301 | no signal (HE weakly inverted) |
| area | 0.550 | **0.746** | only HE has a usable size cue |
| green mean | **0.627** | 0.419 | weak for MA |
| green std | 0.612 | 0.641 | weak |
| green ring contrast | 0.467 | 0.407 | no signal |

**FACT.** A hand-written "vessel-like ⇒ suppress" rule has no basis in these
features: elongation/aspect/fill/solidity are all ≈ chance. The only real
signals are weak green-channel intensity statistics (AUC 0.61–0.64) and HE
component size (0.75, but small true exudates exist: 19 % of train HE
components are ≤ 50 px, so a size gate would delete true positives).

**INFERENCE (not fact).** Because no fixed rule can separate these false
positives, whatever improvement is available must come from *learned local
context* — which is exactly what the refiner provides. This supports the
architecture choice rather than a rule-based post-filter, but it also means the
refiner has to learn something a hand-crafted filter provably cannot.

---

## 4. Audit A4 — the disk-minimal, reproducible cache design

Full probability maps were rejected (§2). The design that was implemented and
measured:

```
<patch_cache>/index_{split}.json                 provenance + per-image manifest
<patch_cache>/patch/{split}/{stem}.npz           one compressed npz per image
     tiles  int32 (N,4)      tile rect, original coords
     rects  int32 (N,4)      extraction window actually stored
     shape  int32 (N,2)      per-patch (h,w)  -> reader rebuilds flat offsets
     role   uint8 (N,)       0 cand-pos 1 cand-neg-fp 2 cand-neg-plain 3 background
     rgb    uint8    flat    RGB, SAME rectangle as prob/gt
     prob   float16  flat    frozen champion (MA, HE)
     gt     uint8    flat    GT MA/HE
     valid  uint8    flat    1 inside the tile = write-back region
```

* probability stored **fp16**, RGB **uint8**, GT **uint8** — as instructed;
* **ragged** (natural patch size, no padding): the first prototype padded every
  patch to the largest window in the image and cost **3.67 MB/image**; ragged
  storage cost **1.40 MB/image**. The reader rebuilds offsets from `shape`;
* the window is the tile grown by **32 px of REAL neighbouring pixels**
  (the refiner's effective half-receptive-field is ≈ 28 px), clipped at the
  borders. Zero padding would change what the model sees at a tile edge; real
  context does not. **No padding is ever fed to the model**;
* `rgb`, `prob`, `gt` are always cut from the **same rectangle**, so the
  "same spatial transform for image / probability / label" requirement is
  structural, not a convention;
* `valid` = the tile only. The context ring and batch padding are `valid=0`, so
  they neither contribute to the loss nor get written back.

Measured (4-image smokes, `--limit 4`):

| split | images | tiles | candidate tiles | background tiles | npz bytes | per image |
|---|---|---|---|---|---|---|
| val | 4 | 46 | 38 | 8 | 5.59 MB | 1.40 MB |
| train | 4 | 43 | 35 | 8 | 6.12 MB | 1.53 MB |

**Extrapolation:** 454 × 1.53 MB + 76 × 1.40 MB ≈ **800 MB**. Expected patch
count ≈ 21.04 candidate tiles/img × 454 + 2 background/img × 454 ≈
**10 460 patches**. Both are comfortably inside the 16 GB free.

Independent cross-check: the shipping `eval_refiner.py` measured
**1 599 candidate tiles over 76 val images = 21.04/img**, and
19 525 679 candidate pixels / 401 111 888 total = **4.87 %**, matching the
audit's `t0.10_d0` row exactly.

---

## 5. Gate B — the decision

| gate | threshold | observed for the frozen rule | verdict |
|---|---|---|---|
| MA GT-pixel recall upper bound | ≥ 0.85 | **0.9335** | pass (+0.0835) |
| HE GT-pixel recall upper bound | ≥ 0.90 | **0.9302** | pass (+0.0302) |
| candidate footprint | controllable | **4.87 %** of pixels, 21.04 tiles/img | pass |
| cache cost | controllable | **≈ 800 MB**, ≈ 10 460 patches | pass |
| EX/SE and non-candidate pixels preserved | bitwise | verified over 401 111 888 pixels | pass |

### **GO** — build the frozen-champion MA/HE candidate patch refiner.

The gate numbers are *upper bounds*, not expected results: they say the
candidate region contains 93 % of MA and HE truth, so a perfect refiner could
reach MA recall 0.9335 instead of the champion's 0.3919. Realising that is the
refiner's job and is not guaranteed (see risks §7).

---

## 6. Implementation (C) — written, unit-tested, not trained

### 6.1 Frozen inference path

```
frozen champion (4-member ensemble + H4 MA head, original resolution)
   -> candidate rule (t=0.10, d=0, margin=24, tile=512, stride=384)
   -> per-tile patches (tile + 32 px real context)
   -> HaarHeteroRefiner -> (MA, HE) residual
   -> p_refined = clip(p_champ + 0.5 * tanh(r), 0, 1)          # probability space
   -> write back ONLY inside tiles, fixed Hann weighted average on overlap
   -> EX/SE and every non-candidate pixel = bitwise champion
   -> fixed threshold p > 0.5
```

### 6.2 Model — 25 394 trainable parameters

| part | detail |
|---|---|
| input | 11 ch = RGB(3) + champion MA/HE prob(2) + fixed Haar HF(6) |
| Haar | 2-level orthonormal 2×2 Haar on the green channel, bands h/v/d per level, nearest upsampling, **0 parameters** |
| stem | conv3×3 11→48 + GroupNorm(8) + ReLU |
| block1 | HeteroDW (k,d) = (3,1)(5,2)(3,4), depthwise, fused by 1×1 + residual |
| block2 | HeteroDW (k,d) = (5,1)(3,3)(7,2), depthwise, fused by 1×1 + residual |
| head | conv1×1 48→2, **zero-initialised** |

Why the residual is `0.5·tanh(r)` in **probability** space: with the head
zero-initialised, `r == 0`, `tanh(0) == 0`, `0.5*0 == 0.0`, and `p + 0.0 == p`
**bitwise** ⇒ the model returns the champion exactly at step 0.
`tanh'(0) == 1`, so unlike a ReLU gate the branch is **not dead** (unit test
`gradients_reach_head` asserts a non-zero head gradient on the first backward).
`scale = 0.5` is fixed because every candidate pixel has `p > 0.10`, so a
±0.5 residual is always enough to move it across the 0.5 boundary in either
direction, while bounding the perturbation to half a probability unit.
(This mirrors the champion's own H4 head language: `Δ = 6·tanh`.)

GroupNorm rather than BatchNorm: candidate patches have different spatial sizes
and a batch is zero-padded to its maximum; BatchNorm would fold padding into its
running statistics, GroupNorm is per-sample and size-agnostic.

### 6.3 Loss — exactly two terms

`L = 30.0 · focal_bce(alpha=0.75, gamma=2.0) + 1.0 · soft_dice`, valid pixels
only, per-class Dice averaged over the two classes. (The focal weight was
raised from 1.0 to 30.0 on 2026-09-21 to close R1 — see the update section at
the top; it is frozen and config-enforced.)

* focal BCE is the classification / hard-example term (MA is 1.7 % of val
  pixels and candidate patches are dominated by negatives);
* soft Dice is the region term and is aligned with the reported per-class IoU;
* **no third term and no sum of several papers' losses** — the round decision
  forbids stacking unvalidated objectives. Weights are hard-coded in
  `configs/refiner_v1.json` and mirrored in the stage configs.

### 6.4 Patch sampling (declared in config)

Per image: **all** candidate tiles + **2** background tiles (no GT, no champion
activation, drawn deterministically with `seed=20260920`).

* No cap on candidate tiles: dropping any would silently lower the audited
  recall upper bound.
* Background tiles are training-only; they are not candidates, so inference
  never writes them back.
* The four pixel populations the round asked for are present *inside* the
  candidate tiles. Measured on the smoke cache:
  `candidate_pos 20 / candidate_neg_fp 3 / candidate_neg_plain 12 / background 8`
  tiles (4 train images), and on the val smoke
  `21 / 3 / 14 / 8` (4 val images). Their exact per-split pixel counts
  (`gt_*_px_in_cand`, `champ_fp_*_in_cand`) are written into each cache index.

### 6.5 Tests — 47 CPU unit tests, all passing

`tests/test_candidate_rules.py` — **15/15**:
strict `>` thresholding, seeds = union of classes, exact dilation radius,
tiles always cover their box and never exceed 512, end-to-end candidate tiling,
Hann weights, single-tile stitch identity, overlap = fixed weighted mean,
weight normalisation reproduces a constant, uniform-weight average, undersized
patch rejected, NaN-free untouched region, forbidden-path guard, shape mismatch
rejected.

`tests/test_refiner_units.py` — **32/32**:
1. **identity is bitwise** at init (`torch.equal(refined, p_champ)`)
2. identity comes from the **zero head only** (holds after randomising every
   other conv/GroupNorm)
3. residual is bounded by ±0.5
4. **gradients reach the head** (not a dead branch)
5. Haar level exactly invertible (incl. odd sizes)
6. Haar HF channel count/shape for levels 1–3
7. Haar HF has no parameters but is differentiable
8. 11-channel input assembly
9. **candidate-outside is bitwise champion**
10. **assembled == champion** when the refiner returns the champion, inside candidates too
11. context patch is centre-cropped (no spatial shift)
12. undersized patch rejected
13. empty tile list is a no-op
14. overlap = weighted average, untouched channel bitwise
15. threshold fixed at p>0.5 (non-fixed threshold rejected)
16. **full refiner→assembly round trip is bitwise** on a synthetic image
17. ignore pixels do not contribute (flipping GT in the ignored band is loss-invariant)
18. fully-ignored patch → exactly-zero, finite loss
19. empty lesion (all-zero GT) → finite focal and Dice, no NaN
20. near-perfect prediction matches the closed-form Dice
21. the objective is exactly the two declared terms
22. Dice rewards region overlap
23. **only the refiner receives gradients**; the frozen reference gets `grad is None`
24. frozen-reference guard rejects train-mode and trainable references
25. no-test-path / split guards
26. **strict checkpoint load**, plus per-tensor bitwise verification, plus a
    missing-key load that must raise
27. trainable parameter count < 200 k (`25 394` printed)
28. **the frozen objective is exactly 30.0·focal + 1.0·soft_dice** in the main
    config and in both stage configs, and any drift (weight, extra term, alpha,
    gamma) is rejected (`objective_30focal_1dice`,
    `objective_frozen_fail_closed`, `loss_total_and_focal_share`)
29. **the pre-registered validation nodes are frozen**
    (`validation_nodes_frozen`): stage 2 has exactly 6 nodes every 5 epochs,
    stage 1 exactly 1 at its final epoch, and the node list is consistent with
    `eval_every_epochs` — the count GO_NO_GO R8 budgeted

### 6.6 Executed smokes (the only GPU work this round)

| smoke | command | result |
|---|---|---|
| candidate audit, full val | `audit_candidate_rules.py` | 76 images, 359 s, 12 rules, table §3.3 |
| **champion baseline self-check** | `eval_refiner.py --no-refine` | champion mIoU **0.42669428454584546** vs contract 0.42669428454584546 → **Δ = 0**; MA 0.259672313 / HE 0.437582561 (abs diff 3.1e-7 / 4.4e-7); `EX/SE changed = 0`; `outside-candidate changed = 0`; **accepted = True** |
| patch cache smoke (val) | `build_patch_cache.py --split val --limit 4` | 46 tiles, 5.59 MB |
| patch cache smoke (train) | `build_patch_cache.py --split train --limit 4` | 43 tiles, 6.12 MB |
| **learning smoke** | `train_refiner.py --stage-config configs/stage1.json --train-images 4 --overfit-steps 40` | 43 TRAIN patches; loss **0.6473 → 0.3409** in 40 steps; `non-refiner trainable = 0`; finite grad norms |

The learning smoke is a plumbing check on 4 train images, not a training run.
It shows the model and the two-term loss can reduce the objective and that
gradients flow to every refiner parameter.

---

## 7. Risks and things that are NOT established

**R1 — RESOLVED 2026-09-21: `w_focal = 30.0`.** In the learning
smoke, focal contributed 0.0016–0.0071 against Dice 0.30–0.65, i.e. **≈ 0.5 %
of the total**. The main agent fixed `L = 30·focal_bce(0.75, 2) + 1·soft_dice`
before the first formal step; the value is config-enforced and must not be
revisited. (Original text, kept for the record: *"the focal term is
numerically negligible at w = 1.0 … this must be resolved before the first
formal run."*)

**R2 — the gate is an upper bound, not a forecast.** MA recall 0.9335 assumes a
perfect refiner inside the candidate region. Nothing here shows the refiner can
get anywhere near it. A realistic first target is a *small* MA IoU gain
(+0.005–0.03), and it is entirely possible that the refiner converges to the
identity.

**R3 — val→test transfer is the known failure mode of this project.** The
locked `0.75·champion + 0.25·C@2000` mix gained **+0.1963 on val** and lost
**−0.4052 on test**. Any val improvement of a few thousandths is inside the
historical bootstrap interval width (≈ 0.6–1.4 points), so **a positive val
delta is not evidence of a real gain**. The honest acceptance test is val-only
and the decision must be made knowing that.

**R4 — test has already been viewed ≥ 5 times** (previous round, recorded as
F4). Generalisation estimates from this split are not trustworthy. This round
did not touch test, which does not repair the earlier damage.

**R5 — false positives cannot be filtered by a fixed rule** (§3.5): elongation,
aspect, fill and solidity are at chance for MA. If the refiner does not learn
local context, there is no fallback heuristic to suppress vessel-like false
positives.

**R6 — MA lesions are 8 × 8 px at the median** (48 px area). At that scale a
single probability step flips many pixels at once, so MA IoU is high-variance
across images (val MA: 1 074 components over 59 images). Small MA deltas on val
should be treated as noise.

**R7 — candidate rule is a one-shot decision.** It was frozen by a pre-declared
rule applied to a pre-declared grid. If the refiner fails, the *correct* next
step is a different architecture, **not** a re-scan of `t` on val.

**R8 — disk/time.** Cache ≈ 800 MB (fine). One-time build ≈ 43 min GPU. Formal
stage-2 ≈ 30 epochs × ≈ 600–650 steps ≈ **1.5–2.5 h** GPU plus
6 val evaluations × ≈ 5.5 min ≈ **2–3 h total**, based on the measured
≈ 0.17 s/step at batch 8 in the smoke. All exclusive; no concurrent GPU jobs.

**R9 — reported per-class numbers for EX/SE** come from the previous round's
`variant B` consistency fix (F2). They are unchanged by this work because the
refiner never touches those channels.

**R10 — the legacy runtime `chdir()`s into `settings['repo']`.** Any *relative*
output path passed to a script that touches the champion silently lands inside
`/root/autodl-tmp/hr_exec_20260919/repo/`. This actually happened once during
this round (the shipping-code audit wrote to
`repo/refiner_v1/audit/candidate_audit_shipping.json`); the file was located,
moved back and verified unchanged. `audit_candidate_rules.py` now resolves
`--out` to an absolute path *before* importing the runtime. Any future script
that takes an output path must do the same.

---

## 8. Recommended next steps (for the main agent)

1. **Decide R1** (focal weight) — this is the only open design question, and it
   must be closed before the first formal step.
2. Build the full patch caches (≈ 43 min GPU):
   `build_patch_cache.py --split train` then `--split val`.
3. Run stage 1 (`configs/stage1.json`) for real, then stage 2
   (`configs/stage2.json`), selecting on val with the frozen rule
   "keep the checkpoint only if **both** val MA IoU and val HE IoU improve".
4. Evaluate with `eval_refiner.py --ckpt ...`; it re-verifies the champion
   baseline and the EX/SE + outside-candidate bitwise invariants on every run
   and refuses to report if they fail.
5. **Do not** adopt anything that only improves val. R3 is the standing reason.

---

## 9. Reproduce this report

```bash
PY=/root/miniconda3/bin/python
cd /root/autodl-tmp/HRDecoder_FullCompetition_20260920

$PY refiner_v1/tests/test_candidate_rules.py                 # 15/15
$PY refiner_v1/tests/test_refiner_units.py                   # 32/32
$PY refiner_v1/audit/probe_dims.py                           # -> audit/input_stats.json
$PY refiner_v1/audit/audit_candidate_rules.py                # -> audit/candidate_audit*.json
$PY refiner_v1/eval_refiner.py --no-refine                   # contract self-check
$PY refiner_v1/build_patch_cache.py --split val   --limit 4  # cache smoke
$PY refiner_v1/build_patch_cache.py --split train --limit 4
$PY refiner_v1/train_refiner.py --stage-config refiner_v1/configs/stage1.json \
      --cache-root refiner_v1/patch_cache_smoke_train \
      --train-images 4 --overfit-steps 40 --out-dir refiner_v1/runs/smoke
$PY refiner_v1/tools/make_manifest.py --root refiner_v1 \
      --out refiner_v1/manifests/REFINER_V1_MANIFEST.json

# --- formal run (2026-09-21): detached supervisor -------------------------
$PY refiner_v1/run_formal_v1.py --dry-run                    # print the plan
bash refiner_v1/tools/start_formal_v1.sh                     # launch detached
cat  refiner_v1/runs/formal_20260921/status.json             # monitor
bash refiner_v1/tools/stop_formal_v1.sh                      # graceful stop
```

File hashes for every source, config, test and audit artefact are in
`manifests/REFINER_V1_MANIFEST.json`.

**This report is the deliverable.** Nothing was trained; the server is still
running and was not modified outside `refiner_v1/`.
