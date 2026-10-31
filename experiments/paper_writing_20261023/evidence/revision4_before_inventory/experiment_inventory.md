# Experiment identity, fairness and test-use inventory

Internal audit; failures and historical designs are not intended as manuscript tables. Primary source snapshots and exact numeric selectors are in `evidence/` and `numeric_evidence.json`.

## Selected result: what “s20ema” means

The selected branch is **S_seed42, 20,000 completed updates, exponential moving average weights**. Its checkpoint identity is `c933256b07556689a7fd199dd80e50d4d0c27eb6b9ea782713d95ec51e6d8d62`. The actual reported system is

`P_final = .75 P_frozen + .25 P_spatial_EMA = .5625 P_corrected_HRDecoder + .1875 P_ConvNeXt + .25 P_spatial_EMA`.

This is a probability ensemble with a nonlinear MA correction inside the HRDecoder branch, not a single trained lightweight model. “s20ema” is a checkpoint alias, not an architectural method name. Manuscript method names should describe HRDecoder improvements, class-selective correction and spatial adaptation; historical run labels remain internal provenance.

S initialization: DDR-only Q branch at 16k → full-12-block continuation for another 12k (F12 **raw** parent) → spatially adapted branch with fresh optimizer. The F12 parent SHA is `68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728`. MA-corrected HRDecoder and ConvNeXt are prediction partners, not parameters used to initialize S. S has no TJDR, IDRiD or MAPLES pixel supervision in this lineage. Large-scale DINO pretraining remains separate pretraining provenance, not clinical external validation.

Dataset: custom DDR 454 train / 76 validation / 227 test images (757 total), channels EX/HE/SE/MA, original label IDs 1/2/3/4. Test mIoU is the mean of four dataset-level IoUs after summing per-image integer TP/FP/FN; background excluded. Threshold is strictly `p > .5`, FP32 sigmoid/probability fusion, original-image coordinates. The S branch uses native 1024 windows with center768 and halo128; old HRDecoder mixture includes horizontal flip averaging. It is incorrect to describe the complete ensemble as no-TTA.

## Training and selection, with version boundary

The October 5 original protocol proposed two seeds. `single_seed_amendment.json` records the explicit revision to **seed42 only** and cancellation of the second parent/arms. The executed result is exploratory single-seed evidence. One cannot claim two-seed replication from the original plan or from older Q runs with two seeds.

All four final arms share the same within-seed F12 parent and DDR window/augmentation streams, eight DDR windows per update (microbatch2 × accumulation4), four foreground and four uniform. AdamW, betas .9/.999, eps1e-8, weight decay .01 except bias/norm, gradient clip1.0; 500-update warmup and cosine decay to 1% of peak over 32k. ViT peak block learning rates `1e-5*.8**(11-b)`, old decoder5e-5, new spatial modules1e-4; EMA decay .999 after each successful optimizer update. The executed training loss is .5 BCE + .5 Dice with validity masking and class mean. Training uses bfloat16 forward and FP32 loss/output accounting. No new loss is inferred from later experiments.

Spatial on/off is C versus S. External-supervision off/on is C/S versus D/SD. D/SD add TJDR windows, independent source decoder and compute; they are **not equal-compute pure data ablations**. TJDR uses 439 accepted of 448 pairs after nine exclusions; source mapping MA/SE differs from DDR and was explicitly reordered. This package does not enter the selected S lineage.

All 4k validation nodes are retained, but only 20k/32k and three output formulas (single, .75 frozen+.25 branch, half fusion) were eligible for deployment: 4 arms ×2 endpoints ×3 formulas =24 recipes. Seed42 validation improvement over its prior reference must meet .5pp; ranking uses unrounded values and registered ties. The selected S20 EMA fixed25 system had validation **45.321831227388415%**. Its lock was frozen before the one final campaign test. Diagnostic 16k highs are not deployable endpoints. Fixed32k is the scientific terminal comparison; selected20k is deployment selection, and both are shown separately.

## Result tables available to authors

| Portable file | Meaning |
|---|---|
| `evidence/system_overview.csv` | Historical original HRDecoder, HRDecoder mixture before/after paired MA correction, distinct corrected frozen re-export, ConvNeXt partner, pre-spatial reference, selected system; four class IoUs and macro IoU. System progression is descriptive, not single-factor causal attribution. |
| `evidence/test_class_metrics.csv` | Selected system and previous system; exact 227-image integer totals, IoU/Dice/pixel precision/recall. |
| `evidence/ma_correction_metrics.csv` | Direct historical four-member-mixture MA correction before/after, with non-MA count identity checked. |
| `evidence/registered_endpoints.csv` | All C/S/D/SD 20k and 32k three-formula validation outcomes. |
| `evidence/validation_metrics.csv` | All 34 retained validation records, including two reference parents; same split and explicit formulas. |
| `evidence/test_per_image_counts.csv` | Paired selected/previous integer counts for all 227 images ×4 classes. |
| `evidence/bootstrap_draws.csv` | Exact 2,000 conditional bootstrap draws, regenerated from stored counts. |

At20k spatial-minus-continuation single-branch validation difference is1.2459498985952138pp, whereas fixed25 system difference is0.14558808247063837pp. At32k these are1.0450595416663901pp and0.09735083292878244pp. These are single-seed validation contrasts, not statistical significance claims or matched test contrasts.

The selected test system has45.86038383266797% mIoU, MA29.243546560152616%, MA precision50.04591075459427%, MA recall41.298548223296834%. Prior system mIoU45.72155710642771%, improvement0.1388267262402607pp; its conditional paired-image95% interval is[-0.2625660524346422,0.6519375896498986]pp and includes zero. MA IoU improves by only0.08246383787530931pp relative to that prior system; its recall decreases. Do not imply every improvement increases sensitivity.

Original HRDecoder test42.20%, MA24.11% are rounded historical values. The total approximate3.6604pp mIoU and5.1335pp MA gains involve multiple architectures, training changes, probability fusion and TTA. They cannot be assigned to spatial adaptation or MA correction alone.

## MA correction: paired result and export discrepancy

`ma_correction_source.json` is the primary September `h7_test.json`: same four-member flip mixture,43.228924245323747→44.2115035614162% mIoU; MA24.72547127882145→28.65578854319127%. Non-MA TP/FP/FN are identical. Gains are0.9825793160924523pp and3.930317264369819pp respectively, exactly related by a factor of four. This supports class selectivity and an observed correction effect on that retained test outcome.

Later frozen export (`frozen_partner_source.json`) reports44.21100725782683% and slightly different integer counts; it is the version used by the ConvNeXt partner. Treat this as a separate exported artifact, retain the discrepancy, and do not silently round it into the earlier paired result. The mechanism training uses local/frozen-reference inputs, whereas deployment applies it to a broader mixture; model-disagreement input is not a calibrated uncertainty estimate. No matched ablation isolates every input channel or proves why the correction helps.

## Test-use and fairness audit

The October 5 final recipe is locked before its campaign test, but the same custom test has been evaluated repeatedly in preceding rounds. Reports from October2/3 explicitly acknowledge this history; some historical choices retained the older deployment after candidate test failure. Accordingly the result is an **internal, historically reused benchmark**, not an untouched holdout or external validation. No additional test inference or selection was performed for the paper.

Original baseline and final system share the documented split name and metric structure, but baseline historical checkpoint/sample-byte snapshot and per-image integers were not obtained. Differences in pretraining, ensemble size, training exposure, inference geometry and TTA remain. Do not label this a controlled architecture comparison. Patient overlap is unknown; do not claim either confirmed leakage or confirmed patient independence.

## Historical/negative evidence retained internally

`evidence/historical_experiment_ledger.json` and CSV preserve all79 curated records through October5, including earlier45-record archive snapshots. Status, source paths, parents and missing extraction fields are retained as recorded; plans are not inferred as completed runs.

Later experiments are indexed separately:

- `HRDecoder_ArchitectureDesign_20261008/final_report/HRDecoder完整实验报告.md`:11 architecture rounds, many stopped by screening. CET40k validation38.313759%, best40.487643% at32k. The last IoUCorrection round stopped at4k,26.269023%, with no40k or test result. These runs do not establish a single causal failure mechanism.
- `HRDecoder_NativePyramid_20261011/server_execution_20261016/completion_20261017/completion_report.json`: distinct NativePyramid v3 run_seed42_launch02, source `d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627`,40k raw validation39.42555100502815%, secondary best36k40.27704590850366%, no test. Remote checkpoint not downloaded in that completion audit. Do not transplant the selected ensemble score to this design.
- `HRDecoder_NativePyramid_20261011/review_20261018/jointcontext_spec_r4.json`: JointContext r4 specification/reference CPU checks only; complete model not implemented or trained. No model-performance result.
- Root `architecture_spec.json`: DirectMultiscaleAdapter v11, changed architecture and private learning rate together; no accuracy guarantee. Distinct from all the above.

Detailed negative evidence remains an internal ledger; no failed-experiment list is required in the paper or supplement under the current author scope. It constrains claims of universal benefit and method novelty without becoming a chronological manuscript.
