# Claim–evidence map

All percentages use original-image global pixel counts followed by macro class averaging unless explicitly marked historical rounded. Exact selectors and portable source paths are maintained in `numeric_evidence.json`. “Confirmed” below means supported by saved artifacts/offline arithmetic, not independent model rerunning.

| ID | Permissible claim | Located evidence | Boundary / wording to avoid |
|---|---|---|---|
| C1 | The implemented system extends HRDecoder with native-window local prediction, class-selective MA correction, and complementary ConvNeXt and spatially adapted DINO predictions. | `evidence/p0_identity.json`, `selection_lock.json`, actual implementation audit; original paths in source manifest | A system contribution; not a single lightweight model or all-end-to-end architecture. |
| C2 | The final selected system achieves45.8604% test mIoU, EX55.1213%, HE46.1978%, SE52.8789%, MA29.2435%. | `locked_test_source.json` `/locked_test_miou_pct`, `/test_per_class/*/iou_pct`; 227 paired per-image integer counts | Single seed42, historically reused internal DDR test. No external/untouched test or state-of-the-art claim without common protocol. |
| C3 | Compared with the historical HRDecoder report, mIoU increases from42.20% to45.8604%, approximately3.6604pp; MA24.11% to29.2435%, approximately5.1335pp. | `historical_baseline_source.json` `/historical_test`; locked source above | Baseline rounded, unreproduced; cumulative system improvement. No single-module attribution. |
| C4 | A small MA-only correction head preserves the other three channels by construction; the saved paired test has identical non-MA TP/FP/FN. | `ma_correction_source.json` `/fused/counts`, `/final/counts`; source implementation and `recomputation_audit.json` | Structural noninterference applies only to this correction stage, not all later probability fusion. Bound on residual is not an accuracy guarantee. |
| C5 | In the historical paired four-member mixture, correction changes MA IoU24.7255→28.6558%, overall43.2289→44.2115%. | `ma_correction_source.json` `/fused`, `/final`, `ma_correction_metrics.csv` | Exact historical paired output, not a second unseen evaluation. Later44.2110 frozen export is separate. No isolated value of disagreement input or proof of calibrated uncertainty. |
| C6 | Spatial adaptation improves single-branch validation by1.2459pp at20k and1.0451pp at32k relative to matched continued fine-tuning. | `validation_source.json` `/paired_registered_endpoint_contrasts/{0,3}/contrasts/S_minus_C_pp` | Same seed and DDR exposure, validation only; no multi-seed significance. Comparison changes spatial-module parameterization, compute and associated parameters. |
| C7 | With the fixed prediction partner, corresponding spatial gains shrink to0.1456pp at20k and0.0974pp at32k. | `/paired_registered_endpoint_contrasts/{1,4}/contrasts/S_minus_C_pp` | Do not present single-branch gain as deployed system gain. |
| C8 | Final system improvement over previous retained system is0.1388pp; conditional paired-image95% interval[-0.2626,0.6519]pp includes zero. | `locked_test_source.json` `/gain_vs_R_pp`, `/conditional_bootstrap_from_original_final_test`; exact regenerated draws | Fixed selected models, image units,2,000 draws seed100303; ignores model selection and seed/patient uncertainty; no statistically significant claim. |
| C9 | MA remains weakest and pixel recall is41.2985%; correction/system gains do not establish clinical screening sensitivity. | `/test_per_class/MA`, `test_class_metrics.csv` | Pixel sensitivity is not lesion-level sensitivity or patient diagnosis. MA recall decreases versus previous system. |
| C10 | Deployment was selected on validation among24 registered recipes, using EMA, and frozen before one campaign test. | `selection_lock.json`, `single_seed_amendment.json`, `original_protocol.json` plus revision | Original two-seed plan was amended to one seed. Final lock does not undo historical test reuse. |
| C11 | Final spatial branch lineage uses DDR-only supervised training. | `Q_parent_identity.json`, `original_protocol.json` `/initialization`, shared parent identity | Does not mean no external pretraining: DINOv3 uses large-scale pretraining. External-supervision experiments did not initialize selected S. |
| C12 | The complete system is not yet established as fast, clinically robust, or broadly generalizable. | Missing end-to-end latency; custom split/test history; no patient-level audit/external data in selected evaluation | Treat these as evidence gaps, not measured negative clinical results. |

## Discrepancies and checks

- September paired MA correction mIoU44.2115035614162% versus later frozen re-export44.21100725782683%: preserve both exact artifacts and their distinct counts. No causal explanation of the small difference is established here.
- Root panorama opening text mislabels EX/SE; actual campaign protocol and all selected metric records use EX=hard exudates, HE=hemorrhages, SE=soft exudates, MA=microaneurysms.
- Parent AGENTS generic class map is inapplicable to this executed segmentation campaign.
- Two-seed original protocol versus single-seed final execution: explicitly resolved by recorded amendment, not silently averaged with older runs.
- NativePyramid v3 and JointContext r4 are distinct later projects; neither can inherit the selected ensemble score.
- Offline checks recompute test metrics and paired bootstrap, validate saved validation source hashes and integer metric formulae, and confirm MA correction non-target count identity. They do not rerun models or reconstruct missing baseline integers.

## Explicitly unsupported claims

Universal superiority; clinical diagnostic sensitivity; statistically significant final gain; patient-independent/never-used holdout; state of the art under an official DDR split; a new general attention paradigm; principled calibrated uncertainty from raw expert disagreement; causal benefit of each input or loss; benchmark efficiency of the entire ensemble; new spatial module explaining the full historical baseline gain; trained JointContext/r4 results; two-seed replication of S.
