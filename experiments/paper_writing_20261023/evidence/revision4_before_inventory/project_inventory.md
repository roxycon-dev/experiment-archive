# Workspace inventory and reading boundary

Audit date: 2026-10-23. This is an offline inventory of `D:\dataset\experiment`; no training, network/server access, checkpoint loading, or model inference was performed. Original research files were not modified. The paper project is excluded from the research snapshot to prevent recursive self-indexing.

The recursive metadata index contains **35,524 files, 44,478,886,432 bytes**: 5,528 code files, 4,620 result/configuration files, 518 reports, 21,758 images, 38 weights and 42 archives (other extensions: 3,020). These are file-role classifications, not counts of unique experiments or images. Duplicated copies and historical versions remain separate records.

- `evidence/workspace_file_metadata.csv`: every reachable preexisting file, relative path, bytes, modification time, extension and role.
- `evidence/workspace_inventory_summary.json`: exact directory totals, role totals and traversal errors (none).
- `evidence/archive_inventory.csv`: all 42 archive files and listing status.
- `evidence/archive_members.csv`: 9,606 archive members located without extraction. ZIP central directories and TAR headers were inspected; compressed TAR files above 256 MiB and unsupported formats remain metadata-only. No content-completeness claim is made for these files.
- `evidence/source_snapshot_manifest.json`: portable evidence snapshots, original paths and SHA256 hashes.
- `evidence/recomputation_audit.json`: arithmetic and hash checks, distinct from inference verification.

Applicable parent instructions were read from `D:\dataset\AGENTS.md`. No workspace-local AGENTS file was found. Its generic dataset mapping (`MA=0`) is for broader dataset tasks and is not the executed segmentation mapping. This campaign uses output channels **EX, HE, SE, MA**, and fused label IDs **1, 2, 3, 4** (background 0); mappings were taken from the actual protocol and saved metrics.

## Directory roles and evidence priority

| Workspace family | Role and audit treatment |
|---|---|
| `HRDecoder_FinalDecision_20261005` | Primary completed-system evidence. Read closeout README, full report, internal audit, table index, best-model report, original protocol, seed amendment, final selection lock, final-test records, all 34 summarized validation records, integer per-image test counts and historical ledger. |
| `HRDecoder_PaperInnovation_20261006` | Candidate scientific framing. Read innovation positioning; its novelty judgments are hypotheses to compare with verified literature, not proof of priority. |
| `HRDecoder_审查轮_20260916` | Direct MA-correction experiment. Read `h7_test.json`, `h4_test_confirmation.json`, training history and implementation source references. These two test files have different base predictions and cannot be substituted. |
| `HRDecoder_MA残差头与集成_20260915-16`, `hrdecoder_plus3` | Local-window student, frozen reference and earlier MA residual origins. Indexed code/configuration, located through primary ledger and source references; not every source file read. |
| `autodl100` | Historical HRDecoder detail and low-LR continuation implementation/results; indexed and selectively read via primary reports and baseline provenance. |
| `HRDecoder_Ready_20261002` | Frozen corrected HRDecoder export plus ConvNeXt partner. Read `best_pipeline/final_test.json`; its old-model counts differ slightly from September correction audit. |
| `HRDecoder_Q_20261003`, `HRDecoder_NextPlan_20261004` | DDR DINO lineage, full-versus-partial fine-tuning parent and pre-spatial reference. Read final test summary, exact parent identities and corresponding retained validation records. |
| `HRDecoder_Next_20261003` | Historical high-resolution continuation and matched sampling investigations. Read final report and frozen old-pipeline test counts. Test reuse is explicitly documented. |
| `HRDecoder_ArchitectureDesign_20261008` | Later unsuccessful architecture exploration. Read final full report and training-execution trend entry. Later designs do not inherit the locked system's result. Raw and rendered reports remain indexed; detailed failures retained internally. |
| `HRDecoder_NativePyramid_20261011` | Completed v3 training and later untrained JointContext r4 design. Read README, v3 retrospective, completion report, final validation, r4 handoff and machine specification entry. v3 test was not accessed; r4 is not a trained method. |
| Root `architecture_spec.json` | DirectMultiscaleAdapter v11 historical architecture specification; neither the selected S20 EMA system nor NativePyramid v3 nor JointContext r4. |
| `_全实验档案_20260920` | Earlier curated experiment archive; preserved in the later 79-record ledger. Provenance references remain visible rather than treating summaries as independent runs. |
| `HRDecoder_FullCompetition_20260920`, `HRDecoder_FusionAware_20260917`, diagnosis/plan families | Historical alternatives, incomplete/failed experiments and implementation evolution. All file metadata indexed; primary facts located via curated ledger and final closeout. |
| Root `实验全景解读报告.md` | Historical narrative with appended status. Read initial sections and update boundaries; opening sentence misnames EX/SE, while its actual mapping table is correct. Executed mapping takes precedence. |

Directory counts and all remaining smaller families are enumerated in the summary JSON. “Indexed” means metadata covered; “read” means the named entry was inspected or JSON parsed. It does not mean all 35,524 files or all binaries were read. Weight binaries, raw image pixels, duplicate archive payloads, complete large training logs, and every source file were not read.

## Known missing or bounded evidence

1. Original HRDecoder baseline checkpoint, baseline per-image test counts and immutable historical image-byte identity were absent from its audit. The source report/log is available through the saved baseline evidence and retains two-decimal test precision.
2. The selected checkpoint's SHA is documented but this paper audit did not deserialize or rehash the large checkpoint. It verifies stored metric/source identities and archived lock bytes.
3. Patient IDs, patient-level split isolation and cross-center independence remain unverified; image-level identities do not establish them.
4. Some original paths are outside this workspace (`D:\HRDecoder`, `D:\dataset\lesion_segmentation`). They are preserved as source references, not included in the whole-workspace inventory boundary.
5. Large/unsupported archives are explicitly marked metadata-only. No remote-only contents were accessed. Endpoint latency and full-system resource use have not been measured here.
