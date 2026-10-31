# Evidence figure plan (fixed before result plotting)

## fig:ma_size_fp

- Class: evidence-result; role: difficulty breakdown.
- Message: determine whether bypassing the fixed H4 correction changes small-MA lesion detection and false positives per image; improvement is not assumed.
- Evidence hierarchy: native-coordinate one-to-one lesion counts are primary; image bootstrap intervals and denominators provide uncertainty/context.
- Entities: full system and no-H4 system; training-defined area groups <=32, 33–74, >74 pixels; lesion detection and FP/image.
- Relationships/layout: paired grouped bars for detection; companion FP/image comparison. Same image identities and rules throughout.
- Backend: deterministic matplotlib, PDF/SVG/300dpi PNG; data CSV/JSON retained.
- Source: `results/<run>/analysis/lesion_metrics.json` and `scripts/analyze_fixed_run.py`.
- Caption: 8-connected MA components, IoU>=0.10 one-to-one matching, all images included in FP denominator; conditional image bootstrap does not measure seed variation.
- Evidence status: exact-data when the run is complete; partial outputs must carry a visible preview marker.
- Reviewer risk: original HRDecoder unavailable, retrospective DDR test use, new runtime differs numerically from historical environment, no patient IDs.

## fig:ma_examples

- Class: evidence-result; role: failure analysis.
- Message: show actual recovered lesions, persisting misses, and newly introduced false detections under the fixed matching rules.
- Evidence hierarchy: real image crop plus GT/no-H4/full mask panels; exact sid/component IDs and crop coordinates provide traceability.
- Entities: original RGB, MA GT, no-H4 prediction, full prediction.
- Layout: four aligned panels per example; three representative rows in the contact sheet, with up to three separately saved examples per category.
- Backend: deterministic PIL overlays from real stored masks; no generated image content.
- Source: native DDR images/labels and hash-verified per-image prediction archives.
- Caption: first three examples in sid/component order for each predefined outcome category; this is a deterministic sample, not a balanced or exhaustive clinical set.
- Evidence status: qualitative-example derived from exact-data; no claim that examples are independent evidence of mechanism.
- Reviewer risk: tiny lesions, one-pixel boundary changes, greedy visual selection (prevented by preregistered ordering); negative cases may be absent and must be reported as absent.
- Fallback: retain selection manifest and tables if no eligible cases exist; never fabricate examples.
