# Manifest 审计报告

- 生成时间：2026-09-20T03:17:24.313594+00:00
- Manifest：`D:\dataset\experiment\HRDecoder_FullCompetition_20260920\tests\_tmp\manifests\audit_negative_valid_none_with_mask.jsonl`
- 记录数：2；来源/split 分布：{"IDRiD/train": 1, "MAPLES-DR/train": 1}
- 精确重复对：0；近重复对：0（阈值 6 bit）；DDR 比对图数：0
- **结论：FAIL**（errors=1, warnings=0）

## 降级记录
- jsonschema 未安装：使用内置最小校验器（覆盖 required/type/enum/pattern）

## Errors（必须修复）
- E4 valid: MAPLES-DR_train_20051019_38557_0100/SE: valid_mode=none 但 mask_path/sha256 非空（未标注绝不能挂掩膜）
