# PREFLIGHT_TEST_REPORT — 契约闸门测试报告

日期：2026-09-20 ｜ 范围：`tools/preflight_contract.py` + `tools/audit_manifest.py` 合成冒烟验收
执行方式：`python tests/run_contract_smoke.py`（离线、无网络、不训练、不读任何图像像素）

## 结果总览

**12/12 用例通过（退出码 0）。** 真实配置正式 preflight 以明确 blocker 非零退出（设计行为）。

| 用例 | 验证点 | 结果 |
|---|---|---|
| P0 | `py_compile`：tools/*.py（5 个）+ tests/*.py（1 个）全部可编译 | PASS |
| P1 | preflight 正式模式：合成 train-only 四配置（占位符全部替换为可验证合成资产）rc=0、零错误 | PASS |
| N1 | H0↔H1 equivalence_lock 锁定字段不一致 → rc=1，定位到 first-difference | PASS |
| N2 | U0↔U1 锁定字段不一致 → rc=1 | PASS |
| N3 | 预训练 manifest 混入 official_split=test（泄漏）→ rc=1，报"只允许 official_split=train" | PASS |
| N4 | 正式模式遇 `REQUIRED` 占位符 → rc=1 | PASS |
| N4b | 正式模式遇 `TBD` 占位符 → rc=1 | PASS |
| P2 | audit：train-only 合成 manifest（图像/掩膜/SHA256/感知哈希真实对应）rc=0、零错误 | PASS |
| N5 | audit：image 文件缺失 → rc=1（`image missing`） | PASS |
| N6 | audit：mask 文件缺失 → rc=1（`mask missing`） | PASS |
| N7 | audit：valid_mode=none 却携带 mask_path/sha256 → rc=1 | PASS |
| R1 | 真实四配置（configs/*.yaml）正式 preflight：资产未齐 → rc=1，60 个明确 blocker | PASS |

## 真实配置正式 preflight（R1 细节）

- 退出码 **1**（契约失败，非用法错误）；`mode=formal`。
- **60 个 blocker**，全部为 REQUIRED 占位符（training_entry、权重 file/sha256、
  external_manifest、micro_batch/grad_accum 等），逐条定位到 JSON 路径；另有 10 个跳过项显式记录，无静默。
- 真实 DDR（`DDR_split_613`）文件名级检查通过：train/val/test = **454/76/227**，
  四臂 stem 对齐、三 split 互斥，无非占位符错误。
- 结论：资产未齐时闸门 fail closed，**不得开机训练**；占位符补齐并复核后重跑。

## preflight 修复记录（本次发现并修复 2 处 fail-closed 缺口）

1. `check_initialization`：U 臂 `encoder_weights_file`/`encoder_weights_sha256`
   只提供其一时，原先**静默跳过**哈希校验（fail-open）。修复为报错"必须成对提供"。
2. H1/U1 `external_manifest` 同文件比较原先区分大小写，Windows 下
   `D:/x.jsonl` 与 `d:/x.jsonl` 会误判为两份文件；已加 `os.path.normcase` 归一。

修复后 P1 正例与全部负例回归通过（见上表）。

## 测试脚本约定

`tests/run_contract_smoke.py` 自身只读写 `tests/_tmp`（白名单守卫：规范化路径 +
前缀校验 + 拒绝 `..`）；R1 由 preflight 子进程按契约只读配置与真实 DDR 文件名清单。
退出码约定：0=全部通过；1=有用例失败；4=PyYAML/Pillow 缺失（显式报告，绝不静默）。
