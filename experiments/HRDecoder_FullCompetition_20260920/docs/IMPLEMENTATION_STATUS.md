# IMPLEMENTATION_STATUS — 契约闸门工具链实施状态

日期：2026-09-20 ｜ 本阶段为收尾验收：未搭服务器、未训练、未联网、未安装新依赖。

## 已完成并验证

| 组件 | 状态 | 说明 |
|---|---|---|
| `tools/contract_common.py` | ✅ | 通道序 EX/HE/SE/MA 单一权威定义、SHA256、manifest JSONL 读写、感知哈希（PIL 可用） |
| `tools/preflight_contract.py` | ✅（修复 2 处） | 四臂配置闸门 A–J 全项检查；修复 U 臂权重缺一静默跳过（fail-open）、manifest 路径 Windows 大小写误判 |
| `tools/audit_manifest.py` | ✅ | manifest 审计 E0–E5（存在性/可读性/二值/valid 语义/唯一性） |
| `tools/build_external_manifest.py`、`tools/inspect_ddr_contract.py`、`tools/audit_manifest.py` | ✅ | `py_compile` 全部通过 |
| `tests/run_contract_smoke.py` | ✅ 12/12 | 合成正例 ×2（preflight/audit）+ 负例 ×7（配对不一致 ×2、test 泄漏、REQUIRED/TBD、缺 image/mask、valid_mode=none 带 mask）+ P0 py_compile + R1 真实正式 preflight |
| 真实配置正式 preflight | ✅ 行为正确 | rc=1，60 个明确 blocker（全部 REQUIRED 占位符），DDR 454/76/227 对齐互斥 |
| 配置 `configs/{H0,H1,U0,U1}*.yaml` | 🔶 locked-intent | 结构与公平锁就绪；REQUIRED 资产字段待填 |

## 阻塞放行的事项（＝正式 preflight 当前 60 个 blocker 的清空条件）

1. `structure.training_entry`：✅ 已解决——四臂统一指向 `train.py`；
   训练核心与入口的实现状态见 `docs/TRAINING_IMPLEMENTATION_STATUS.md`，
   损失/指标口径见 `docs/TRAINING_OBJECTIVE.md`。
2. 权重资产：ImageNet 初值、ConvNeXt-S 编码器、外部预训练 checkpoint——落到本地并回填 `file`+`sha256`。
3. 外部预训练 manifest：用 `tools/build_external_manifest.py` 产出 train-only JSONL，`tools/audit_manifest.py` 通过后回填 H1/U1 的 `external_manifest`（两臂必须同一份文件）。
4. 超参定档：`micro_batch`/`grad_accum`（乘积守恒 =8）、`gradient_checkpointing`、`per_class_valid_masking` 等按显存实测回填。
   （`per_class_valid_masking` ✅ 已实现并置 `true`；其余仍待 GPU 校准。）
5. 全部占位符清零后重跑 `python tools/preflight_contract.py`，**rc=0 且无跳过项**方可作为开机放行依据。
   `--layer readiness` 只回答"能否先开一次短时 GPU 校准"，**不是**正式训练放行依据。

## 退出码契约

preflight：0=通过；1=契约失败；2=用法/IO 致命；3=PyYAML 缺失。
smoke：0=全过；1=有用例失败；4=依赖缺失。

详细测试矩阵见 `docs/PREFLIGHT_TEST_REPORT.md`；契约依据见 `docs/ASSET_AND_INTERFACE_AUDIT.md`。
