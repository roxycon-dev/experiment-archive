# 统一外部病灶数据契约（manifest contract）

版本：1.0（`schema_version: "1.0"`）。定义日期：2026-09-20。
Schema：[manifest.schema.json](manifest.schema.json)；权威常量：[../tools/contract_common.py](../tools/contract_common.py)。
示例：[example_manifest.jsonl](example_manifest.jsonl)（合成数据，不含真实私人路径）。

## 目的

把 IDRiD、MAPLES-DR（以及未来取得授权的 FGADR）的像素级病灶标注统一成一个
机器可检查的 JSONL 契约，供"外部分割预训练（H1/U1）→ DDR 适配"使用，
并让 preflight 能在开机前机械地验证：

1. **四通道顺序**由代码常量唯一给出；
2. **未标注类别 valid=0，绝不能当阴性**；
3. **官方 test 永不默认进入预训练池**；
4. 每个样本可溯源（来源、官方 split、患者/图像 ID、原始尺寸、SHA256）。

## 四通道顺序（最高优先级契约）

```
通道 0 = EX  (hard exudates, 硬性渗出)
通道 1 = HE  (hemorrhages, 出血)
通道 2 = SE  (soft exudates / cotton wool spots, 棉绒斑)
通道 3 = MA  (microaneurysms, 微动脉瘤)
```

- 权威定义：`tools/contract_common.py` 的 `LESION_CHANNEL_ORDER = ("EX","HE","SE","MA")`。
- 与 DDR 的映射：DDR 融合标签 class id 为 bg=0, EX=1, HE=2, SE=3, MA=4
  （`D:\HRDecoder\tools\convert_dataset\ddr.py:39-44`）；DDR 评估端把 sigmoid
  通道 `i-1` 与 class `i` 对齐（`lesion_metric.py:63-79`）。
- **结论：本契约通道序与 DDR 评估序一致，无需置换**；映射仍由
  `DDR_CLASS_ID_OF_CHANNEL` 显式给出（channel i → DDR class i+1），禁止任何
  消费端隐式对齐。
- 其他病变类别（IDRiD 无；MAPLES-DR 的 IRMA/VB/NV 等）**不映射进四通道**，
  只允许出现在 `notes`，避免语义污染。

## 每样本字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `schema_version` | 是 | 固定 `"1.0"` |
| `sample_id` | 是 | 全局唯一，约定 `<source>_<official_split>_<image_id>` |
| `source` | 是 | `IDRiD` / `MAPLES-DR`（白名单，见 `ALLOWED_SOURCES`） |
| `source_version` | 否 | 数据集版本；未知写 null，禁止编造 |
| `official_split` | 是 | 数据集官方划分 `train/val/test`，逐字保留 |
| `patient_id` | 否 | 数据集不提供时写 null + `patient_id_source=null`；禁止冒充 |
| `image_id` | 是 | 官方文件名 stem |
| `original_width/height` | 是 | 原图尺寸（像素），与实际解码尺寸必须一致 |
| `image.path/sha256/format` | 是 | 原图路径、精确 SHA256、扩展名 |
| `lesions.{EX,HE,SE,MA}` | 是 | 恰好四键，见下表 |
| `perceptual_hash` | 否 | `{ahash,dhash,phash}` 64bit hex，由 audit 工具补齐 |
| `notes` | 否 | 备注文本 |

`lesions.<class>` 子字段：

| 字段 | 说明 |
|---|---|
| `annotated` | 该类是否被官方标注 |
| `valid_mode` | `full`（全图有效，隐式全 1）/ `file`（valid 栅格在 `valid_mask_path`）/ `none`（未标注，valid 全 0） |
| `mask_path` / `mask_sha256` | 该类二值掩膜；`valid_mode=none` 时必须为 null |
| `valid_mask_path` | 逐像素有效栅格（非零=有效）；`full`/`none` 时为 null |
| `positive_pixels` | 可审计的阳性像素计数（可 null） |

### valid 语义（防止"未标注被当阴性"）

- `valid_mode=none`：该类在本图**没有人工标注**。消费端构造训练目标时该类
  通道的损失必须整图屏蔽（valid=0），**不得**把空区域当阴性监督。
- `valid_mode=full`：整图均被标注，可作为阴性/阳性监督。
- `valid_mode=file`：仅 `valid_mask_path` 非零像素参与监督。
- 现有 HRDecoder 的 `BinaryLoss._make_one_hot`（binary_loss.py:11-37）没有
  per-class ignore 语义（0 和 255 都按负样本处理），因此 valid 掩蔽必须由
  新训练入口实现，不能依赖旧损失代码。preflight 会检查该语义的一致性。

## 目录布局（manifest 自身不强制）

```
external_data/                     # 建议布局（可配置）
├── IDRiD/
│   ├── images/train/*.jpg         # 官方 537 张原图
│   └── labels/train/{EX,HE,MA,SE}/*.tif   # 官方逐类 TIF
├── MAPLES-DR/
│   ├── images/{train,test}/*.jpg  # MESSIDOR 原图（单独取得）
│   └── anomalies/{train,test}/    # 官方标注（布局见 build 工具）
└── manifests/
    ├── external_train_v1.jsonl
    └── external_all_v1.jsonl      # 含 official test 时必须显式命名
```

- 官方 test 样本允许出现在 manifest 中（`official_split: "test"` 保留语义），
  但**默认只构建 train**；要包含 test 必须显式传 `--include-splits test`，
  且 preflight 在配置引用的预训练 manifest 中发现 official_split=test 时直接
  判失败。

## 工具链

| 工具 | 作用 | 失败语义 |
|---|---|---|
| `tools/build_external_manifest.py` | 从官方目录构建 manifest；缺图/缺掩膜/尺寸不符 → fail closed | 非零退出 + 明确错误 |
| `tools/audit_manifest.py` | 存在性/尺寸/二值性/valid 语义/泄漏/精确与感知哈希重复检查，输出 JSON+MD | error → 非零退出；warning 不阻塞 |
| `tools/inspect_ddr_contract.py` | 只读解析 DDR_split_613 与标签契约（test 标签只做文件级检查，不读像素） | 结构异常 → 非零退出 |
| `tools/preflight_contract.py` | 一条命令检查四配置 + manifest + split + 禁止 test 训练/选型 + 类序 + valid | 任何 blocker → 非零退出 |

## 依赖策略

- 必需：Python 3.9+，numpy，Pillow（本机已具备）。
- 可选：`jsonschema`（schema 校验更严格）、`imagehash`（交叉验证自实现哈希）。
  缺失时自动降级为内置校验器 / 自实现 aHash+dHash+pHash，并在报告 `degradations`
  字段记录；**降级不阻塞，但静默跳过被禁止**。
- 不新增全局依赖；不下载任何需要登录/许可协议的数据。
