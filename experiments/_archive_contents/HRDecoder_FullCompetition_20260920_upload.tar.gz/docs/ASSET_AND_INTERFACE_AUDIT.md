# HRDecoder 资产与接口审计（只读盘点）

审计日期：2026-09-20。审计方式：直接读取代码与原始数据文件，逐项给出文件路径与行号。
所有结论以代码与原始结果为准；与既有报告冲突之处在第九节显式记录。
本文件只读盘点，未修改 D:\HRDecoder 与任何旧实验目录。

## 1. 类别定义与通道顺序（最重要契约）

| 事实 | 位置 |
|---|---|
| 数据集类名 `CLASSES = ['bg', 'EX', 'HE', 'SE', 'MA']`（含背景共 5 类） | `D:\HRDecoder\mmseg\datasets\lesion_dataset.py:16` |
| 调色板注释：EX=红, HE=绿, SE=黄, MA=蓝 | `lesion_dataset.py:28-34` |
| DDR 逐类 TIF → 融合 PNG 的 class id：`EX=1, HE=2, SE=3, MA=4`（bg=0） | `D:\HRDecoder\tools\convert_dataset\ddr.py:39-44` |
| IDRiD 逐类 TIF → 融合 PNG 的 class id：同上 | `D:\HRDecoder\tools\convert_dataset\idrid.py:33-38` |
| 解码头输出 4 个通道（`num_classes=4`，无背景通道），`use_sigmoid=True` | `D:\HRDecoder\configs\_base_\models\hrdecoder_fcn_hr18.py`（decode_head 段） |
| 评估时 sigmoid 路径断言 `pred_logit.shape[0] == num_classes - 1`（=4 通道），并按 `label == i` 与通道 `i-1` 对齐 | `D:\HRDecoder\mmseg\core\evaluation\lesion_metric.py:63-79` |

**推导出的通道契约（本工作区代码常量的依据）：**
- 整图评估的 sigmoid 输出通道顺序为 **通道0=EX, 通道1=HE, 通道2=SE, 通道3=MA**（对应 DDR class id 1..4）。
- 因此统一外部契约采用 `EX/HE/SE/MA` 通道序后，与 DDR 评估序**完全一致，无需置换**；但映射必须由代码常量显式给出（见 `tools/contract_common.py` 的 `LESION_CHANNEL_ORDER` 与 `DDR_CLASS_INDEX`），不允许隐式对齐。
- 主指标 `mIoU` 只对最后 4 类（EX/HE/SE/MA）取平均，背景不参与：`lesion_dataset.py:94`（`iou[-4:]`）。

## 2. 标签编码与 ignore 规则

| 事实 | 位置 |
|---|---|
| `CustomDataset` 默认 `ignore_index=255`，`reduce_zero_label=False` | `D:\HRDecoder\mmseg\datasets\custom.py:88-89` |
| 整图评估加载融合 PNG，`flag='unchanged'`，不做任何值重映射 | `custom.py:235-246` |
| `lesion_metrics(..., ignore_index=...)` **接受** ignore_index 参数但函数体内从未使用 | `lesion_metric.py:173-205`（逐行核对，无 ignore 分支） |
| 训练损失 `BinaryLoss._make_one_hot(ignore=(0,255))`：把 0 和 255 都压入被丢弃的附加通道 → **训练中没有真正的 ignore，255 像素按四类病灶的负样本被监督** | `D:\HRDecoder\mmseg\models\losses\binary_loss.py:11-37` |
| 实测 DDR_split_613 融合标签值域（抽样 16 张，train/val/test 各抽样）：值域 ⊆ {0,1,2,4}，**未发现 255、未发现 3**（SE 极稀有，抽样未覆盖到 3 属正常） | 本次实测，2026-09-20 |

**工程含义（写入契约）：**
- DDR 融合标签实际是纯 0..4 编码，评估端忽略规则形同虚设；训练端 255 会被当负样本。
- 外部数据（IDRiD/MAPLES-DR）转成本契约时，**绝不能把"未标注"编码成 0 或 255 塞进融合标签**，必须用 `valid_mask` 表达（valid=0 的类别在损失端需要显式屏蔽，不能依赖现有 BinaryLoss 的行为）。本工作区在 `data_contract/README.md` 与 preflight 中固化该规则。
- 评估端：valid=0 的类别在整图计数中应从该类的 GT 正样本与预测正样本中同时剔除（本工作区未来训练入口需自行实现；现有 HRDecoder 评估代码不处理此情况，见第九节冲突 3）。

## 3. 数据集接口（CustomDataset / LesionDataset）

| 项 | 值 | 位置 |
|---|---|---|
| 图片后缀 `.jpg`，掩膜后缀 `.png`（融合图） | `img_suffix='.jpg'`，`seg_map_suffix='.png'` | `custom.py:82-84` |
| 掩膜文件名 = 图片 stem + `.png`，放 `ann_dir` 平铺目录 | `custom.py:124-139`（load_annotations） | 同左 |
| 类别可由 config 的 `classes` 列表覆盖 | `custom.py:248-274` | 同左 |
| `LesionDataset.evaluate` 固定走 `_evaluate`（含 F1/PPV/召回/AUPR/MAE），metric 限 `mIoU`/`mIoU2` | `lesion_dataset.py:39-60` | 同左 |
| 训练时验证由 mmseg `EvalHook` 调 `dataset.evaluate`，即上表整图口径 | `baseline_split613.py:57`（`evaluation=dict(interval=4000, save_best='mIoU')`） | 同左 |

## 4. 训练入口与模型结构（HRDecoder 臂 H0/H1 相关）

| 项 | 值 | 位置 |
|---|---|---|
| 训练入口 | `python tools/train.py <config>`（mmcfg + `build_train_model` + `train_segmentor`） | `D:\HRDecoder\tools\train.py:145-175` |
| seed 来自 config 或 CLI `--seed` | `train.py:51-55, 134-142` | 同左 |
| 冠军配置链 | `configs/lesion/exp/final_combo_split613.py` → `baseline_split613.py` → `_base_`（hrdecoder_fcn_hr48 + hr_ddr_2048 + default_runtime + sgd + poly10warm） | 各文件 `_base_` 段 |
| HRDecoder 多裁剪训练：`hr_settings=(hr_scale 1024², scale_ratio 0.75-1.25, divisible 8, crop_num 4, hr_loss_weight 0.1, lr_loss_weight 0, fuse_mode 'simple')` | `baseline_split613.py:15-33`；裁剪采样实现 `D:\HRDecoder\mmseg\models\segmentors\HRDecoder.py:64-68, 123-126, 153-160` | 同左 |
| 冠军损失：`BinaryLoss(loss_type='ce_dice', smooth=1e-5)` + DetailBranch(32ch) + DetailHead(num_classes=4) | `configs/lesion/exp/final_combo_split613.py:10-34` | 同左 |
| 优化器（冠军）：SGD lr=0.005, momentum 0.9, wd 5e-4；40000 iters；batch 2 | `baseline_split613.py:36, 52-56`（samples_per_gpu=2 在 hr_ddr_2048.py:56） | 同左 |
| ImageNet 预训练声明：`pretrained='open-mmlab://msra/hrnetv2_w48'`（HRNet-W48 官方权重） | `configs/_base_/models/hrdecoder_fcn_hr48.py` | 同左 |
| 数据 pipeline（train）：Resize 2048(ratio 0.5-2.0) → RandomCrop 2048(cat_max_ratio 0.75) → Flip 0.5 → Rotate ±45° → Normalize(mean [81.205,50.636,21.216], std [76.252,48.798,21.625]) → Pad | `configs/_base_/datasets/hr_ddr_2048.py:15-38` | 同左 |
| 归一化均值/方差文件头注明按 DDR 统计 | `hr_ddr_2048.py:3-7` | 同左 |

**对新臂的含义：** H1 的"外部预训练 → DDR 适配"必须以 `load_from`（或等效机制）注入预训练 checkpoint；`pretrained` 字段只接受 backbone ImageNet 权重，不能携带已在 DDR 上训练过的 decoder。这与主方案"官方已在 DDR 上训练过的 checkpoint 不能冒充 ImageNet 初始化"一致。

## 5. 测试/整图验证入口

| 项 | 值 | 位置 |
|---|---|---|
| 测试入口 | `python tools/test.py <config> <checkpoint> --eval mIoU`，`--inference-mode {same,whole,slide}` | `D:\HRDecoder\tools\test.py:37-158` |
| 默认整图模式：`test_cfg = dict(mode='whole', compute_aupr=True)` | `configs/_base_/models/hrdecoder_fcn_hr18.py`（test_cfg 段） | 同左 |
| `whole` 推理后按 `ori_shape` 双线性还原到原始分辨率再计数 | `lesion_encoder_decoder.py:101-123`（rescale 分支） | 同左 |
| sigmoid 输出：`compute_aupr=True` 时返回 `(logits[4,H,W], use_sigmoid, True)` 元组；否则 `>0.5` 二值化 | `lesion_encoder_decoder.py:162-178`；`logit_activation` 在 :191-203 | 同左 |
| 整图验证=在原始尺寸 GT 上跨全部图片累计 TP/FP/FN 后再算 IoU/F1/PPV/召回（不是逐图平均） | `lesion_metric.py:105-170`（`sigmoid_metrics` 累计逻辑） | 同左 |
| 阈值固定 0.5：`SIGMOID_THRESH = 0.5` | `lesion_metric.py:1` | 同左 |
| `--test-set` 会把数据路径中的 `val` 字符串替换为 `test`（评估 test 的开关） | `test.py:171-174` | 同左 |

**禁止事项的代码级落点：** DDR test 用于选型的路径只有两条——训练时把 `data.val` 指到 test，或用 `--test-set` 反复评估。新工作区配置在 `checkpoint_selection.allowed_splits=[val]` 中显式锁定 val-only，preflight 会扫描配置字段禁止任何 `images/test`、`labels/test`、`split: test` 出现在训练/验证/选型字段。

## 6. 固定 DDR split（D:\HRDecoder\data\DDR_split_613）

| 项 | 值 |
|---|---|
| 来源 | 757 张 DDR 全量按 SE 分层 6:1:3 重划分，seed=42（`split_report.md` 头注；`split_assignment.json` 的 `"seed": 42`） |
| 数量（本次实测） | train 454 / val 76 / test 227（images 与 labels 完全一致） |
| 目录结构 | `images/{train,val,test}/*.jpg` + `labels/{train,val,test}/*.png`（融合掩膜）+ `labels/train|val/{EX,HE,MA,SE}/*.tif`（每类原始 TIF，各 454/76/…张，仅存档未参与训练接口） |
| 互斥性（本次实测） | train∩val = train∩test = val∩test = ∅；每个 split 的 image stem 与 label stem 一一对应 |
| 已知异常 | `split_assignment.json` 的 `anomalies` 字段列出数十条 "fused png vs tif mismatch"（融合 PNG 与原始 TIF 像素数差 1~数百），划分脚本已记录并保留 PNG 为准 |
| 类别分布 | 见 `split_report.md`/`split_stats.csv`：train EX 66.74%图/0.2466%像素，HE 80.62%/0.3781%，SE 31.50%/0.0566%，MA 74.45%/0.0225% |
| 分层标记 | tierA/B/C（518/119/120） |

注意：`split_assignment.json` 的 anomalies 说明"PNG 为训练真值、TIF 仅供参考"这一决定在旧实验中已做出；新工作区沿用该口径并在 `inspect_ddr_contract.py` 输出中复述。

## 7. 旧 IDRiD 转换器（两份副本逐字节一致）

位置：`D:\HRDecoder\tools\convert_dataset\idrid.py` 与 `D:\dataset\experiment\hrdecoder_plus3\repo\tools\convert_dataset\idrid.py`（`diff` 验证一致）。

约定与行为：
- 期望目录（:11-28）：`IDRiD/images/{train,test}/*.jpg`；`IDRiD/labels/{train,test}/{EX,HE,MA,SE}/*.tif`。
- 掩膜文件名（:58）：`<image_stem>_<CLASS>.tif`（如 `IDRiD_001_EX.tif`）。
- class id（:33-38）：EX=1, HE=2, SE=3, MA=4。
- 正样本判定（:60）：`each_class_gt == 76`，四类统一阈值。76 是 IDRiD 官方逐类 TIF 在灰度读入下的正样本灰度值（纯红 (255,0,0) 经 OpenCV luma 0.299R≈76 舍入）。该值依赖"每类一张独立掩膜"前提，**不是**从融合彩色图猜 class id。
- 输出（:62-63）：`labels/<subset>/<image_stem>.png` 融合图，直接叠加（后写的类覆盖先写的类）。
- 已知局限（记录，不在本轮修复）：叠加顺序 EX→HE→SE→MA 使 MA 最后写入，重叠区域最终标为 MA；重叠像素天然存在时各类别像素数会互相影响。新工作区契约改为四类独立掩膜 + valid mask，不复刻该行为。
- `convert_dataset/ddr.py`（:66）对 DDR 使用 `==255`，与 IDRiD 的 76 不同——两个数据集正样本编码不同，适配层必须按数据集区分，禁止共用阈值。

## 8. 外部数据资产现状（2026-09-20 实测）

| 资产 | 状态 | 证据 |
|---|---|---|
| IDRiD 原图（537 张）与官方逐类 train masks | **本机未找到** | `D:\dataset` 及其三层子目录内 `*IDRiD*` 只命中 `D:\dataset\best_segmentation\verification\` 下的对比派生图（非原图/官方掩膜） |
| MAPLES-DR 标注（figshare DOI 10.6084/m9.figshare.24328660） | **未下载** | 同上搜索无 `*MAPLES*` 目录 |
| MESSIDOR 原图（MAPLES-DR 的眼底图不随标注发布） | **未找到** | 同上搜索无 `*MESSIDOR*` 目录 |
| FGADR | 未申请、无资产 | 主方案明确需签研究协议，本轮不代签 |
| DDR_split_613 | 完整可用 | 第 6 节 |
| IDRiD 官方目录结构约定 | 已核实（旧转换器） | 第 7 节 |
| MAPLES-DR 发布物内部布局/掩膜编码 | **未能离线核实**（沙箱网络受限，仅 GitHub README 可达） | README 证实：标注经 figshare 分发、眼底图需向 MESSIDOR (adcis.net) 单独申请、库提供 `load_train_set()/load_test_set()` 与 `'MAPLES-DR/train/'`、`'MAPLES-DR/test/'` 导出路径；archive 内部文件命名需下载后用 `build_external_manifest.py --dataset maples-dr` 的 fail-closed 探测确认 |

## 9. 报告与代码冲突记录（以代码/原始结果为准）

1. **base 配置的 val 指向 test**：`configs/_base_/datasets/hr_ddr_2048.py:66-81` 中 val/test 都指向 `images/test`；但 `exp/baseline_split613.py:42-45` 覆盖为 `images/val`。实际训练验证用 val 76 张。若直接继承 base 而不经 exp 配置，会把 test 当验证集——新工作区配置必须显式写 val 路径，不得依赖继承。
2. **ignore_index 形同虚设**：`lesion_metric.py:173` 接受参数但未使用（代码为准）。因 DDR 融合标签值域实测无 255，现有结果不受影响；但外部数据若直接套用该评估，valid=0 类别会被当负样本。
3. **训练端 ignore 行为**：`binary_loss.py:11-37` 把 255 与 0 同等对待（均为四类负样本目标）。旧报告若声称"255 被忽略"，与代码不符。新训练入口必须自带 per-class valid 掩蔽，不能复用该损失语义。
4. **主方案"DDR 757 张"**：与实测 split 总数 757 一致，无冲突（复核通过）。
5. **冠军 44.2115 的出处**：`D:\HRDecoder\work_dirs` 现存日志（20260721_210304.log）最后 val mIoU≈0.308，不是冠军训练；44.2115 来自后续实验（`D:\dataset\experiment\HRDecoder_FusionAware_20260917\delegation\protocolfix_prompt.txt:8` 称 "champion44.2115 verifiedhistorical"）。本审计按"历史已核实结果"记录，不重新验证。
6. **`lesion_dataset.py:15` 注释掉的 7 类定义**（含 IRMA/NV）：代码实际启用 5 类（bg+4 病灶），7 类仅是历史残留。外部数据中出现 IRMA/VB/NV 等其他病变时，按主方案"不强行塞入四类"，在新契约中记录为未映射类，不参与训练/评估。

## 10. 本机环境事实（影响工具实现）

- Windows 10 x64，Python 3.13.11 (Anaconda)：numpy 2.4.1、Pillow 12.1.0、OpenCV 4.13.0、PyYAML 可用；**scikit-image 缺失**。
- 因此感知哈希在本工作区工具内用 numpy+PIL 自实现（aHash/dHash/pHash-DCT），不引入新全局依赖；`jsonschema` 若存在则优先用于 schema 校验，否则用内置校验器（tools 中已做降级记录）。
- D:\HRDecoder 的训练栈为 Linux + mmcv-full 1.7.2（whl 在仓库根目录）+ 定制 mmseg v0.x；本机不安装、不运行训练。
