# A/U/S 教师概率缓存核查

**结论先行。** 现有缓存的索引、通道声明、原图坐标、float32 数值范围与无损保存链条没有发现异常。单图 `007-6383-400` 上，GT 阳性的 EX、HE 掩膜落在对应概率通道且均有高响应，未见 EX/HE 通道互换；但该图没有任何 SE 阳性像素，因此这组探针不能判断教师是否保留 SE 阳性信号，也不能解释 20k 学生的 SE 零检出。尤其不能用全图均值代替 GT 阳性区域均值：探针图 EX 只占 2.21% 像素，A/U/S 的 EX 全图均值约 0.021–0.028，而 EX GT 阳性区均值为 0.749–0.891。

## 核查边界

只读取本地缓存导出脚本、缓存审计 JSON、真实缓存探针 NPZ、对应训练图像及四类 GT TIFF。用 CPU/NumPy 读取缓存概率与掩膜并做统计；没有加载模型权重、调用模型推理、训练或访问远端，也没有改动历史文件。

## 导出与数据契约

- `cache_common.py:15-17` 将缓存类别顺序声明为 `EX, HE, SE, MA`；`cache_common.py:66-80` 将成员、训练 split、通道顺序、float32、原图坐标以及 teacher in-sample / `OOF=false` 写入身份元数据。
- A 是历史 D/E/B/E4 融合，成员权重为 3:2:1:1，包含水平翻转 TTA 与 H4 MA 修正，导出用 FP32；见 `cache_execution\export_A.py:1,30-34,47-61`。导出器按锁定样本 SID 查找图像，并核对原图尺寸。
- U0 使用原尺寸整图、ImageNet 均值/方差归一化和 32 倍数补齐，模型 logits 截回原图后用 FP32 sigmoid；见 `export_U.py:21-24,35-44`。
- S20 使用 1024 窗口、768 中心区和 128 halo 在原图坐标拼接，前向为 BF16，sigmoid 与输出为 FP32；见 `export_S.py:20-24,38-48`。加载的是锁定的 20k EMA，`strict=True`，并核对保存步数；见 `export_S.py:26-34`。
- `cache_common.py:89-109` 写出前强制 `[4,H,W]`、float32、finite 且落在 `[0,1]`，随后对 NPZ 往返数组做逐位 SHA 校验；`cache_common.py:121-124,142-168` 将缓存绑定到 manifest 的 SID/尺寸与图像、标签 SHA，并在完成时核对预期样本集合。
- `CACHE_READY.json` 的 `status/stage` 为 PASS / `TRAIN_CACHE_READY`，每个成员 454 张，通道为 EX/HE/SE/MA、坐标为原图，所有文件与张量 SHA 及无损往返通过；`OOF=false`、预测为 in-sample。具体见 `cache_execution\server_receipts\CACHE_READY.json:2-21`。S20 另有单张 val canary 与旧缓存逐位一致（`bitwise_equal=true`）；见同文件 `:31-38`。它只覆盖该 canary，不代表所有模型输出的语义质量。

上述合同及 SHA 检查说明保存阶段没有发现重排、缩放坐标或 float32 压缩损失；它们本身不证明每个语义通道都正确。下方单样本掩膜对照可直接检验 EX/HE 通道。

## 探针样本与概率统计

本地探针身份为 `007-6383-400`、尺寸 `1152×1536`，其 `image_sha256` 与 `D:\dataset\lesion_segmentation\train\image\007-6383-400.jpg` 完全相同。A/U/S 的三个缓存索引均登记同一 SID、尺寸 `[4,1152,1536]`、float32、相同图像及标签身份哈希，且均标记 `lossless_roundtrip_verified=true`；见 `training_execution\real_cache_probe\probe_identity.json` 及 `training_execution\real_cache_probe\cache\train\{A,U,S}\index.json`（A 条目 `index.json:5953-5968`，U `:5971-5987`，S `:5949-5964`）。

对应 GT 文件位于 `D:\dataset\lesion_segmentation\train\label\{EX,HE,SE,MA}\007-6383-400.tif`。四张掩膜均为 `1152×1536`，GT 阳性像素数依次为 EX 39,172、HE 27,881、SE 0、MA 0；这与 `train_cache_audit.json` 中该 SID 的 `gt_positive` 逐类计数一致（`:33500-33510`, `:33528-33531`, `:33549-33552`, `:33570-33573`）。探针 `label_sha256` 是数据清单中的标签身份字段；本核查没有把它宣称为四张 TIFF 任一单文件 SHA 的字节匹配。对应关系由 SID、图像 SHA、空间尺寸以及逐类阳性像素数与缓存审计一致共同支持。因此该探针可检验 EX、HE 阳性区域与全图 SE 阴性响应，但不含 SE 阳性区域。

缓存的 EX/HE/SE/MA 通道均为有限 float32，取值均在 `[0,1]`。下表对 A/U/S 逐个数组计算全图均值，以及对应 GT=1 掩膜内的均值。GT 阳性区的像素不是全图均值的替代统计；两者在稀疏病灶类别上差异很大。

| GT 类别 | GT 阳性像素 | A：全图均值 / GT+均值 | U：全图均值 / GT+均值 | S：全图均值 / GT+均值 |
|---|---:|---:|---:|---:|
| EX | 39,172 | 0.02095 / **0.74880** | 0.02794 / **0.89108** | 0.02263 / **0.83830** |
| HE | 27,881 | 0.02335 / **0.69130** | 0.01957 / **0.87820** | 0.01124 / **0.59908** |
| SE | 0 | 0.002446 / 不适用 | 0.0000535 / 不适用 | 0.000000000204 / 不适用 |
| MA | 0 | 0.000109 / 不适用 | 0.0000435 / 不适用 | 0.00000000107 / 不适用 |

对 EX 掩膜使用 `p>0.5` 仅作可读的阈值示例：A 命中 31,062/39,172（79.3%），U 35,014（89.4%），S 33,006（84.3%）。HE 对应通道均值也明显高于其他通道：A 0.691、U 0.878、S 0.599；这为本图上的 EX/HE 通道对应关系提供了概率侧证，没有看到这两类互换。该示例阈值不是校准结论或全数据集召回率。

这张图的 SE GT 全为负像素；A/U/S 的 SE 通道全图均值分别约 0.002446、0.0000535、2.0e-10，`p>0.5` 的像素分别为 51、44、0。它说明教师在这张阴性图上给 SE 的概率低，不能说明 SE 阳性区是否被抑制。特别是 S 的近零值在此例与全阴性掩膜相容，不能单独据此诊断训练缓存把 SE 信号弄丢。

## 与 20k EX/SE 零检出的关系

训练集并非没有 EX/SE 标签：`CACHE_READY.json` 的类别统计记录 EX 有 5,740,603 个阳性像素、分布在 303 张图；SE 有 1,327,424 个阳性像素、分布在 143 张图（见 `:40-147`）。在本探针上 EX 教师目标仍有强 GT+ 区域响应，因而“缓存整体没写入 EX 正信号”不符合该样本。SE 则缺少 GT+ 对照；现有全训练 audit 记录 GT 阳性数量及 U/S 差分能量/方差等全域统计，没有给 A/U/S 在 GT+ SE 像素上的概率均值、分位数或阈值命中率。其解释字段也明确指出差异能量/秩只能证明目标有变化，不能证明知识有用、能泛化或带来提升（`train_cache_audit.json:43774`）。

因此目前没有序列化损失或 EX/HE 通道交换的证据，但也没有足够证据把 SE 零检出归因于某一教师或排除教师对 SE 正区响应不足。一个含 EX/HE 阳性而 SE 阴性的样本，只能作为单样本核查，不能推及 454 张训练图或 20k 学生行为。

## 建议补充的诊断

1. 在全部 454 张训练 GT 上，按教师 A/U/S、类别和 `GT=1/GT=0` 分开汇总：正像素总数与阳性图数、正区概率均值/中位数/P90/P99、`p>0.1/0.25/0.5` 的正区命中率；负区报告均值/P99/P99.9 与超过同阈值的像素数。另列逐图宏平均，避免巨图和稀疏病灶被像素池化掩盖。对拟使用的加权软目标也算同一组统计。
2. 对 20k 学生按类别记录每个 batch 中包含该类 GT 阳性的图数/像素数、学生在 GT+ 区的概率分位数及阈值曲线，并把监督 BCE/Dice 与 KD 损失分别拆成 GT+/GT− 两部分；同时记录训练采样到 SE 阳性图的频率。这样可区分缓存信号、正样本采样、损失压制与最终阈值问题。

## 证据文件

- 导出与通道契约：`D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\cache_execution\export_A.py:1,30-34,47-61`；`export_U.py:21-24,35-44`；`export_S.py:20-34,38-48`；`cache_common.py:15-17,66-80,89-109,121-124,142-168`。
- 全量缓存完成记录：`D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\cache_execution\server_receipts\CACHE_READY.json`，JSON 键 `status`, `stage`, `images_per_member`, `channel_order`, `coordinates`, `teacher_predictions_are_in_sample`, `OOF`, `all_file_and_tensor_SHA_verified`, `all_lossless_roundtrips_verified`, `S20_locked_val_canary_parity`（分别见行 2-21、31-38）。
- GT 数量与统计边界：`D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\cache_execution\server_receipts\reports\train_cache_audit.json:40-183`（全量类别统计）、`:33500-33584`（探针 SID 逐类 GT 像素数和 U/S 全图和）、`:43774`（解释边界）。
- 实际概率探针：`D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\training_execution\real_cache_probe\cache\train\A\007-6383-400.npz`、对应 U/S NPZ；三个 `index.json` 条目分别在 A `:5953-5968`、U `:5971-5987`、S `:5949-5964`。身份记录：`D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\training_execution\real_cache_probe\probe_identity.json`。
- GT 对照：`D:\dataset\lesion_segmentation\train\label\EX\007-6383-400.tif`，以及 `HE`、`SE`、`MA` 子目录中的同名 TIFF；配对图像为 `D:\dataset\lesion_segmentation\train\image\007-6383-400.jpg`。
