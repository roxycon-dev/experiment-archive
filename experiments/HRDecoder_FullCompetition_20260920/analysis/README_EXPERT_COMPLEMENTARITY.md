# 历史类别专家 — val 逐像素互补性分析（HRDecoder FullCompetition 2026-09-20）

本目录新增一组**纯本地、只读 val**的脚本：把若干"完整预测管线"在同一 DDR
454/76/227 划分的 **val 76 张**上导成**严格 `p>0.5` 的二值 packbits 掩膜**，
再做**逐像素**互补性分析。任何一步都**不联网、不登录、不训练、不跑 GPU**
（导出脚本本身需要 CUDA 运行模型，由用户在 3090 上手动执行；本仓库只写代码与
CPU 单元测试）。

> 结论口径：本工具**只做 val**。split=test、路径含 `test` 分量、图像数 ≠ 76
> 一律 fail closed。选择/融合/阈值**全部只在 val**，绝不触碰 test。

---

## 1. 文件清单

| 文件 | 作用 |
|---|---|
| `analysis/expert_mask_contract.py` | 共享契约：packbits 编解码、严格二值化、计数、OR/AND、原子写、val-only 闸门、GT 共享与校验、契约读/写/校验。被其余脚本 import。 |
| `analysis/export_hrplus_expert_masks.py` | 导出 HRPlus 家族 6 条管线：`champion / BASE_0 / D_blend / C2000 / MIX_C2000 / S4000`。复用历史 `runtime.ChampionEnsemble` 与 `eval_candidates`。 |
| `analysis/export_fullmodel_masks.py` | 导出本工程整模 `U0 / H0`。复用 `trainer` 的整图/滑窗评测路径，同一契约。 |
| `analysis/analyze_expert_complementarity.py` | 校验样本 ID/形状/GT 哈希后，逐类算共同 TP、champ-only TP、candidate-only TP/FP、OR、AND、候选纯替换 IoU、每图差值、GT 连通域面积分桶召回；输出 JSON/CSV/Markdown。 |
| `analysis/expert_eval_settings.example.json` | 所有服务器路径与候选定义的**模板**（复制为 `expert_eval_settings.json` 再填）。**不含任何账号/密码/密钥**。 |
| `tests/test_packed_mask_contract.py` | CPU 单元测试：pack/unpack、变尺寸、严格阈值、计数与 OR/AND、契约完整性、原子写、端到端合成分析。 |

---

## 2. 命令

全部命令在仓库根目录执行，`python` 为本地解释器（无需 GPU 的部分：测试与
`py_compile`；导出环节需要 3090 + CUDA）。

```bash
# 0) 语法检查（无 GPU）
python -m py_compile analysis/expert_mask_contract.py \
    analysis/export_hrplus_expert_masks.py analysis/export_fullmodel_masks.py \
    analysis/analyze_expert_complementarity.py tests/test_packed_mask_contract.py

# 1) CPU 单元测试（含端到端合成 fixture，自动清理临时目录）
python tests/test_packed_mask_contract.py

# 2) 复制并按服务器真实路径填写 settings
cp analysis/expert_eval_settings.example.json analysis/expert_eval_settings.json
#    填满所有 REQUIRED_* 与 expected_sha256 / sha256

# 3) 预处理：先 dry-run 校验 settings 与资产（不加载模型；缺资产即 BLOCKED）
#    （HRPlus）—— 首次小样本 smoke：--limit 5 只跑前 5 张且**不写 contract**
python analysis/export_hrplus_expert_masks.py \
    --settings analysis/expert_eval_settings.json \
    --pipelines champion --limit 5

# 4) 正式导出 HRPlus 6 条管线（写 <out_root>/hrplus/<name>/ + contract.json）
python analysis/export_hrplus_expert_masks.py \
    --settings analysis/expert_eval_settings.json \
    --pipelines champion BASE_0 D_blend C2000 MIX_C2000 S4000

# 5) 正式导出整模 U0/H0（写 <out_root>/fullmodel/<name>/）
python analysis/export_fullmodel_masks.py \
    --settings analysis/expert_eval_settings.json --models U0 H0

# 6) 逐像素互补性分析（读 <out_root>/_shared/gt + 各模型 contract.json）
python analysis/analyze_expert_complementarity.py \
    --settings analysis/expert_eval_settings.json \
    --out-root <out_root> \
    --models champion BASE_0 D_blend C2000 MIX_C2000 S4000 U0 H0 \
    --champion champion
```

**退出码约定**（两个导出脚本与分析脚本一致，供 CI/人判断）：

| 退出码 | 含义 |
|---|---|
| `0` | 成功 |
| `2` | `CONTRACT FAILURE`：契约违规（哈希不符、掩膜篡改、GT 不一致、路径含 test、图像数 ≠ 76 等） |
| `3` | `BLOCKED`：前置条件缺失（资产不存在、settings 未填、服务器源码接口不符） |
| `1` | 其它未预期异常（打印 traceback） |

**两个导出脚本必须共享同一个 `out_root`**：共享 GT 写在
`<out_root>/_shared/gt`，HRPlus 写在 `<out_root>/hrplus`，整模写在
`<out_root>/fullmodel`。

---

## 3. 契约（安全格式）

每个模型目录 `<out_root>/<family>/<name>/` 只包含：

- `mask_<stem>.npz`：`packed`（uint8 `[4, ceil(N/8)]`，bitorder='big'）+
  `n_pixels` + `orig_hw`。**只存 1 bit/像素，不存任何概率**；`n_pixels` 记录
  有效位数，变尺寸（val 有 17 种 H×W）可无损还原。
- `contract.json`：样本 ID 列表、逐图 `orig_hw/n_pixels/file_sha256/semantic_sha256`、
  逐图逐类 `TP/FP/FN`、全局计数、`metrics`（含 canonical `mIoU_lesion4`）、
  模型/配置/权重 **SHA256**、评估契约（family / eval_loader / inference / TTA /
  阈值口径 `p > 0.5`）、`blockers`。
- 共享 GT 只写一次到 `<out_root>/_shared/gt/`（4 类 packbits + `ignore` packbits）。

硬性保护（全部与 CPU 测试对应）：

1. **只 val**：`split`、路径分量、图像数三处同时封死（`assert_val_only`）。
2. **float32 + 严格 `>0.5`**：概率在 float32 完成 sigmoid/融合/H4 后才二值化；
   `p == 0.5` 判阴性；**禁止 fp16 概率**（`check_float32_like` 在 `.float()`
   之前就拦截 fp16/bfloat16）。
3. **不保存概率**：契约中若出现概率字段（`prob/float16/logits/...`）直接拒绝
   （`assert_no_probability_payload`）。
4. **原子写**：临时文件 + `os.replace` 原子 rename；半截产物分析端拒绝。
5. **逐图位图 ↔ 计数一致**：掩膜位内容必须与契约里的逐图 TP/FP/FN 完全一致，
   否则 fail closed。
6. **不把各模型独立聚合数字拼成互补**：分析端读掩膜**逐像素**重算，并同时用
   OR/AND 分解恒等式（构造性校验）交叉验证。

### 预期磁盘

val 76 图合计 401,111,888 像素（17 种尺寸）。每模型 4 类 packbits ≈
**191 MiB**；8 模型 ≈ **1.5 GiB**，加共享 GT（另 ≈191 MiB）与 `ignore`，
总计约 **1.7 GiB**。掩膜为近似随机位图，压缩收益很小，按位图大小估即可。
**不产生任何 FP32 概率缓存**（因而不存在数十 GB 的概率落盘）。

---

## 4. 四个关键点的代码级证据

> 本地参照的"历史交接版本"位于 `D:/dataset/experiment/HRDecoder_FusionAware_20260917/`
> （`runtime.py` / `run_protocol.py` / `execution_20260919/eval_candidates.py`）。
> 服务器上 `fusionaware_src_dir` / `fusionaware_exec_dir` 必须指向**同一版本**，
> 否则导出脚本在 preflight 直接 `BLOCKED`（见 §5 阻塞项）。

### 4.1 D_blend 公式 = 0.5·P_student(D) + 0.5·P_reference(B)  ✅ 与历史一致

- 历史公式：`HRDecoder_审查轮_20260916/tools/diag_train_inference_contract.py:8`
  `prob_f = 0.5*prob_r + 0.5*prob_s`，`:135` `pf = 0.5 * pr + 0.5 * ps`
  （注释即"部署实际使用的融合"，无重归一化）。
- 语义证据：`analysis/historical_class_shortlist.md:63` 记 HE 最优
  `D_blend (D student + B reference 0.5/0.5) = 0.44727`；`:161` 明确
  "`全部结果.csv` 的 VAL D 单模型 41.6148 **实为 0.5 blend** 而非 student-only"；
  `:184` 记 `D_blend` val mIoU `0.41614841513965567`。
- 本工具实现：`analysis/export_hrplus_expert_masks.py:53` `BLEND_W = 0.5`，
  `:369` `"D_blend": BLEND_W * ps + (1.0 - BLEND_W) * pr`。其中
  `ps`/`pr` 来自 `runtime.py:735` `pre_head_probs()`（`ps = D 的 _student_prob`，
  `pr = D 的 _reference_prob`，D 的 reference 即 settings 的
  `reference_checkpoint` = B）。
- **结论**：公式逐字一致，无重归一化，**非 BLOCKED**。

### 4.2 MIX_C2000 公式 = 0.75·champion + 0.25·C2000  ✅ 与历史 eval_candidates 一致

- 历史实现：`HRDecoder_FusionAware_20260917/execution_20260919/eval_candidates.py:67-69`
  `MIX_CHAMPION = 0.75` / `MIX_CANDIDATE = 0.25`，并 `assert` 二者之和 = 1；
  `:412` `pm = MIX_CHAMPION * pc + MIX_CANDIDATE * pn`。
- 语义证据：`historical_class_shortlist.md:49` "MIX_C2000 (0.75 champ + 0.25 C@2000)"
  val EX `0.46144246242771647`；`:132` 记 MIX_C2000 可由 champion + C_step_2000 复现；
  `historical_class_evidence.json` records[2] `arch = "fixed 0.75/0.25 prob fusion, no weight scan"`。
- 本工具实现：`export_hrplus_expert_masks.py:50-52` 同样定义 `MIX_CHAMPION=0.75`、
  `MIX_CANDIDATE=0.25` + `assert` 和为 1；`:373-374`
  `MIX_CHAMPION * champ + MIX_CANDIDATE * cand_prob["C2000"]`。
- **结论**：冻结常数与表达式逐字一致，**不可扫参**，**非 BLOCKED**。

### 4.3 H4 只修改 MA，且输入为 7 通道 f7  ✅ 与历史一致

- 历史实现：`runtime.py:733` `self.heads = {'MA': load_flat_head(...)}`——
  champion **只有 MA 一个头**；`:775-791` `predict()` 对 `c in range(4)`，
  **仅当** `CLASSES[c] in self.heads`（即只有 MA）才走
  `corrected_class_prob`，其余直接取 `pf[c:c+1]`。
- f7 构造：`runtime.py:755-773` `corrected_class_prob` 中
  `f7 = cat([img_f, _logit(pf_c), _logit(ps_c), _logit(pr_c), ps_c - pr_c])` →
  `img_f` 为 3 通道 + 4 个标量通道 = **恰好 7 通道**；输出
  `sigmoid(logit(pf_c) + head(f7))`。
- 语义证据：`historical_class_shortlist.md:111` 记小头家族
  "EX/HE/SE 恒等于 BASE_0，**仅 MA 不同**"，与本实现一致。
- 本工具实现：`export_hrplus_expert_masks.py:188` `champion_shared_prob()`
  **逐行复现** `predict()` 的通道循环（复用同一 `ensemble.heads` 与
  `ensemble.corrected_class_prob`，不重写 f7）。并有两个**逐位自检**：
  1. `:378-384` 首图 `champion_shared_prob(...)` 必须与
     `champion.predict(...)` **逐位相等**，否则拒绝出图；
  2. `:386-390` `champion[:3]`（EX/HE/SE）必须与 `BASE_0`（`pf`）**逐位相等**，
     否则报"H4 只应改 MA"并拒绝。
- **结论**：H4 仅改 MA、输入固定 7 通道，**非 BLOCKED**。

### 4.4 val-only 与严格 float32 `> 0.5`  ✅

- val-only：`expert_mask_contract.py:159` `assert_val_only` 同时检查
  `split`、`paths` 分量（`FORBIDDEN_PATH_COMPONENTS = ("test",)`，
  `:56`）、`n_images != EXPECTED_VAL_IMAGES(=76)`（`:55`）。两个导出脚本与
  分析脚本均首先调用；测试 `test_val_only_guard` 覆盖 `test/TEST/train/''`
  与 `/a/test/b` 等路径。
- 严格阈值：`expert_mask_contract.py:45` `STRICT_THRESHOLD = 0.5`、
  `:231` `return prob > np.float32(STRICT_THRESHOLD)`（`>` 而非 `>=`，
  `p == 0.5` 判阴性）。
- float32：`check_float32_like`（`:203`）在 `.float()` 前拦截
  fp16/bfloat16；导出端 `to_fp32_numpy`（`export_hrplus_expert_masks.py:173`）
  同样先 `check_float32_like` 再转 float32；`binarize_strict` 还要求数值有限且
  ∈[0,1]。测试 `test_strict_threshold_and_precision` 覆盖 `0.5 → False`、
  fp16 被拒。
- **结论**：**非 BLOCKED**。

---

## 5. 阻塞（BLOCKED）清单 — 需要服务器资产才能解

以下项**无法**在本地确认，**不猜**；脚本遇到缺失会以 `BLOCKED(exit 3)` 退出：

1. **H0 权重缺失（BLOCKED）**：本地仅有
   `runs/H0_stopped6000_metadata.tar.gz`（元数据，含 `eval_u6000.json`，**无 .pth**）。
   `historical_class_shortlist.md:131,142` 亦记 "H0 因无本地权重被排除"。
   → `export_fullmodel_masks.py` 在 `models[H0].checkpoint` 缺失时 BLOCKED；
   需服务器提供 H0 的 `ckpt_best.pth` 及其 `expected_sha256`。
   同时 H0（hrdecoder family）需要 `reference_root` 指向**旧 HRDecoder 仓库只读副本**；
   若服务器未提供该目录，构建即 BLOCKED。
2. **C2000 / S4000 node checkpoint 契约（需服务器路径）**：必须来自 `phase_cs.py`
   产出的 node checkpoint（`payload` 内含 `state_dict/arm/steps`，并做逐张量
   bitwise 回验，见 `eval_candidates.py:100` `load_node_checkpoint`）。settings 中
   `cs_candidates[*].{arm,steps}` 必须与冻结定义
   （`C2000=(C,2000)`、`S4000=(S,4000)`）一致，否则检查期即拒绝。
3. **服务器端源码版本一致性（潜在 BLOCKED）**：本工具按
   `HRDecoder_FusionAware_20260917` 这一本地交接版本的**真实接口**编写
   （`ChampionEnsemble.pre_head_probs/corrected_class_prob/predict`、
   `build_student`、`student_prob` 等）。若服务器
   `fusionaware_src_dir/fusionaware_exec_dir` 指向其它版本、缺少上述符号，
   导出脚本 preflight 会打印缺失符号并 `BLOCKED(exit 3)`——**不降级、不猜测**。
   此项需在服务器上以 §2 步骤 3 的 `--limit` smoke 验证。
4. **U0 权重与历史 mIoU**：settings 中 U0 的
   `expected_miou = 0.3844641257662971` 取自
   `historical_class_evidence.json:233,388` 与
   `docs/champion_asset_inventory_workbuddy.md:147`（U0 @12000 canonical）。
   需服务器提供 `ckpt_best.pth` 及其 sha256；若实际 mIoU 与之差 >1e-6，
   导出拒绝（防止拿错权重）。

> 说明：D_blend / MIX_C2000 / H4 / val-only / 严格阈值五项**均已在既有代码中
> 逐行确认**（§4），不属于阻塞项。

---

## 6. 失败保护小结

- 导出端与分析端**共享同一 GT**：分析前先校验 `gt_semantic_fingerprint` 与
  `gt_contract_sha256`，样本 ID 顺序、逐图 `orig_hw/n_pixels` 必须完全一致。
- 逐图掩膜文件哈希（`file_sha256`）在分析时重新计算；**篡改一个字节即
  `exit 2`**（测试 `test_end_to_end_complementarity_synthetic` 末段覆盖）。
- `--limit` 的 smoke 运行**不写 contract**，因此该目录会被分析端视为不完整而拒绝，
  不会污染正式结果。
- 分析只在 val 上进行；`--allow-missing NAME` 被显式使用时会把跳过项写入输出，
  绝不静默。
