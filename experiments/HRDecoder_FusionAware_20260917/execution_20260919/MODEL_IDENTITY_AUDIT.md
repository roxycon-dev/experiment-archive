# MODEL_IDENTITY_AUDIT — 模型身份审计（source_checkpoint / 训练输出 / 推理融合 / 评估口径）

- 日期：2026-09-19
- 性质：静态代码审计。仅阅读下列文件，**未运行任何代码、未修改任何代码**；本报告是本次唯一新增产出。
- 审计对象（均未修改）：
  - `D:\dataset\experiment\HRDecoder_FusionAware_20260917\fork_hrplus_patch.py`
  - `D:\dataset\experiment\HRDecoder_FusionAware_20260917\fork_cfg_hrplus_patch_dice.py`
  - `D:\dataset\experiment\HRDecoder_FusionAware_20260917\runtime.py`
- 对照文档：`outputs\HRDecoder_训练优化方案_20260919_深度修订.md`（2.3 节代码事实、4.1 节、第 7 节融合顺序）
- 行号引用约定：`patch:Lxx` = fork_hrplus_patch.py，`cfg:Lxx` = fork_cfg_hrplus_patch_dice.py，`rt:Lxx` = runtime.py。

## 0. 结论速览

| 问题 | 结论 | 关键证据 |
|---|---|---|
| source_checkpoint 何时覆盖 pretrained | pretrained **从未生效**（配置层与运行时两层置 None）；source_checkpoint 是**构造函数内**唯一且 strict 的权重加载事件（`patch:L78`）；构造后 `init_weights()` 被覆写为 no-op（`patch:L91-94`），任何后续初始化调用无法再引入 pretrained。冠军集成路径下构造后另有一次 `strict_initialize` 二次覆盖学生权重（`rt:L697-713`） | cfg:L28-29,47-48；rt:L62-64,95-98；patch:L64-88,91-94 |
| 训练输出是否 student_only | **是**。`forward_train` 只前向学生 `encode_decode` 的 crop logits + `decode_head.losses`（可选 lovasz）；reference 冻结、不进损失；首步证据门验证 `reference.*` 参数与 BN buffer 均不变 | patch:L103-113,84-100；rt:L145-148,151-193 |
| 推理是否内含 reference 融合 | **是**。`HRPlusRefinement.inference` 固定输出 `0.5×reference + 0.5×student`（`patch:L150-151`）；冠军 D/E 成员概率同样内含该融合；若外部再 `mix_probs` 0.5/0.5，新学生实际权重仅 0.25、旧 reference 重复进入 | patch:L116-151；rt:L649,744-752,793-797,800-801 |
| 四通道阈值与 mIoU 口径 | 4 通道独立 sigmoid 概率，**严格 `p > 0.5`**；**按全数据集累计 TP/FP/FN** 后逐类 `IoU = TP/(TP+FP+FN)`，**四类简单算术平均**，无背景类；ignore(255) 像素只从 FP 剔除；union=0 直接报错不跳类 | rt:L487,490-502,505-515,536-540,562-582 |

---

## 1. source_checkpoint 何时覆盖 pretrained

### 1.1 配置层：pretrained 双重置空，不存在可加载的 pretrained

- cfg:L28 `pretrained=None`；cfg:L29 `backbone=dict(pretrained=None)` —— 通过 mmcfg 合并覆盖基类配置（cfg:L1 `_base_ = '../exp/hr100_a.py'`）中的同名字段。
- cfg:L47-48 `load_from = None`、`resume_from = None` —— runner 层无 checkpoint 装载路径。
- rt:L62-64 `load_config()` 在构建前再次强制 `model['pretrained'] = None`（条件 `'pretrained' in model`）。
- rt:L57-59（docstring）：`load_config` **不**触碰 `source_checkpoint`；由 driver 在构造前把 `cfg.model['source_checkpoint']` 改写为绝对 B 路径（driver 脚本不在本次审计清单内，此句为 runtime 自述）。
- cfg:L33 `source_checkpoint='weights/b_weights.pth'`；cfg:L5 经 `custom_imports` 注册 `hrplus_patch`。

### 1.2 构造时：唯一生效的权重事件（严格时序）

真实调用链：driver → `build_model`（rt:L95-98）→ `build_segmentor(cfg.model)`（rt:L98）→ `HRPlusRefinement.__init__`（patch:L64）。构造函数内时序：

1. patch:L65 `assert source_checkpoint` 必填（不存在无 source_checkpoint 的构造路径）。
2. patch:L66-68 强制 `semantic_refinement=False`、`hard_weight=0`、`freeze_bn=True`。
3. patch:L71 `ref_kwargs = copy.deepcopy(kwargs)` —— 在 super() 变异 kwargs **之前**快照，供 reference 独立构建。
4. patch:L72 `super(HRPlusRefinement, self).__init__(**kwargs)` —— 父类按其自身逻辑构建学生模块；此时 pretrained 已为 None，无外部权重加载（模块处于默认初始化状态）。
5. patch:L73 `_assert_config(self)`：4 类 sigmoid、hr_scale=(1024,1024)（patch:L36-41）。
6. **patch:L78 `_strict_load_state(self, source_checkpoint)` —— source_checkpoint 在此处（构造函数内、任何训练更新之前）一次性覆盖全部学生权重**。`_strict_load_state`（patch:L24-33）以 `strict=True` 加载（patch:L33）：模型缺失任何 key 或 checkpoint 多出任何 key 均报错（patch:L27-29 注明 checkpoint 不含 `ma_refiner.*`）。strict 语义保证加载后**不存在残留的随机初始化张量**；state_dict 同时包含 buffer，故 BN running stats 一并来自 B checkpoint。
7. patch:L79 `_freeze_all_bn(self)`（patch:L43-49 全部 BN 置 eval）。
8. patch:L82-83 reference 由 pre-super 快照 kwargs 独立构建（`HR100Refinement(**ref_kwargs)`，是**父类实例**），并以同一 source_checkpoint strict 加载；patch:L84-88 冻结并置 eval。

### 1.3 构造后：再初始化被焊死

- patch:L91-94 `init_weights()` 被覆写为 no-op（docstring 明言"never re-initialize"）—— mmseg 训练脚本的 `model.init_weights()` 等任何后续初始化调用都**不可能**重新随机化或引入 pretrained。
- 配合 1.1 的 `load_from/resume_from=None`，训练全程没有第二条权重注入路径。

**结论（Q1）**：不是"先加载 pretrained、再被 source_checkpoint 覆盖"；而是 pretrained 在配置层（cfg:L28-29）与运行时（rt:L63-64）被双重置空后**从未加载**，source_checkpoint 在构造函数内（patch:L78，即 `build_segmentor` 构建模型的时刻，早于首个训练 step）就是唯一且最终的初始化事件；no-op `init_weights`（patch:L91-94）保证该状态此后不可被覆盖。

### 1.4 冠军集成路径的第二覆盖点（构造之后）

`ChampionEnsemble`（rt:L660-733）按成员加载：

- rt:L687-691：成员 config 为 `HRPlusRefinement` 类型（或含 `source_checkpoint` 字段）时，把 `c.model['source_checkpoint']` 注入为 settings 的 B 路径（`ref_abs`，经 `resolve_repo_path` 解析，rt:L671-672、L101-107），或成员 spec 显式指定的路径（如 E4 的 base read source）。
- rt:L697-708：`mode == 'student_only'`（冠军 D/E 等 HRPlus 家族成员）→ `strict_initialize(m, ckpt_abs, ref_abs)`：**构造器内刚加载的 B 权重随即被成员 checkpoint 覆盖（学生部分），reference 部分取 B 并与成员 checkpoint 内嵌 reference 交叉校验**（语义依据 rt:L699-708 docstring，复现 h5 历史状态）。
- rt:L709-713：`mode == 'full'`（B、E4 等非 HRPlus 成员）→ `strict_initialize(m, ckpt_abs, None)`，整模型严格完整加载、不注入 reference。
- ⚠️ `strict_initialize` 的实现位于 `fusion_core.py`（rt:L28-31 导入），**不在本次给定的三个审计文件内，未逐行核验**；上述语义以 rt:L699-708 的 docstring 为证。

---

## 2. 训练输出是否 student_only

**是。** 证据链（forward_train，patch:L103-113）：

- patch:L105 `logits = self.encode_decode(img, img_metas)` —— 只有**学生**的 crop logits；`self.reference` 不出现在前向计算中。
- patch:L106-108 ignore(255) 像素的 logits 置 -1e4（sigmoid 后 ≈0），注释说明旧 BinaryLoss 把 ignore 映射为 0 但不遮 prediction，此处理使被忽略像素从 squared-Dice 项中剔除。
- patch:L109 `losses = self.decode_head.losses(dice_logits, gt_semantic_seg)`。
- patch:L110-112 `lovasz_weight > 0` 时附加 `multiclass_lovasz_hinge`（作用在**未遮蔽**的原始 logits 上）；本配置 `lovasz_weight=0.0`（cfg:L34）不启用。
- 类 docstring（patch:L56-58）明确：无 MA hard loss、无 HR crop auxiliary、无 fuse losses。

reference 从不参与训练：

- patch:L84-85 reference 全部参数 `requires_grad_(False)`；patch:L86 + L43-49 reference 的 BN 冻结；patch:L96-100 `train()` 强制 `self.reference.eval()`。
- 训练可更新参数白名单：rt:L145-148 `_authorized_param` —— 仅 `decode_head.*` 与 `backbone.stage4.*`，`reference.*` 一律视为冻结。
- 首 optimizer step 证据门 rt:L151-193 `first_update_evidence`：验证冻结参数（含 reference）不变、BN buffer（running_mean/var/num_batches_tracked）不变、decode_head 与 stage4 均有非零有限梯度、至少一个允许参数实际变化（驱动侧 `all(ev.values())`，rt:L143 注释）。

口径补充："训练输出"的两种读法均成立/需区分：

- **损失/梯度意义**：student_only（上述证据）。
- **保存的 checkpoint 文件**：包含冻结 reference 的完整副本（reference 是注册子模块，patch:L88）；`ChampionEnsemble` 的 `student_only` 加载模式正是按"成员 checkpoint 内嵌 reference == B"设计的交叉校验（rt:L699-708）。

---

## 3. 推理是否内含 reference 融合

**是，且是该类推理入口的固定行为。** 证据链：

- patch:L146-151 `inference`：
  - L148 `prob_s = self._student_prob(...)`；L149 `prob_r = self._reference_prob(...)`；
  - L150-151 `return blend_probs(prob_r, prob_s, weight_a=0.5)`，注释"Fixed global fusion: 0.5 reference + 0.5 student, all classes"。
- `prob_s`（patch:L116-137）：`tiled_forward`，tile = `self.hr_scale = (1024,1024)`（patch:L39 断言 + L121-122）、stride=768 重叠平均 → 裁回 `img_shape`（L124）→ `rescale` 时 resize 到 `ori_shape`（L125-127，align_corners 同基线）→ sigmoid（L128）→ flip 还原（L129-136）。
- `prob_r`（patch:L139-144）：`reference.eval()` + `torch.no_grad()` + `self.reference.inference(img, img_meta, rescale)`。reference 是父类 `HR100Refinement` 实例（patch:L82），因此调用其 `inference` **不会**递归触发 `HRPlusRefinement.inference` 的二次 0.5/0.5 融合（类 docstring patch:L58-62 明言 "Does NOT recurse into reference.inference"）。
- 推论：把 `HRPlusRefinement` 实例当"单模型"评估的任何路径，拿到都是融合概率。包括 `member_prob` 的 `model(return_loss=False, rescale=True, ...)`（rt:L649；输出结构见 rt:L635-642 docstring：`compute_aupr=True` 路径返回 `[(prob(4,H,W), use_sigmoid, compute_aupr)]`，双重解包）——D/E 冠军成员概率已内含 0.5/0.5 融合。`evaluate_split` 的 `predict_fn` 契约（rt:L521-523）收到的即该融合输出。
- 冠军内部结构（rt）：
  - L744-749 `pf` = Σ w_m×(p+p_flip)/(2Σw)（成员权重 D:3, E:2, B:1, E4:1，rt:L666）——D/E 成员各自内含 0.5/0.5 融合后再加权平均；
  - L751-752 + L793-797：`ps`/`pr` 由 `_raw_prob` **直接调用** `m._student_prob` / `m._reference_prob`（绕过 inference 融合），取的是 D 成员融合前的两个独立分量，供 MA 修正头 f7 特征（rt:L755-773）；
  - L775-791 `predict`：MA 通道过修正头，其余通道保持 `pf`（rt:L789-790）。
- 与外部融合的叠加效应：rt:L800-801 `mix_probs = 0.5×p_champ + 0.5×p_new`。**若 `p_new` 来自 `HRPlusRefinement`（内含 0.5/0.5），则新学生实际权重 = 0.5×0.5 = 0.25，旧 reference 重复进入**——以真实调用路径证实方案 MD 第 184 行的嵌套融合警告。
- 该融合无配置开关：`__init__` 仅强制 semantic_refinement/hard_weight/freeze_bn（patch:L66-68），没有关闭 blend 的选项；学生独立输出只能通过直接调用 `_student_prob` 获得（rt:L793-797 即如此取用）。

---

## 4. 四通道阈值与 mIoU 计算口径

- 类别表 rt:L487：`CLASSES = ['EX','HE','SE','MA']`，对应标签 1..4；0=背景、255=ignore。
- 输入契约（`evaluate_split`，rt:L518-590）：
  - L521-523：predict_fn 返回 `(1,4,H,W)` 或 `(4,H,W)` **概率**；L555-558 仅 squeeze batch 维；L559-561 强制 4 通道 3 维形状。
  - L562-565：prob 空间形状必须等于 `meta[0]['ori_shape'][:2]`。
  - L566-567：强制有限且 ∈ [0,1]（即阈值作用在最终概率上——对冠军/融合输出是融合后概率）。
- GT 对齐：L568-571 形状不符时 `cv2.INTER_NEAREST` 最近邻 resize 到 ori_shape；L572-573 值域校验 `{0,1,2,3,4,255}`，越界报错。
- 阈值与计数（`per_image_counts`，rt:L490-502）：
  - L495：`p = prob[c] > 0.5` —— **严格大于**（docstring L491-492 "Threshold 0.5 strict >"）。
  - L496-500：`TP = p ∧ (gt == c+1)`；`FP = p ∧ ¬(gt==c+1) ∧ ¬(gt==255)`；`FN = ¬p ∧ (gt==c+1)`。
  - ignore(255) 像素上的预测**不计入 FP**（从分母剔除）；背景(0) 像素预测为正**计入 FP**；背景不是类别，不参与 IoU 平均。
- 汇总口径（rt:L577-579 → `global_miou` L505-515）：
  - 逐图计数先按数据集**全量累计**（L577-579 `total[c][j] += cnt[c][j]`），再算 IoU —— **数据集级（global）IoU，不是逐图 IoU 平均**。
  - L512：`IoU = TP / (TP+FP+FN)`。
  - L510-511：`union == 0` 直接 `RuntimeError`，不跳过该类（fail-closed）。
  - L515：`mIoU = np.mean(四类 IoU)` —— 简单算术平均，无背景类。
- split 规模锁：rt:L536-540 期望 `val=76 / test=227`（可由 `expected_count` 覆盖），不符即报错；`audit_split`（rt:L823-867）另校验 train/val/test = 454/76/227、无重复、无交叉。
- 结果落盘：rt:L583-589 `dump_counts` 输出 `miou/per_class/global_counts/per_image`（即 MD 引用的逐图 TP/FP/FN JSON 格式）。
- 关联配置：cfg:L56-57 `evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')` 是 mmseg 原生 EvalHook 配置；**本项目驱动脚本的统一评估口径以 rt 的 `evaluate_split` 为准**（mmseg 原生 hook 在 4 类 sigmoid fork 下的行为不在本次审计文件可验证范围）。

---

## 5. 与方案 MD 断言的逐条对照

| 方案 MD 断言（MD 行号） | 审计结果 | 代码证据 |
|---|---|---|
| 2.3-1（MD L59）：`__init__` 强制从 source_checkpoint 严格加载旧 B 权重；`init_weights()` 是空操作；仅改 config 的 pretrained 字段不足以证明新初始化生效 | **属实** | patch:L64-65、L78、L91-94；cfg:L28-29 仅使 pretrained 为空，真正的权重事件是 L78 |
| 2.3-2（MD L60-61）：推理时包含 `0.5×reference + 0.5×student`，"单模型"实为双路径融合 | **属实** | patch:L146-151 |
| 2.3-3（MD L61-62）：评估四个独立概率通道、严格 `p>0.5`、按数据集累计 TP/FP/FN、四类 IoU 平均、背景不计 | **属实** | rt:L490-502、L505-515、L577-579 |
| 第 7 节（MD L184）：若新模型已含 0.5/0.5 内融合，再做外部 0.5 融合，新学生实际占 0.25、旧 reference 重复进入 | **属实** | patch:L146-151 × rt:L800-801 |
| 4.1（MD L87）：需单独命名 student_only / reference_only / legacy_blend / champion_complete | 四条输出路径在代码中均有唯一对应物（见下） | 见下表 |

四条命名的代码对应物：

| 命名 | 代码路径 |
|---|---|
| `student_only` | `_student_prob`（patch:L116-137，tiled 1024/768 → resize → sigmoid） |
| `reference_only` | `_reference_prob`（patch:L139-144）；R6 训练特征缓存即按它构建（rt:L437，`ReferenceCache.build` 调 `model._reference_prob(img, meta, False)`，存 2048×2048 fp16，rt:L259-260） |
| `legacy_blend` | `HRPlusRefinement.inference`（patch:L146-151，0.5/0.5） |
| `champion_complete` | `ChampionEnsemble.predict`（rt:L775-791，4 成员加权 + 水平翻转 TTA + MA 修正头） |

---

## 6. 范围与限制

- 本审计为**静态阅读**，未执行代码、未做动态权重张量验证（与方案 MD L340 "尚未完成：参数加载动态验证"一致；MD 4.1 建议的构造后/加载后/首更新后张量摘要仍需在 GPU 环境实测）。
- `fusion_core.py`（`strict_initialize`、`blend_probs` 实现）、`hr100_refinement.py`（父类 forward/simple_test）、`hrplus_core.py`（`tiled_forward`/`foreground_crop_bbox` 等）不在本次给定的三个审计文件内；涉及处均以调用点代码与 docstring 为证并已在正文标注（1.4 节、3 节）。
- "冠军成员 D/E 属 HRPlusRefinement 家族"的判断依据 rt:L688-695 的类型检测逻辑与 rt:L699-708 docstring；成员实际清单来自 `champion_cfg`（未在给定文件内）。
- 未读取历史 val/test 计数 JSON，未复算任何指标（方案 MD 已另行独立复算，本审计不重复）。
- driver 脚本（构造前打补丁 source_checkpoint 为绝对路径的行为，rt:L57-59 自述）不在审计文件内，该一步为 runtime 自述而非直接证据。
