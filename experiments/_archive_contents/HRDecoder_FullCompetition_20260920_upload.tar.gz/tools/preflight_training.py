#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""preflight_training.py — 上机前的 **CPU** 训练链路自检（离线、不训练真实数据）。

这个脚本回答一个问题：**"训练代码本身跑得通吗？"**（与 tools/preflight_contract.py
回答的"配置/数据契约齐了吗"互补）。它用合成 mini 模型/数据覆盖公共训练核心，
不需要 GPU、不需要真实 DDR 数据、不联网、不下载权重。

覆盖项：
  E0  依赖体检（python/torch/numpy/PIL/opencv/yaml 版本）
  E1  导入体检（trainer.* 与 train.py 可导入）
  E2  配置两层闸门（readiness / calibration）与 test split 政策
  E3  模型构建（U 臂真实 builtin ConvNeXt-S；H 臂按旧仓库可导入性给
      DEPENDENCY_BLOCKED 或真实构建）
  E4  形状契约（严格 N×4×H×W；非 32 倍数必须报错）
  E5  参数组互斥且完整（无遗漏、无重复、冻结参数不入组）
  E6  一次真实前向/反向（真实模型可用时用真实模型，否则用合成 mini 模型）
  E7  valid 掩蔽语义（valid=0 不产生梯度；整批全无效 ⇒ 零梯度且无 NaN）
  E8  BN 策略审计（backbone BN 统计冻结、affine 可训练）
  E9  checkpoint 往返（weights_only 安全加载、字段齐全、RNG 恢复）
  E10 同 seed 可复现（初始权重逐元素一致）
  E11 整图/滑窗评估路径（合成 evaluator 跑通，指标手算一致）
  E12 GPU-only 项（AMP / gradient checkpointing / micro_batch 显存）
      → CALIBRATION_REQUIRED，不阻塞短时校准开机

状态语义：
  PASS                 真实通过（计入通过数）
  FAIL                 真实失败（必须修）
  DEPENDENCY_BLOCKED   依赖不满足（不计 PASS 也不计 FAIL）
  READINESS_BLOCKED    配置 readiness 层未清空
  CALIBRATION_REQUIRED 只能在 GPU 上校准
  SKIP                 主动跳过（显式记录原因）

退出码：0=无 FAIL 且无 READINESS_BLOCKED；1=存在 FAIL/READINESS_BLOCKED；
        3=环境依赖致命缺失（torch/numpy 缺失，连自检都跑不起来）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

PASS = "PASS"
FAIL = "FAIL"
DEP_BLOCKED = "DEPENDENCY_BLOCKED"
READINESS_BLOCKED = "READINESS_BLOCKED"
CALIB_REQUIRED = "CALIBRATION_REQUIRED"
SKIP = "SKIP"


class Results:
    def __init__(self):
        self.items = []

    def add(self, name, status, detail=""):
        self.items.append({"check": name, "status": status, "detail": detail})
        return status

    def count(self, status):
        return sum(1 for i in self.items if i["status"] == status)

    def summary(self) -> dict:
        return {
            "pass": self.count(PASS),
            "fail": self.count(FAIL),
            "dependency_blocked": self.count(DEP_BLOCKED),
            "readiness_blocked": self.count(READINESS_BLOCKED),
            "calibration_required": self.count(CALIB_REQUIRED),
            "skip": self.count(SKIP),
            "total": len(self.items),
        }


# ---------------------------------------------------------------------------
# 合成 fixture（fake model / fake data）—— 只用于测公共训练核心
# ---------------------------------------------------------------------------

def make_fake_model(*, in_ch=3, num_classes=4, hidden=8):
    """极小分割模型：三层卷积，输出严格 [N,4,H,W] logits。

    命名刻意分成 backbone.* / head.* 两组，用于检验参数分组逻辑。
    """
    import torch.nn as nn

    class FakeSeg(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(hidden, num_classes, 1)

        def forward(self, x):
            return self.head(self.backbone(x))

        def eval_forward(self, x):
            return self.forward(x)

    return FakeSeg()


class SyntheticSegDataset:
    """合成分割数据集：与 DDRSegDataset 的对外契约一致（image/target/valid/meta）。

    可通过 `all_invalid=True` 构造"四类全未标注"的样本来检验零梯度语义；
    可通过 `load_raw` 提供整图评估所需的未归一化数据。
    """

    def __init__(self, n=6, size=64, num_classes=4, all_invalid=False, split="val", seed=0):
        import numpy as np

        self.n, self.size, self.num_classes = n, size, num_classes
        self.all_invalid = all_invalid
        self.split = split
        self.epoch = 0
        self._data = []
        rs = np.random.RandomState(seed)
        for i in range(n):
            img = rs.randint(0, 255, size=(size, size, 3)).astype(np.uint8)
            target = np.zeros((num_classes, size, size), dtype=np.float32)
            target[i % num_classes, 8:24, 8:24] = 1.0
            valid = np.zeros((num_classes, size, size), dtype=np.float32) if all_invalid \
                else np.ones((num_classes, size, size), dtype=np.float32)
            self._data.append((img, target, valid, f"synth_{i}"))

    def __len__(self):
        return self.n

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def load_raw(self, idx):
        return self._data[idx]

    def __getitem__(self, idx):
        import torch

        img, target, valid, sid = self._data[idx]
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        return {
            "image": x, "target": torch.from_numpy(target.copy()),
            "valid": torch.from_numpy(valid.copy()),
            "meta": {"sample_id": sid, "split": self.split, "source": "synthetic"},
        }


def make_runner(model, dataset, *, device="cpu", total_updates=4, grad_accum=2,
                micro_batch=2, mode="calibration", out_dir=None, evaluator=None,
                backbone_prefixes=("backbone",), backbone_norm_eval=False,
                eval_progress_points=(1.0,)):
    """用合成数据装配一个 StageRunner（公共训练核心，不含任何真实数据依赖）。"""
    from trainer.engine import StageRunner
    from trainer.loss import MaskedBinaryCEDice
    from trainer.optim import build_optimizer, build_param_groups
    from trainer.seed import seed_everything

    seed_everything(7)
    groups = build_param_groups(
        model, backbone_lr=1e-3, decoder_lr=1e-3, weight_decay=0.0,
        backbone_prefixes=tuple(backbone_prefixes),
    )
    opt = build_optimizer({"optimizer": "adamw", "weight_decay": 0.0}, groups)
    cfg = {
        "arm_id": "SYNTH", "family": "synthetic", "stage": "ddr_finetune", "mode": mode,
        "is_calibration": mode == "calibration", "seed": 7, "total_updates": total_updates,
        "micro_batch": micro_batch, "grad_accum": grad_accum,
        "effective_batch": micro_batch * grad_accum, "warmup_updates": 1,
        "cosine_min_ratio": 0.01, "grad_clip": 1.0, "mixed_precision": False,
        "eval_progress_points": list(eval_progress_points), "log_every": 0, "num_workers": 0,
        "backbone_prefixes": tuple(backbone_prefixes),
        "backbone_norm_eval": backbone_norm_eval,
        "config_sha256": "synthetic-config-sha", "data_manifest_sha256": "synthetic-manifest-sha",
        "manifest_path": None, "out_dir": out_dir,
    }
    return StageRunner(
        model=model, dataset=dataset, loss_fn=MaskedBinaryCEDice(), optimizer=opt,
        cfg=cfg, device=device, evaluator=evaluator, logger=lambda *a, **k: None,
    )


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------

def check_env(res: Results):
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        res.add("E0 依赖体检", FAIL, f"torch 无法导入: {e}")
        return False
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}
    for mod in ("numpy", "PIL", "cv2", "yaml"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"MISSING ({type(e).__name__})"
    res.add("E0 依赖体检", PASS, info)
    return True


def check_imports(res: Results):
    try:
        import train  # noqa: F401
        from trainer import checkpoint, config, engine, eval as teval, loss, metrics, optim, paths, seed  # noqa: F401
        from trainer.data import augment, common, ddr_dataset, external_dataset  # noqa: F401
        from trainer.models import build_model, probe_family  # noqa: F401
        res.add("E1 导入体检", PASS, "train.py + trainer.*（含 data/models 子包）全部可导入")
        return True
    except Exception as e:  # noqa: BLE001
        res.add("E1 导入体检", FAIL, f"{type(e).__name__}: {e}")
        return False


def check_config_gate(res: Results, cfg_path: str):
    from trainer.config import (
        checkpoint_selection_splits, ddr_data_config, gate_config, load_yaml,
        placeholder_report, resolve_stage_section, validate_config,
    )
    from trainer.gate import GateError

    try:
        cfg = load_yaml(cfg_path)
        validate_config(cfg, cfg_path)
        rep = placeholder_report(cfg)
        arm = cfg["arm_id"]
        detail = {"arm": arm, "readiness": rep["readiness"], "calibration_n": len(rep["calibration"]),
                  "readiness_state": rep["readiness_state"]}
        if rep["readiness"]:
            res.add(f"E2 配置闸门 {cfg_path}", READINESS_BLOCKED,
                    f"readiness 层占位符未清空: {rep['readiness']}")
        else:
            gate_config(cfg, mode="calibration", stage="ddr_finetune", path=cfg_path)
            res.add(f"E2 配置闸门 {cfg_path}", PASS, detail)
        checkpoint_selection_splits(cfg)
        for stage in ("external_pretrain", "ddr_finetune"):
            try:
                sec = resolve_stage_section(cfg, stage)
                if stage == "ddr_finetune":
                    ddr_data_config(cfg)
                res.add(f"E2 stage {arm}/{stage}", PASS,
                        f"节存在，键 {sorted(sec.keys())[:6]}…")
            except GateError as e:
                msg = str(e)
                if "没有 external_pretrain" in msg:
                    res.add(f"E2 stage {arm}/{stage}", SKIP, msg)
                elif "未填写" in msg or "占位符" in msg or "未回填" in msg:
                    # 与 E2 配置闸门同源（readiness 层未清空）：不要重复计成 FAIL
                    res.add(f"E2 stage {arm}/{stage}", READINESS_BLOCKED, msg)
                else:
                    res.add(f"E2 stage {arm}/{stage}", FAIL, msg)
    except Exception as e:  # noqa: BLE001
        res.add(f"E2 配置闸门 {cfg_path}", FAIL, f"{type(e).__name__}: {e}")


def check_test_split_rejection(res: Results):
    from trainer.config import assert_no_test_split
    from trainer.gate import GateError

    ok, msgs = True, []
    for split, expect in (("test", "reject"), ("val", "accept")):
        try:
            assert_no_test_split({}, split=split, role="probe")
            got = "accept"
        except GateError:
            got = "reject"
        if got != expect:
            ok = False
            msgs.append(f"{split} 期望 {expect} 实际 {got}")
    res.add("E2b test split 硬拒绝", PASS if ok else FAIL,
            "; ".join(msgs) or "train/val 放行，test 被 GateError 拒绝")


def check_model_build(res: Results, cfg_path: str):
    """U 臂真实构建 + 形状契约；H 臂按旧仓库可导入性给 DEPENDENCY_BLOCKED。"""
    from trainer.config import load_yaml
    from trainer.models import DependencyBlocked, build_model, probe_family

    cfg = load_yaml(cfg_path)
    arm = cfg["arm_id"]
    family = cfg["structure"]["family"]
    probe = probe_family(family, cfg)
    if probe["status"] != "OK":
        status = DEP_BLOCKED if probe["status"] == DEP_BLOCKED else READINESS_BLOCKED
        res.add(f"E3 模型构建 {arm}", status, probe["detail"])
        return None, None
    try:
        model, meta = build_model(cfg, stage="ddr_finetune", gradient_checkpointing=False)
    except DependencyBlocked as e:
        res.add(f"E3 模型构建 {arm}", DEP_BLOCKED, str(e))
        return None, None
    except Exception as e:  # noqa: BLE001
        res.add(f"E3 模型构建 {arm}", FAIL, f"{type(e).__name__}: {e}")
        return None, None
    n = sum(p.numel() for p in model.parameters())
    res.add(f"E3 模型构建 {arm}", PASS,
            f"{family} params={n/1e6:.2f}M backbone_prefixes={meta['backbone_prefixes']} "
            f"norm_eval={meta['backbone_norm_eval']}")
    check_shape_contract(res, model, arm=arm, family=family)
    return model, meta


def check_shape_contract(res, model, *, arm, family):
    """严格 N×4×H×W；非 32 倍数必须报错（不静默 resize/pad）。"""
    import torch
    from trainer.gate import GateError

    model.eval()
    problems = []
    with torch.no_grad():
        for (h, w) in ((64, 64), (96, 32)):
            out = model(torch.zeros(1, 3, h, w))
            if tuple(out.shape) != (1, 4, h, w):
                problems.append(f"({h},{w}) -> {tuple(out.shape)}")
        bad = False
        try:
            model(torch.zeros(1, 3, 48, 48))
        except (GateError, RuntimeError, ValueError) as e:
            bad = f"{type(e).__name__}"
        if not bad:
            problems.append("非 32 倍数输入 (48,48) 未报错")
    if problems:
        res.add(f"E4 形状契约 {arm}", FAIL, "; ".join(problems))
    else:
        res.add(f"E4 形状契约 {arm}", PASS,
                f"{family}: 输出严格 1×4×H×W；非 32 倍数输入被拒绝（{bad}）")


def check_param_groups(res: Results, model=None, *, backbone_prefixes=("backbone",)):
    """参数组互斥且完整（用真实模型或合成模型）。"""
    from trainer.optim import audit_param_groups, build_optimizer, build_param_groups
    from trainer.gate import GateError

    m = model if model is not None else make_fake_model()
    try:
        pg = build_param_groups(m, backbone_lr=1e-4, decoder_lr=2e-4,
                                weight_decay=0.01, backbone_prefixes=tuple(backbone_prefixes))
        opt = build_optimizer({"optimizer": "adamw", "weight_decay": 0.01}, pg)
        audit = audit_param_groups(m, opt)
    except GateError as e:
        res.add("E5 参数组互斥完整", FAIL, str(e))
        return
    # 反例：故意漏掉一个参数组，审计必须报错（证明审计真在起作用）
    try:
        rows = audit_param_groups(m, build_optimizer(
            {"optimizer": "adamw"}, [{"params": pg[0]["params"], "lr": 1e-3, "name": "backbone"}]))
        leak = rows and audit["n_trainable_tensors"] == 0
    except GateError:
        leak = False
    res.add("E5 参数组互斥完整", PASS,
            f"trainable_tensors={audit['n_trainable_tensors']} 组={audit['groups']} "
            f"（反例：漏组会被 GateError 拦下）")


def check_forward_backward(res: Results, model=None, *, device="cpu", backbone_prefixes=("backbone",)):
    """一次真实前向/反向（有真实模型用真实模型，否则合成 mini 模型）。"""
    import torch
    tag = "真实模型" if model is not None else "合成 mini 模型"
    ds = SyntheticSegDataset(n=4, size=64)
    try:
        runner = make_runner(model or make_fake_model(), ds, device=device,
                             backbone_prefixes=backbone_prefixes)
        out = runner.dry_run_check()
    except Exception as e:  # noqa: BLE001
        res.add("E6 前向/反向", FAIL, f"{tag}: {type(e).__name__}: {e}")
        return
    ok = out["tensors_with_grad"] > 0 and out["loss"] == out["loss"]
    res.add("E6 前向/反向", PASS if ok else FAIL,
            f"{tag}: loss={out['loss']:.6f} 有梯度的张量={out['tensors_with_grad']} "
            f"active_classes={out['parts'].get('active_classes')}")


def check_valid_masking(res: Results):
    """valid=0 整类屏蔽：无梯度；整批全无效 ⇒ 零梯度且无 NaN。"""
    import torch
    from trainer.loss import masked_bce_dice_loss

    msgs, ok = [], True
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.zeros(2, 4, 16, 16)
    target[:, 0, :4, :4] = 1.0
    # 只有通道 0 有效 → 通道 1..3 必须完全无梯度贡献
    valid = torch.zeros(2, 4, 16, 16)
    valid[:, 0] = 1.0
    loss, parts = masked_bce_dice_loss(logits, target, valid)
    loss.backward()
    g = logits.grad
    if float(g[:, 1:].abs().sum()) != 0.0:
        ok = False
        msgs.append("valid=0 的通道产生了非零梯度（缺类被当成阴性）")
    if int(parts["active_classes"]) != 1:
        ok = False
        msgs.append(f"active_classes={parts['active_classes']} 应为 1")

    # 整批全无效：loss=0、无 NaN、梯度全 0
    logits2 = torch.randn(1, 4, 8, 8, requires_grad=True)
    zero_t = torch.zeros(1, 4, 8, 8)
    loss2, parts2 = masked_bce_dice_loss(logits2, zero_t, zero_t)
    loss2.backward()
    if float(loss2) != 0.0 or not torch.isfinite(loss2):
        ok = False
        msgs.append(f"全无效 batch 的 loss={float(loss2)} 应为有限 0")
    if float(logits2.grad.abs().sum()) != 0.0:
        ok = False
        msgs.append("全无效 batch 产生了非零梯度")
    if parts2["active_classes"] != 0:
        ok = False
        msgs.append(f"全无效 batch 的 active_classes={parts2['active_classes']} 应为 0")

    # 手算一致性：单像素单类的 BCE 应对得上解析值
    lg = torch.zeros(1, 4, 1, 1, requires_grad=True)
    tg = torch.ones(1, 4, 1, 1)
    vg = torch.zeros(1, 4, 1, 1)
    vg[:, 0] = 1.0
    l3, _ = masked_bce_dice_loss(lg, tg, vg, ce_weight=1.0, dice_weight=0.0)
    expect_bce = 0.6931471805599453     # -log(sigmoid(0)) = log 2
    expect_dice = 1.0 - (2 * 0.5 + 1e-5) / (0.5 + 1.0 + 1e-5)
    if abs(float(l3) - expect_bce) > 1e-5:
        ok = False
        msgs.append(f"ce_weight=1/dice_weight=0 时 loss={float(l3):.6f} != log2={expect_bce:.6f}")
    l4, _ = masked_bce_dice_loss(lg, tg, vg, ce_weight=0.0, dice_weight=1.0)
    if abs(float(l4) - expect_dice) > 1e-5:
        ok = False
        msgs.append(f"纯 dice 时 loss={float(l4):.6f} != 手算 {expect_dice:.6f}")

    res.add("E7 valid 掩蔽语义", PASS if ok else FAIL,
            "; ".join(msgs) or "valid=0 通道零梯度；全无效 batch 零梯度无 NaN；BCE/Dice 手算一致")


def check_norm_policy(res: Results, model=None, *, backbone_prefixes=("backbone",)):
    """BN 统计冻结 / affine 可训练。用合成 BN 模型必测一遍（真实 U 臂没有 BN，
    只用真实模型会把这条检查变成永远空转的假 PASS）。"""
    import torch.nn as nn
    from trainer.optim import audit_norm_policy, freeze_bn_stats

    class _BNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
            self.head = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))

    checks = []
    for label, m, pref in (("合成 BN 模型", _BNModel(), ("backbone",)),
                           ("真实模型", model, tuple(backbone_prefixes))):
        if m is None:
            continue
        m.train()
        frozen = freeze_bn_stats(m, backbone_prefixes=pref)
        rep = audit_norm_policy(m, backbone_prefixes=pref, backbone_norm_eval=True)
        checks.append((label, frozen, rep))
    ok = True
    parts = []
    for label, frozen, rep in checks:
        if not (rep["batchnorm_in_eval_after_train"] and rep["batchnorm_affine_trainable"]):
            ok = False
        if label == "合成 BN 模型" and (frozen != 1 or rep["batchnorm_backbone"] != 1):
            ok = False
        parts.append(f"{label}: 冻结={frozen} bn_backbone={rep['batchnorm_backbone']} "
                     f"bn_total={rep['batchnorm_total']} ln={rep['layernorm_total']} "
                     f"gn={rep['groupnorm_total']} issues={rep['issues']}")
    res.add("E8 BN 策略审计", PASS if ok else FAIL, " ｜ ".join(parts))


def check_checkpoint_roundtrip(res: Results, tmp_dir: str):
    import torch
    from trainer.checkpoint import (
        build_checkpoint_payload, load_checkpoint, resume_state, save_checkpoint,
        verify_resume_compatibility,
    )
    from trainer.optim import build_optimizer, build_param_groups, scheduler_state

    try:
        model = make_fake_model()
        ds = SyntheticSegDataset(n=4, size=32)
        runner = make_runner(model, ds, total_updates=2, out_dir=tmp_dir)
        p1 = runner.save(metrics_at_save={"mIoU_lesion4": 0.5}, tag="rt")
        ckpt = load_checkpoint(p1)
        required = ("model_state", "optimizer_state", "scheduler_state", "scaler_state",
                    "update", "micro_step", "seed", "config_sha256",
                    "data_manifest_sha256", "total_updates", "rng")
        missing = [k for k in required if k not in ckpt]
        if missing:
            res.add("E9 checkpoint 往返", FAIL, f"缺字段 {missing}")
            return
        # 权重一致
        m2 = make_fake_model()
        o2 = build_optimizer({"optimizer": "adamw"}, build_param_groups(
            m2, backbone_lr=1e-3, decoder_lr=1e-3, weight_decay=0.0,
            backbone_prefixes=("backbone",)))
        st = resume_state(ckpt, model=m2, optimizer=o2, scaler=None)
        same = all(torch.equal(a, b) for a, b in
                   zip(model.state_dict().values(), m2.state_dict().values()))
        res.add("E9 checkpoint 往返", PASS if same else FAIL,
                f"字段齐全；权重逐元素一致={same}；恢复 update={st['update']} "
                f"micro_step={st['micro_step']}；weights_only 安全加载成功")
    except Exception as e:  # noqa: BLE001
        res.add("E9 checkpoint 往返", FAIL, f"{type(e).__name__}: {e}")


def check_seed_repeatability(res: Results):
    import torch
    from trainer.seed import seed_everything

    seed_everything(123, deterministic=True)
    a = make_fake_model()
    seed_everything(123, deterministic=True)
    b = make_fake_model()
    same = all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    res.add("E10 同 seed 可复现", PASS if same else FAIL,
            f"两次构建的初始权重逐元素一致={same}")


def check_eval_path(res: Results, model=None):
    """整图评估路径：合成 evaluator 跑通 + 指标手算一致。"""
    import numpy as np
    import torch
    from trainer.eval import WholeImageEvaluator
    from trainer.metrics import SegMetricsAccumulator

    # (a) 指标手算一致性
    # probs=0.9 全阳性，每类恰 1 个正像素 / 4 像素 ⇒ TP=1, FP=3, FN=0
    #   IoU = 1/(1+3) = 0.25；PPV = 1/4 = 0.25；Recall = 1；F1 = 2/(2+3) = 0.4
    acc = SegMetricsAccumulator(threshold=0.5)
    probs = np.full((1, 4, 2, 2), 0.9, dtype=np.float32)
    target = np.zeros((1, 4, 2, 2), dtype=np.float32)
    target[:, :, 0, 0] = 1.0
    valid = np.ones_like(target)
    acc.update(probs, target, valid)
    r = acc.compute()
    ex = r["per_class"]["EX"]
    hand_ok = (
        ex["tp"] == 1 and ex["fp"] == 3 and ex["fn"] == 0
        and abs(ex["iou"] - 0.25) < 1e-9
        and abs(ex["ppv"] - 0.25) < 1e-9
        and abs(ex["recall"] - 1.0) < 1e-9
        and abs(ex["f1"] - 0.4) < 1e-9
        and abs(r["mIoU_lesion4"] - 0.25) < 1e-9
    )
    # valid=0 的像素完全不计入（否则会被当成负样本压低指标）
    acc_v = SegMetricsAccumulator(threshold=0.5)
    valid_v = np.zeros_like(target)
    valid_v[:, :, 0, 0] = 1.0
    acc_v.update(probs, target, valid_v)
    rv = acc_v.compute()["per_class"]["EX"]
    valid_ok = rv["tp"] == 1 and rv["fp"] == 0 and rv["fn"] == 0
    # 阈值语义：p 恰为 0.5 不算阳性
    acc2 = SegMetricsAccumulator(threshold=0.5)
    acc2.update(np.full((1, 4, 1, 1), 0.5, dtype=np.float32),
                np.ones((1, 4, 1, 1), dtype=np.float32),
                np.ones((1, 4, 1, 1), dtype=np.float32))
    r2 = acc2.compute()
    thr_ok = r2["per_class"]["EX"]["tp"] == 0 and r2["per_class"]["EX"]["fn"] == 1
    # 完美分类器 AUPR = 1.0（历史实现在递减方向累加，只能得 0.2）
    acc3 = SegMetricsAccumulator(threshold=0.5)
    p3 = np.concatenate([np.full((1, 4, 1, 1), 0.99, np.float32)] * 3 +
                       [np.full((1, 4, 1, 1), 0.01, np.float32)] * 3, axis=2)
    t3 = np.concatenate([np.ones((1, 4, 1, 1), np.float32)] * 3 +
                        [np.zeros((1, 4, 1, 1), np.float32)] * 3, axis=2)
    acc3.update(p3, t3, np.ones_like(t3))
    aupr = acc3.compute()["per_class"]["EX"]["aupr"]
    aupr_ok = abs(aupr - 1.0) < 1e-9

    msgs = []
    if not hand_ok:
        msgs.append(f"手算 IoU 不一致（期望 TP=1 FP=3 FN=0 IoU=0.25）: {ex}")
    if not valid_ok:
        msgs.append(f"valid=0 像素未正确屏蔽: {rv}")
    if not thr_ok:
        msgs.append("阈值语义：p==0.5 被当成阳性（应为严格 p>0.5）")
    if not aupr_ok:
        msgs.append(f"完美分类器 AUPR={aupr:.4f} != 1.0")

    # (b) 端到端整图 evaluator（合成模型 + 合成数据集）
    try:
        ds = SyntheticSegDataset(n=3, size=64, split="val")
        ev = WholeImageEvaluator(model or make_fake_model(), ds, device="cpu",
                                 mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0],
                                 mode="whole_image")
        rep = ev.run()
        ran = rep["n_images"] == 3 and "mIoU_lesion4" in rep
    except Exception as e:  # noqa: BLE001
        ran = False
        msgs.append(f"整图 evaluator 运行失败: {type(e).__name__}: {e}")

    res.add("E11 整图评估路径", PASS if (hand_ok and valid_ok and thr_ok and aupr_ok and ran) else FAIL,
            "; ".join(msgs) or ("累计 TP/FP/FN + valid 屏蔽 + p>0.5 + AUPR 手算一致；"
                                "合成整图 evaluator 跑通"))


def check_gpu_only(res: Results):
    """GPU-only 项一律 CALIBRATION_REQUIRED（无论本机是否有 CUDA）。

    本机即使有 GPU，也不代表"上传服务器后的那张卡"的显存/AMP 数值行为一致；
    这些必须在目标机上实测，绝不用本机结果冒充。
    """
    detail = ("mixed_precision(GradScaler) / gradient_checkpointing / "
              "micro_batch 显存实测 / AMP 数值稳定性 必须在目标 GPU 上实测；"
              "本机结果不作为校准依据")
    res.add("E12 GPU-only 项", CALIB_REQUIRED, detail)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="训练链路 CPU 自检（离线、合成数据、不训练真实数据）")
    ap.add_argument("--config", action="append", default=[],
                    help="待检查的臂配置 YAML（可重复）；默认检查 configs/*.yaml")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--tmp-dir", default=None, help="checkpoint 往返用的临时目录")
    ap.add_argument("--skip-model-build", action="store_true",
                    help="跳过真实模型构建（只跑合成核心自检）")
    return ap.parse_args(argv)


def run_checks(*, configs=None, tmp_dir=None, skip_model_build=False, verbose=False) -> Results:
    res = Results()
    if not check_env(res):
        return res
    if not check_imports(res):
        return res

    import tempfile
    tmp = tmp_dir or tempfile.mkdtemp(prefix="preflight_training_")
    os.makedirs(tmp, exist_ok=True)

    if not configs:
        cfg_dir = os.path.join(REPO_ROOT, "configs")
        configs = sorted(os.path.join(cfg_dir, f) for f in os.listdir(cfg_dir)
                         if f.lower().endswith((".yaml", ".yml")))

    check_test_split_rejection(res)

    real_model, real_meta, real_arm = None, None, None
    if not skip_model_build:
        for cfg_path in configs:
            # U 臂优先（可在本机真实构建），H 臂多半 DEPENDENCY_BLOCKED
            import yaml
            try:
                arm = yaml.safe_load(open(cfg_path, encoding="utf-8")).get("arm_id")
            except Exception:  # noqa: BLE001
                arm = None
            if arm == "U0" and real_model is None:
                real_model, real_meta = check_model_build(res, cfg_path)
                real_arm = arm
            else:
                check_model_build(res, cfg_path)
    for cfg_path in configs:
        check_config_gate(res, cfg_path)

    backbone_prefixes = real_meta["backbone_prefixes"] if real_meta else ("backbone",)
    check_param_groups(res, real_model if real_arm == "U0" else None,
                       backbone_prefixes=backbone_prefixes)
    check_forward_backward(res, real_model if real_arm == "U0" else None,
                           backbone_prefixes=backbone_prefixes)
    check_valid_masking(res)
    check_norm_policy(res, real_model if real_arm == "U0" else None,
                      backbone_prefixes=backbone_prefixes)
    check_checkpoint_roundtrip(res, tmp)
    check_seed_repeatability(res)
    check_eval_path(res, real_model if real_arm == "U0" else None)
    check_gpu_only(res)
    return res


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"[preflight_training] FATAL: torch 缺失，无法自检训练链路: {e}", file=sys.stderr)
        return 3
    try:
        res = run_checks(configs=args.config or None, tmp_dir=args.tmp_dir,
                         skip_model_build=args.skip_model_build)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 3

    print("=" * 78)
    print("训练链路 CPU 自检（合成 fake model/data；不训练真实数据、不联网）")
    print("=" * 78)
    for item in res.items:
        print(f"  [{item['status']:<20}] {item['check']}")
        if item["detail"]:
            print(f"      {item['detail']}")
    s = res.summary()
    print("-" * 78)
    print(f"[preflight_training] PASS={s['pass']} FAIL={s['fail']} "
          f"DEPENDENCY_BLOCKED={s['dependency_blocked']} "
          f"READINESS_BLOCKED={s['readiness_blocked']} "
          f"CALIBRATION_REQUIRED={s['calibration_required']} SKIP={s['skip']}")
    local_ready = s["fail"] == 0 and s["readiness_blocked"] == 0
    print(f"[preflight_training] LOCAL_READY={local_ready} "
          f"GPU_CALIBRATION_READY={local_ready} （依赖阻塞不计 PASS）")
    if args.output_json:
        payload = {"summary": s, "items": res.items,
                   "local_ready": local_ready,
                   "note": "DEPENDENCY_BLOCKED/CALIBRATION_REQUIRED 不计入 PASS"}
        with open(args.output_json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[preflight_training] report -> {os.path.abspath(args.output_json)}")
    return 0 if local_ready else 1


if __name__ == "__main__":
    sys.exit(main())
