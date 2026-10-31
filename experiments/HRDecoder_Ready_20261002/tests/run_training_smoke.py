# -*- coding: utf-8 -*-
r"""run_training_smoke.py — 训练链路合成冒烟测试（离线、CPU、合成 fake model/data）。

回答的问题：**公共训练核心的行为是否正确**（不是"模型好不好"）。所有断言都
基于合成 mini 模型 + 合成数据，不需要 GPU、不需要真实 DDR 数据、不联网、
不下载权重、不训练任何真实数据。

用例（T0–T15）：
  T0  py_compile：train.py + trainer/**.py + tools/preflight_training.py + tests/*.py
  T1  tools/preflight_training.py 合成核心自检 FAIL=0
  T2  梯度累积以 optimizer update 计数（9 micro / accum=3 ⇒ 恰 3 次 update）
  T3  scheduler 按 optimizer update（LR 序列 == lr_multiplier 手算）
  T4  checkpoint 断点后 update 连续（2 + 2 == 4，且不重放/不跳变）
  T5  checkpoint 字段齐全（模型/优化器/scheduler/scaler/update/seed/config hash/manifest hash）
  T6  校准 checkpoint 不得作为正式训练恢复源
  T7  formal 模式缺整图 evaluator ⇒ fail closed
  T8  CLI 硬拒绝 test 作为训练 split（子进程实测退出码）
  T9  CLI formal 模式遇占位符 ⇒ 非零退出（不放宽正式闸门）
  T10 CLI formal 模式禁止校准覆盖
  T11 指标手算一致（TP/FP/FN、valid 屏蔽、p>0.5、完美分类器 AUPR==1.0）
  T12 valid=0 整类屏蔽零梯度；整批全无效零梯度且无 NaN
  T13 参数组互斥且完整（含漏组负例）
  T14 真实 U 臂模型：输出严格 N×4×H×W；detail_branch=true 必须被拒绝
  T15 train.py 端到端 dry-run（合成 U0 配置 + 合成 DDR 树，calibration 模式）

依赖阻塞（H 臂旧仓库 mmcv/mmengine 不可导入）单列 DEPENDENCY_BLOCKED，不计 PASS。

退出码：0=全部断言通过；1=存在失败用例；4=环境依赖缺失（torch/PyYAML/Pillow/opencv）。
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
TRAIN_PY = os.path.join(REPO_ROOT, "train.py")
for p in (REPO_ROOT, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"[run_training_smoke] FATAL: torch 不可导入: {e}", file=sys.stderr)
    sys.exit(4)
try:
    import yaml
except Exception as e:  # noqa: BLE001
    print(f"[run_training_smoke] FATAL: PyYAML 不可导入: {e}", file=sys.stderr)
    sys.exit(4)
try:
    from PIL import Image
except Exception as e:  # noqa: BLE001
    print(f"[run_training_smoke] FATAL: Pillow 不可导入: {e}", file=sys.stderr)
    sys.exit(4)

import numpy as np  # noqa: E402

from preflight_training import (  # noqa: E402  复用同一套合成 fixture
    SyntheticSegDataset,
    make_fake_model,
    make_runner,
    run_checks,
)

TMP = os.path.normpath(os.path.join(REPO_ROOT, "tests", "_tmp_training"))
CONFIGS_OUT = os.path.join(TMP, "configs")
DDR_ROOT = os.path.join(TMP, "ddr")
WEIGHTS_DIR = os.path.join(TMP, "weights")
REPORTS = os.path.join(TMP, "reports")

_results = []


def record(case, ok, detail="", status=None):
    _results.append((case, ok, detail, status or ("PASS" if ok else "FAIL")))
    print(f"  [{'PASS' if ok else 'FAIL'}] {case}" + (f" — {detail}" if detail and not ok else ""))


def skip(case, detail):
    _results.append((case, True, detail, "DEPENDENCY_BLOCKED"))
    print(f"  [DEPENDENCY_BLOCKED] {case} — {detail}")


# ---------------------------------------------------------------------------
# 合成资产
# ---------------------------------------------------------------------------

def write_fused_label_png(path, size, plan):
    """DDR 融合标签 PNG（mode L，值域 0..4），plan = [(class_id, y0,y1,x0,x1)]。"""
    arr = np.zeros((size, size), dtype=np.uint8)
    for cid, y0, y1, x0, x1 in plan:
        arr[y0:y1, x0:x1] = cid
    Image.fromarray(arr, mode="L").save(path)


def write_rgb(path, size, seed):
    rs = np.random.RandomState(seed)
    arr = rs.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def build_synthetic_ddr(size=64):
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(DDR_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DDR_ROOT, "labels", split), exist_ok=True)
    plan = {"train": (("DDR_tr_A", 1), ("DDR_tr_B", 2), ("DDR_tr_C", 3)),
            "val": (("DDR_va_D", 4),), "test": (("DDR_te_E", 5),)}
    for split, items in plan.items():
        for stem, seed in items:
            write_rgb(os.path.join(DDR_ROOT, "images", split, stem + ".jpg"), size, seed)
            # 四类都要出现，否则 mIoU 会因分母为 0 被剔除
            write_fused_label_png(
                os.path.join(DDR_ROOT, "labels", split, stem + ".png"), size,
                [(1, 4, 20, 4, 20), (2, 20, 36, 4, 20), (3, 4, 20, 20, 36), (4, 20, 36, 20, 36)],
            )


def build_encoder_weights(state_dict, path):
    torch.save({k: v.clone() for k, v in state_dict.items()}, path)
    from trainer.data.common import file_sha256
    return file_sha256(path)


def make_synthetic_u0_config():
    """从真实 U0 配置派生一份可端到端 dry-run 的合成配置（只填必需字段）。"""
    src = os.path.join(REPO_ROOT, "configs", "U0_convnext_unet_ddr_only.yaml")
    cfg = yaml.safe_load(open(src, encoding="utf-8"))

    from trainer.models.convnext_unet import ConvNeXtUNet
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    enc = ConvNeXtUNet(drop_path_rate=0.0)
    wp = os.path.join(WEIGHTS_DIR, "synthetic_encoder.pth")
    sha = build_encoder_weights(enc.encoder.state_dict(), wp)

    st = cfg["structure"]
    st["encoder_weights_source"] = "synthetic://smoke"
    st["encoder_weights_file"] = wp.replace(os.sep, "/")
    st["encoder_weights_sha256"] = sha
    st["library_version"] = f"torch-{torch.__version__}-cpu-smoke"
    cfg["data"]["ddr_root"] = DDR_ROOT.replace(os.sep, "/")
    cfg["data"]["ddr_normalization"] = {"mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0]}
    cfg["data"]["train_crop"] = 64
    dfs = cfg["ddr_finetune_stage"]
    # 批量守恒：effective_batch(8) 必须等于 micro_batch × grad_accum
    dfs["micro_batch"] = 2
    dfs["grad_accum"] = 4
    dfs["gradient_checkpointing"] = False
    dfs["num_workers"] = 0
    dfs["updates"] = 1
    dfs["eval_progress_points"] = [1.0]
    out = os.path.join(CONFIGS_OUT, "U0_smoke.yaml")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def run_py(*argv, timeout=1800):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, *argv], capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=REPO_ROOT, env=env, timeout=timeout,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


# ---------------------------------------------------------------------------
# 用例
# ---------------------------------------------------------------------------

def case_py_compile():
    import py_compile
    pyc_dir = os.path.join(TMP, "pyc")
    os.makedirs(pyc_dir, exist_ok=True)
    targets = [TRAIN_PY]
    for sub in ("trainer", os.path.join("trainer", "data"), os.path.join("trainer", "models")):
        d = os.path.join(REPO_ROOT, sub)
        targets += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".py")]
    targets += [os.path.join(TOOLS_DIR, f) for f in sorted(os.listdir(TOOLS_DIR)) if f.endswith(".py")]
    targets += [os.path.join(REPO_ROOT, "tests", f)
                for f in sorted(os.listdir(os.path.join(REPO_ROOT, "tests"))) if f.endswith(".py")]
    failures = []
    for src in targets:
        cfile = os.path.join(pyc_dir, f"{len(failures)}_{os.path.splitext(os.path.basename(src))[0]}.pyc")
        try:
            py_compile.compile(src, cfile=cfile, doraise=True)
        except py_compile.PyCompileError as e:
            failures.append(f"{os.path.basename(src)}: {e}")
    record(f"T0 py_compile {len(targets)} 个脚本", bool(targets) and not failures,
           "; ".join(failures[:4]))


def case_preflight_core():
    res = run_checks(configs=None, tmp_dir=os.path.join(TMP, "pf"), skip_model_build=True)
    s = res.summary()
    ok = s["fail"] == 0 and s["pass"] >= 10
    dep = s["dependency_blocked"]
    record(f"T1 合成核心自检 FAIL=0（PASS={s['pass']} DEPENDENCY_BLOCKED={dep} "
           f"CALIBRATION_REQUIRED={s['calibration_required']}）", ok,
           json.dumps(res.items[-3:], ensure_ascii=False)[:400])


def _dummy_evaluator(ds, model=None):
    """合成整图 evaluator（formal/calibration 都必须提供，禁止 crop 指标冒充 val）。"""
    from trainer.eval import WholeImageEvaluator
    return WholeImageEvaluator(model or make_fake_model(), ds, device="cpu",
                               mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])


def case_grad_accum_counts():
    """9 个 micro-batch / accum=3 ⇒ 恰 3 次 optimizer update、0 次多余 step。"""
    ds = SyntheticSegDataset(n=8, size=32)
    total_updates, accum, micro = 3, 3, 1     # loader 每 epoch 8 个 micro-batch
    runner = make_runner(make_fake_model(), ds, total_updates=total_updates,
                         grad_accum=accum, micro_batch=micro,
                         out_dir=os.path.join(TMP, "accum"),
                         evaluator=_dummy_evaluator(ds), eval_progress_points=())
    result = runner.run()
    expected_micro = total_updates * accum
    ok = (result["updates_done"] == total_updates
          and runner.state.micro_step == expected_micro
          and len(runner.history) == total_updates)
    record(f"T2 梯度累积按 optimizer update 计数（{runner.state.micro_step} micro / accum={accum} "
           f"⇒ {result['updates_done']} updates）", ok,
           f"micro_step={runner.state.micro_step} updates={result['updates_done']} "
           f"history={len(runner.history)}")


def case_scheduler_by_update():
    """scheduler 按 optimizer update：LR 序列必须等于 lr_multiplier 的手算值。"""
    from trainer.optim import lr_multiplier
    ds = SyntheticSegDataset(n=6, size=32)
    total_updates, warmup = 4, 1
    runner = make_runner(make_fake_model(), ds, total_updates=total_updates, grad_accum=2,
                         micro_batch=2, out_dir=os.path.join(TMP, "sched"),
                         evaluator=_dummy_evaluator(ds), eval_progress_points=())
    runner.cfg["warmup_updates"] = warmup
    runner.run()
    expected = [lr_multiplier(u, total_updates=total_updates, warmup_updates=warmup,
                              cosine_min_ratio=0.01) for u in range(total_updates)]
    got = [round(h[1]["backbone"], 12) for h in runner.history]
    base = runner.base_lrs["backbone"]
    exp_lr = [round(base * m, 12) for m in expected]
    ok = (len(runner.history) == total_updates and got == exp_lr
          and got[0] != got[-1])       # 确实在变化，不是常数假象
    record(f"T3 scheduler 按 update 步进（LR 序列 {got} vs 手算 {exp_lr}）", ok,
           f"got={got} expected={exp_lr}")


def case_resume_continuity():
    """断点恢复后 update 连续（不重放、不跳变），LR 轨迹与不中断一致。"""
    from trainer.checkpoint import load_checkpoint
    from trainer.eval import WholeImageEvaluator
    from trainer.optim import lr_multiplier
    out = os.path.join(TMP, "resume")
    ds = SyntheticSegDataset(n=6, size=32)
    total, warmup = 4, 1
    mk = lambda: _mk_runner_resume(ds, out, total, warmup)
    a = mk()
    r1 = a.run(max_updates=2)
    ck = load_checkpoint(a.checkpoint_paths()["final"])
    upd_at_save = ck["update"]

    b = mk()
    b.resume(a.checkpoint_paths()["final"])
    upd_after_resume = b.state.update
    b.run()

    full = mk()
    full.run()
    # 恢复后只记录新产生的 update（3,4）；其 LR 必须等于不中断跑完的第 3、4 次
    got = [h[1]["backbone"] for h in b.history]
    exp = [h[1]["backbone"] for h in full.history[2:]]
    ok = (upd_at_save == 2 and upd_after_resume == 2 and b.state.update == 4
          and [h[0] for h in b.history] == [3, 4]
          and got == exp)
    record(f"T4 断点恢复 update 连续（save@{upd_at_save} → 恢复点 {upd_after_resume} → "
           f"继续到 {b.state.update}；LR 轨迹与一次性跑完的对应段一致）", ok,
           f"upd_at_save={upd_at_save} resume_at={upd_after_resume} final={b.state.update} "
           f"lr_got={got} lr_expected={exp}")


def _mk_runner_resume(ds, out, total, warmup):
    runner = make_runner(make_fake_model(), ds, total_updates=total, grad_accum=2,
                         micro_batch=2, out_dir=out,
                         evaluator=_dummy_evaluator(ds), eval_progress_points=())
    runner.cfg["warmup_updates"] = warmup
    return runner


def case_checkpoint_fields():
    required = ("format", "arm_id", "family", "stage", "mode", "update", "micro_step",
                "total_updates", "seed", "config_sha256", "data_manifest_sha256",
                "model_state", "optimizer_state", "scheduler_state", "scaler_state", "rng")
    ds = SyntheticSegDataset(n=4, size=32)
    runner = make_runner(make_fake_model(), ds, out_dir=os.path.join(TMP, "ckfields"),
                         evaluator=None)
    p = runner.save(metrics_at_save={"mIoU_lesion4": 0.25}, tag="fields")
    from trainer.checkpoint import load_checkpoint
    ck = load_checkpoint(p)
    missing = [k for k in required if k not in ck]
    sched_ok = ck["scheduler_state"].get("type") == "warmup_cosine_by_update" \
        and ck["scheduler_state"].get("update") == ck["update"]
    record("T5 checkpoint 字段齐全（含 scheduler.update == update，weights_only 安全加载）",
           not missing and sched_ok, f"missing={missing} sched_ok={sched_ok}")


def case_calibration_ckpt_rejected():
    from trainer.checkpoint import load_checkpoint, verify_resume_compatibility
    ds = SyntheticSegDataset(n=4, size=32)
    runner = make_runner(make_fake_model(), ds, mode="calibration",
                         out_dir=os.path.join(TMP, "calibck"))
    p = runner.save(metrics_at_save={}, tag="calib")
    ck = load_checkpoint(p)
    rejected = False
    try:
        verify_resume_compatibility(ck, arm_id="SYNTH", stage="ddr_finetune",
                                    config_sha256="synthetic-config-sha")
    except ValueError as e:
        rejected = "校准产物" in str(e) or "calibration" in str(e)
    # 正例：普通 checkpoint 必须放行
    runner2 = make_runner(make_fake_model(), ds, mode="formal", out_dir=os.path.join(TMP, "calibck"))
    runner2.cfg["is_calibration"] = False
    p2 = runner2.save(metrics_at_save={}, tag="formal")
    ok2 = True
    try:
        verify_resume_compatibility(load_checkpoint(p2), arm_id="SYNTH",
                                    stage="ddr_finetune", config_sha256="synthetic-config-sha")
    except ValueError as e:
        ok2 = False
    record("T6 校准 checkpoint 拒绝作正式恢复源 / 正式 checkpoint 放行", rejected and ok2,
           f"rejected={rejected} formal_accepted={ok2}")


def case_formal_requires_whole_image_evaluator():
    from trainer.eval import WholeImageEvaluator, require_whole_image_evaluator
    from trainer.gate import GateError
    ds = SyntheticSegDataset(n=4, size=32)
    no_eval_raised = False
    try:
        require_whole_image_evaluator(None, mode="formal")
    except GateError:
        no_eval_raised = True
    # engine 层：formal + evaluator=None 必须在跑任何 update 之前就 fail closed
    runner = make_runner(make_fake_model(), ds, mode="formal", total_updates=2,
                         out_dir=os.path.join(TMP, "noeval"), evaluator=None)
    engine_raised = False
    try:
        runner.run()
    except GateError:
        engine_raised = True
    # 反例守卫：传一个"非整图 evaluator"也必须被 formal 拒绝
    class FakeCropEvaluator:
        def run(self):
            return {}

    wrong_type = False
    try:
        require_whole_image_evaluator(FakeCropEvaluator(), mode="formal")
    except GateError:
        wrong_type = True
    # 正例：WholeImageEvaluator 放行
    good = True
    try:
        require_whole_image_evaluator(
            WholeImageEvaluator(make_fake_model(), ds, device="cpu",
                                mean=[0., 0., 0.], std=[1., 1., 1.]), mode="formal")
    except GateError:
        good = False
    ok = (no_eval_raised and engine_raised and wrong_type and good
          and runner.state.update == 0)
    record("T7 formal 模式缺整图 evaluator ⇒ fail closed（未执行任何 update）", ok,
           f"none={no_eval_raised} engine={engine_raised} wrong_type={wrong_type} good={good} "
           f"updates={runner.state.update}")


def case_cli_test_split_rejected(cfg_path):
    rc, out = run_py(TRAIN_PY, "--config", cfg_path, "--stage", "ddr_finetune",
                     "--mode", "calibration", "--train-split", "test", "--dry-run")
    ok = rc == 1 and "test split" in out
    record("T8 CLI 硬拒绝 test 作为训练 split（rc=1）", ok, f"rc={rc} out={out[-400:]}")


def case_cli_formal_placeholders():
    real = os.path.join(REPO_ROOT, "configs", "H0_hrdecoder_ddr_only.yaml")
    rc, out = run_py(TRAIN_PY, "--config", real, "--stage", "ddr_finetune",
                     "--mode", "formal", "--dry-run")
    ok = rc == 1 and "占位符" in out
    record("T9 CLI formal 模式遇占位符 ⇒ rc=1（不放宽正式闸门）", ok, f"rc={rc} out={out[-400:]}")


def case_cli_formal_no_calibration_override(cfg_path):
    rc, out = run_py(TRAIN_PY, "--config", cfg_path, "--stage", "ddr_finetune",
                     "--mode", "formal", "--micro-batch", "1", "--dry-run")
    ok = rc == 1 and "formal 模式禁止" in out
    record("T10 CLI formal 模式禁止 --micro-batch 覆盖 ⇒ rc=1", ok, f"rc={rc} out={out[-300:]}")


def case_metrics_hand_calc():
    from trainer.metrics import SegMetricsAccumulator
    acc = SegMetricsAccumulator(threshold=0.5)
    prob = np.full((1, 4, 2, 2), 0.9, dtype=np.float32)
    tgt = np.zeros((1, 4, 2, 2), dtype=np.float32)
    tgt[:, :, 0, 0] = 1.0
    acc.update(prob, tgt, np.ones_like(tgt))
    ex = acc.compute()["per_class"]["EX"]
    ok1 = (ex["tp"], ex["fp"], ex["fn"]) == (1, 3, 0) and abs(ex["iou"] - 0.25) < 1e-12

    acc2 = SegMetricsAccumulator(threshold=0.5)
    acc2.update(np.full((1, 4, 1, 1), 0.5, np.float32), np.ones((1, 4, 1, 1), np.float32),
                np.ones((1, 4, 1, 1), np.float32))
    e2 = acc2.compute()["per_class"]["EX"]
    ok2 = e2["tp"] == 0 and e2["fn"] == 1              # 严格 p>0.5

    acc3 = SegMetricsAccumulator(threshold=0.5)
    p3 = np.concatenate([np.full((1, 4, 1, 1), .99, np.float32)] * 3 +
                        [np.full((1, 4, 1, 1), .01, np.float32)] * 3, axis=2)
    t3 = np.concatenate([np.ones((1, 4, 1, 1), np.float32)] * 3 +
                        [np.zeros((1, 4, 1, 1), np.float32)] * 3, axis=2)
    acc3.update(p3, t3, np.ones_like(t3))
    ok3 = abs(acc3.compute()["per_class"]["EX"]["aupr"] - 1.0) < 1e-12

    acc4 = SegMetricsAccumulator(threshold=0.5)
    v4 = np.zeros((1, 4, 2, 2), np.float32)
    v4[:, :, 0, 0] = 1.0
    acc4.update(prob, tgt, v4)
    e4 = acc4.compute()["per_class"]["EX"]
    ok4 = (e4["tp"], e4["fp"], e4["fn"]) == (1, 0, 0)
    record("T11 指标手算一致（TP/FP/FN、valid 屏蔽、严格 p>0.5、完美分类器 AUPR=1.0）",
           ok1 and ok2 and ok3 and ok4, f"ok1={ok1} ok2={ok2} ok3={ok3} ok4={ok4}")


def case_valid_masking_grad():
    from trainer.loss import masked_bce_dice_loss
    lg = torch.randn(2, 4, 16, 16, requires_grad=True)
    tg = torch.zeros(2, 4, 16, 16)
    tg[:, 0, :4, :4] = 1.0
    vg = torch.zeros(2, 4, 16, 16)
    vg[:, 0] = 1.0
    loss, parts = masked_bce_dice_loss(lg, tg, vg)
    loss.backward()
    ok1 = float(lg.grad[:, 1:].abs().sum()) == 0.0 and parts["active_classes"] == 1

    lg2 = torch.randn(1, 4, 8, 8, requires_grad=True)
    z = torch.zeros(1, 4, 8, 8)
    loss2, parts2 = masked_bce_dice_loss(lg2, z, z)
    loss2.backward()
    ok2 = float(loss2) == 0.0 and torch.isfinite(loss2) and float(lg2.grad.abs().sum()) == 0.0
    ok3 = parts2["active_classes"] == 0 and not np.isnan(float(loss2))
    record("T12 valid=0 整类零梯度；整批全无效零梯度且无 NaN", ok1 and ok2 and ok3,
           f"ok1={ok1} ok2={ok2} ok3={ok3}")


def case_param_group_exclusivity():
    from trainer.gate import GateError
    from trainer.optim import audit_param_groups, build_optimizer, build_param_groups
    m = make_fake_model()
    pg = build_param_groups(m, backbone_lr=1e-3, decoder_lr=1e-3, weight_decay=0.0,
                            backbone_prefixes=("backbone",))
    audit = audit_param_groups(m, build_optimizer({"optimizer": "adamw"}, pg))
    ok1 = audit["n_trainable_tensors"] == len(pg[0]["params"]) + len(pg[1]["params"])

    # 负例：只给 backbone 组 ⇒ 必须报"有参数未被覆盖"
    missed = False
    try:
        audit_param_groups(m, build_optimizer(
            {"optimizer": "adamw"}, [{"params": pg[0]["params"], "lr": 1e-3, "name": "backbone"}]))
    except GateError:
        missed = True
    # 负例：同一参数出现在两组 ⇒ 必须被拦（Python 侧 torch 自己也会拦，两者都算拦截成功）
    dup = False
    try:
        audit_param_groups(m, build_optimizer({"optimizer": "adamw"}, [
            {"params": pg[0]["params"], "lr": 1e-3, "name": "a"},
            {"params": pg[0]["params"] + pg[1]["params"], "lr": 1e-3, "name": "b"}]))
    except (GateError, ValueError) as e:
        dup = "more than one parameter group" in str(e) or isinstance(e, GateError)
    # 负例：冻结参数被放进组 ⇒ 必须被拦
    frozen_rejected = False
    try:
        m2 = make_fake_model()
        for p in m2.backbone.parameters():
            p.requires_grad = False
        audit_param_groups(m2, build_optimizer({"optimizer": "adamw"}, [
            {"params": list(m2.backbone.parameters()) + list(m2.head.parameters()),
             "lr": 1e-3, "name": "all"}]))
    except GateError:
        frozen_rejected = True
    # 负例：前缀互斥（"encoder" 不应吞掉 "encoder2"）
    from trainer.optim import classify_parameter
    excl = (classify_parameter("encoder.weight", ("encoder",)) == "backbone"
            and classify_parameter("encoder2.weight", ("encoder",)) == "decoder")
    record("T13 参数组互斥且完整（漏组/重复持有/冻结参数入组/前缀误并 四个负例）",
           ok1 and missed and dup and frozen_rejected and excl,
           f"ok1={ok1} missed={missed} dup={dup} frozen={frozen_rejected} "
           f"prefix_exclusive={excl}")


def case_real_u_model_contract():
    from trainer.config import load_yaml
    from trainer.gate import GateError
    from trainer.models import build_model
    cfg = load_yaml(make_synthetic_u0_config())
    model, meta = build_model(cfg, stage="ddr_finetune", gradient_checkpointing=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    shape_ok = tuple(out.shape) == (1, 4, 64, 64)
    bad_rejected = False
    try:
        model(torch.zeros(1, 3, 48, 48))
    except (GateError, RuntimeError, ValueError):
        bad_rejected = True
    detail_true_rejected = False
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["structure"]["detail_branch"] = True
    try:
        from trainer.models.convnext_unet import build_convnext_unet
        build_convnext_unet(bad_cfg)
    except GateError:
        detail_true_rejected = True
    record("T14 U 臂真实模型：严格 N×4×H×W + 非 32 倍数拒绝 + detail_branch=true 拒绝",
           shape_ok and bad_rejected and detail_true_rejected,
           f"shape={tuple(out.shape)} bad_rejected={bad_rejected} "
           f"detail_true_rejected={detail_true_rejected}")


def case_cli_end_to_end_dry_run(cfg_path):
    out_dir = os.path.join(TMP, "cli_run")
    rc, out = run_py(TRAIN_PY, "--config", cfg_path, "--stage", "ddr_finetune",
                     "--mode", "calibration", "--out-dir", out_dir, "--dry-run",
                     timeout=2400)
    ok = rc == 0 and "dry-run forward/backward OK" in out and "未写任何 checkpoint" in out
    produced = os.path.isfile(os.path.join(out_dir, "ckpt_final.pth"))
    ok = ok and not produced
    record("T15 train.py 端到端 dry-run（合成 U0 + 合成 DDR 树，calibration）", ok,
           f"rc={rc} ckpt_written={produced} out={out[-600:]}")


def case_hr_arm_dependency_status():
    """H 臂真实依赖状态：可导入则记录 OK；不可导入则 DEPENDENCY_BLOCKED（不计 PASS）。"""
    from trainer.config import load_yaml
    from trainer.models import probe_family
    cfg = load_yaml(os.path.join(REPO_ROOT, "configs", "H0_hrdecoder_ddr_only.yaml"))
    rep = probe_family("hrdecoder", cfg)
    if rep["status"] == "OK":
        record("T16 H 臂旧仓库可导入（本机环境满足；结构未用假模型冒充）", True,
               "旧仓库可导入，H 臂结构可用真实构建验证")
    else:
        _results.append(("T16 H 臂真实依赖状态（不使用假模型冒充）", True,
                         f"status={rep['status']}: {str(rep['detail'])[:260]}",
                         "DEPENDENCY_BLOCKED"))
        print(f"  [DEPENDENCY_BLOCKED] T16 H 臂真实依赖状态（不使用假模型冒充） — "
              f"status={rep['status']}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 78)
    print("[run_training_smoke] 训练链路合成冒烟（离线 / CPU / 合成 fake model+data）")
    print("=" * 78)

    if os.path.isdir(TMP):
        shutil.rmtree(TMP)
    for d in (CONFIGS_OUT, DDR_ROOT, WEIGHTS_DIR, REPORTS):
        os.makedirs(d, exist_ok=True)

    print("\n-- T0 编译 --")
    case_py_compile()

    print("\n-- T1 公共核心自检 --")
    case_preflight_core()

    print("\n-- T2–T7 训练状态机 / checkpoint / 闸门 --")
    build_synthetic_ddr()
    cfg_path = make_synthetic_u0_config()
    case_grad_accum_counts()
    case_scheduler_by_update()
    case_resume_continuity()
    case_checkpoint_fields()
    case_calibration_ckpt_rejected()
    case_formal_requires_whole_image_evaluator()

    print("\n-- T8–T10 CLI 闸门（子进程） --")
    case_cli_test_split_rejected(cfg_path)
    case_cli_formal_placeholders()
    case_cli_formal_no_calibration_override(cfg_path)

    print("\n-- T11–T13 指标 / loss / 参数组 --")
    case_metrics_hand_calc()
    case_valid_masking_grad()
    case_param_group_exclusivity()

    print("\n-- T14–T16 真实模型与端到端 --")
    case_real_u_model_contract()
    case_cli_end_to_end_dry_run(cfg_path)
    case_hr_arm_dependency_status()

    print("\n" + "=" * 78)
    failed = [(c, d) for c, ok, d, st in _results if not ok]
    blocked = [(c, d) for c, ok, d, st in _results if st == "DEPENDENCY_BLOCKED"]
    for case, ok, detail, status in _results:
        tag = "PASS" if ok else "FAIL"
        if status == "DEPENDENCY_BLOCKED":
            tag = "DEPENDENCY_BLOCKED"
        print(f"  [{tag}] {case}")
        if detail and (not ok or status == "DEPENDENCY_BLOCKED"):
            print(f"         {detail}")
    print("-" * 78)
    n_pass = sum(1 for c, ok, d, st in _results if ok and st != "DEPENDENCY_BLOCKED")
    print(f"[run_training_smoke] {n_pass}/{len(_results)} PASS"
          f"（DEPENDENCY_BLOCKED={len(blocked)}，不计 PASS）；资产: {TMP}")
    if failed:
        print(f"[run_training_smoke] FAIL: {len(failed)} 个用例失败")
        return 1
    print("[run_training_smoke] PASS: 公共训练核心行为全部符合预期")
    return 0


if __name__ == "__main__":
    sys.exit(main())
