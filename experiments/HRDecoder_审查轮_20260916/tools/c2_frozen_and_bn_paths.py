#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 多时点冻结快照（prompt §6.1/§6.2）+ §9.3 BN 路径验证。

C2: 对 HRPlusMAOnBase 的**全部非 ma_refiner.* 状态**（parameters + buffers，
含 BN running_mean/var 与 num_batches_tracked）在 7 个时点做逐 tensor 精确快照：
  S1 构造+加载后  S2 model.train()后  S3 forward后  S4 backward后
  S5 真实 optimizer.step()后  S6 连续5步后  S7 保存/重建/恢复后
用真实训练配置的优化器与 grad_clip。

§9.3: 用 forward hook 给每个 BN 注册访问计数，验证 BN 重估时
       reference 分支是否真的被访问（旧 bn_changed_count 只是数量不是差值）。
"""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np
import torch

REPO = "/root/autodl-tmp/hrplus/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)
import ma_residual  # noqa
from mmcv import Config
from mmcv.runner import build_optimizer
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor


def unwrap(x):
    n = 0
    while hasattr(x, "data") and n < 8:
        x = x.data
        n += 1
    return x


def snapshot(module, exclude_prefix="ma_refiner."):
    """逐 tensor 快照：name -> (shape, dtype, 内容哈希)。覆盖 parameters + buffers。"""
    snap = {}
    for kind, items in (("param", module.named_parameters()),
                        ("buffer", module.named_buffers())):
        for name, t in items:
            if name.startswith(exclude_prefix):
                continue
            snap[kind + ":" + name] = {
                "shape": list(t.shape), "dtype": str(t.dtype),
                "sum": float(t.detach().float().abs().sum()),
                "h": hashlib.sha256(
                    t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16],
            }
    return snap


def diff(a, b):
    ch = [k for k in a if a[k]["h"] != b[k]["h"]]
    return {"n_changed": len(ch), "changed": ch[:20], "identical": len(ch) == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "tests"), exist_ok=True)

    cfg = Config.fromfile(args.config)
    model = build_segmentor(cfg.model).cuda()
    print("[S1] 构造+加载完成；可训练参数 =",
          [n for n, p in model.named_parameters() if p.requires_grad][:5], "…")
    S1 = snapshot(model)

    # 真实优化器（用配置里的，而不是简化 SGD）
    opt_cfg = copy.deepcopy(cfg.optimizer)
    optimizer = build_optimizer(model, opt_cfg)
    opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    opt_check = {
        "optimizer_param_groups": len(optimizer.param_groups),
        "optimizer_param_count": sum(len(g["params"]) for g in optimizer.param_groups),
        "opt_ids_subset_of_trainable": opt_param_ids <= trainable_ids,
        "trainable_param_count": len(trainable_ids),
        "frozen_in_optimizer": len(opt_param_ids - trainable_ids),
    }
    print("[opt]", opt_check)

    # 数据（train-mode，含 gt）
    dtr = copy.deepcopy(cfg.data["train"]); dtr.test_mode = False
    ds = build_dataset(dtr)
    samples = [ds[i] for i in range(min(args.steps + 1, len(ds)))]

    def batch(i):
        s = samples[i]
        img = unwrap(s["img"]).unsqueeze(0).cuda()
        _im = unwrap(s["img_metas"])
        meta = [_im] if isinstance(_im, dict) else [
            (e if isinstance(e, dict) else unwrap(e)) for e in _im]
        gt = unwrap(s["gt_semantic_seg"]).cuda()
        while gt.dim() < 4:
            gt = gt.unsqueeze(0)
        return img, meta, gt

    model.train()
    S2 = snapshot(model)
    print("[S2] model.train() 后 与 S1 差异:", diff(S1, S2)["n_changed"],
          "（BN 若被 train() 打开应在此出现）")

    img, meta, gt = batch(0)
    losses = model.forward_train(img, meta, gt)
    names = sorted(k for k in losses if "loss" in k)
    loss = sum(losses[k] for k in names)
    finite = bool(torch.isfinite(loss))
    loss.backward()
    S3 = snapshot(model)   # forward 后（未 backward 前 S2 与 S3 之间无参数变化）
    S4 = snapshot(model)   # backward 后
    grad_head = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                    if n.startswith("ma_refiner.") and p.grad is not None)
    grad_base = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                    if not n.startswith("ma_refiner.") and p.grad is not None)
    print("[S3/S4] loss项=%s finite=%s 头梯度=%.4f 主干梯度=%.6f"
          % (names, finite, grad_head, grad_base))

    optimizer.step()
    S5 = snapshot(model)
    d15 = diff(S1, S5)
    print("[S5] 一次真实 step 后：变更=%d（必须全部是 ma_refiner.*，"
          "但快照已排除头，所以应为 0）" % d15["n_changed"])
    if d15["n_changed"]:
        print("   !! 非头状态变化:", d15["changed"][:10])

    for i in range(1, args.steps):
        im, mt, g = batch(i)
        model.zero_grad(set_to_none=True)
        ls = model.forward_train(im, mt, g)
        sum(ls[k] for k in sorted(k for k in ls if "loss" in k)).backward()
        optimizer.step()
    S6 = snapshot(model)
    d16 = diff(S1, S6)
    print("[S6] %d 步后：非头状态变更=%d（应为 0）" % (args.steps, d16["n_changed"]))
    if d16["n_changed"]:
        print("   !! 变化:", d16["changed"][:10])

    # S7 保存/重建/恢复
    p = os.path.join(args.out_dir, "tests", "c2_roundtrip.pth")
    torch.save({"state_dict": model.state_dict()}, p)
    m2 = build_segmentor(cfg.model).cuda()
    m2.load_state_dict(torch.load(p, map_location="cpu")["state_dict"], strict=True)
    S7 = snapshot(m2)
    d17 = diff(S1, S7)
    print("[S7] 保存/重建/恢复后：与 S1 差异=%d（含头差异应为 17 项 ma_refiner，"
          "非头应为 0）" % d17["n_changed"])
    if d17["n_changed"]:
        print("   !! 非头变化:", d17["changed"][:10])

    # 恢复后输出一致性（同一样本）
    model.eval(); m2.eval()
    with torch.no_grad():
        o1 = model(return_loss=False, img=[img], img_metas=[meta])
        o2 = m2(return_loss=False, img=[img], img_metas=[meta])
    import numpy as np
    def _p(o):
        e = o[0]
        return np.asarray(e[0] if isinstance(e, tuple) else e, dtype=np.float64)
    d_out = float(np.abs(_p(o1) - _p(o2)).max())
    print("[S7] 恢复后输出 max diff =", d_out)

    out = {
        "S1_S2_diff": diff(S1, S2),
        "S1_S5_diff": d15, "S1_S6_diff": d16, "S1_S7_diff": d17,
        "S7_output_max_diff": d_out,
        "optimizer_check": opt_check,
        "losses": {k: float(v) for k, v in losses.items()},
        "loss_finite": finite,
        "grad_head_sum": grad_head, "grad_base_sum": grad_base,
        "status": ("PASS" if (d15["identical"] and d16["identical"]
                              and finite and grad_base == 0.0
                              and grad_head > 0 and d_out < 1e-6) else "FAIL"),
    }
    p2 = os.path.join(args.out_dir, "tests", "frozen_state_steps.json")
    tmp = p2 + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, p2)
    print("\nC2 状态:", out["status"], " 写出:", p2)

    # ---------------- §9.3 BN 路径验证 ----------------
    print("\n== §9.3 BN 路径验证 ==")
    visits = {}
    hooks = []
    for name, m in model.named_modules():
        if type(m).__name__.endswith("BatchNorm2d") or "BatchNorm" in type(m).__name__:
            visits[name] = {"n": 0}
            hooks.append(m.register_forward_hook(
                (lambda nm: (lambda mod, i, o: visits[nm].__setitem__(
                    "n", visits[nm]["n"] + 1)))(name)))
    # 模拟旧 BN 重估的访问路径：eval + 只把 BN 设 train + encode_decode
    model.eval()
    for n, m in model.named_modules():
        if "BatchNorm" in type(m).__name__:
            m.train()
    for n, m in model.named_modules():
        if "BatchNorm" in type(m).__name__:
            visits[n]["before_reset_trainable"] = m.training
    for name, m in model.named_modules():
        if "BatchNorm" in type(m).__name__:
            m.reset_running_stats(); m.momentum = None
    with torch.no_grad():
        model.encode_decode(img, meta)
    for h in hooks:
        h.remove()
    by_scope = {"student": 0, "reference": 0}
    for n, v in visits.items():
        scope = "reference" if n.startswith("reference.") else "student"
        by_scope[scope] += v["n"]
    res = {
        "bn_visit_counts_total": sum(v["n"] for v in visits.values()),
        "by_scope": by_scope,
        "student_bn_modules": sum(1 for n in visits if not n.startswith("reference.")),
        "reference_bn_modules": sum(1 for n in visits if n.startswith("reference.")),
        "verdict": ("reference 分支未被访问" if by_scope["reference"] == 0
                    else "reference 分支被访问 %d 次" % by_scope["reference"]),
        "implication": ("旧重估只调 encode_decode；若 reference BN 被选中重估却未被访问，"
                        "则该干预不成立（prompt §9.3）"),
    }
    print("  student BN 访问 =", by_scope["student"], " 模块数 =", res["student_bn_modules"])
    print("  reference BN 访问 =", by_scope["reference"], " 模块数 =", res["reference_bn_modules"])
    print("  判定:", res["verdict"])
    p3 = os.path.join(args.out_dir, "diagnostics", "branch_visit_counts.json")
    tmp = p3 + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p3)
    print("写出:", p3)


if __name__ == "__main__":
    main()
