#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J0/J1 接口预检（契约 §10.1 / §10.3 的准备步骤，不是实验本体）。

目的：在 U0 结束前把 J0/J1 的**真实解冻接口**验证清楚，避免到时 GPU 再空转。
- 从代码定位 HRNet 的**真实 stage 模块名**（不猜前缀）
- 构造 J0（只训 student decoder）与 J1（decoder + HRNet 最后 stage）
- 提交两臂 trainable 名称集合与 optimizer ID 集合的差异
- 跑**一轮真实更新**，验证 J1 最后 stage 确实有参数更新、未授权参数与 BN buffers 不变

两臂共享同一 D student 起点；reference B、其他集成成员、后融合头均冻结。
"""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO = "/root/autodl-tmp/hrplus/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)
D_CK = "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth"


def fp(mod, exclude=()):
    h = hashlib.sha256()
    for n, p in sorted(mod.named_parameters()):
        if any(n.startswith(e) for e in exclude):
            continue
        h.update(n.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def bfp(mod):
    """BN buffers（含 running_mean/var/num_batches_tracked）的指纹。"""
    h = hashlib.sha256()
    for n, b in sorted(mod.named_buffers()):
        h.update(n.encode())
        h.update(b.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="configs/lesion/exp/hrplus_patch_dice.py")
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(args.config)
    model = build_segmentor(cfg.model)
    load_checkpoint(model, D_CK, map_location="cpu", strict=False)
    model = model.cuda()

    # ---------- 1) 定位真实模块名 ----------
    backbone_prefixes = sorted({n for n, _ in model.backbone.named_children()})
    stage_like = sorted({n for n, _ in model.backbone.named_modules()
                         if n and n.count(".") <= 1})
    dec_children = sorted({n for n, _ in model.decode_head.named_children()})
    rep = {"config": args.config, "d_ckpt": D_CK,
           "backbone_top_children": backbone_prefixes,
           "backbone_modules_depth<=1": stage_like,
           "decode_head_children": dec_children,
           "top_children_of_segmentor": sorted({n for n, _ in model.named_children()})}
    print("backbone 顶层子模块:", backbone_prefixes, flush=True)
    print("decode_head 子模块:", dec_children, flush=True)
    print("segmentor 顶层:", rep["top_children_of_segmentor"], flush=True)

    # HRNet 的 stage 命名**不统一**：本仓库实际是 layer1 + stage2 + stage3 + stage4。
    # 契约 §10.1 要求"从代码定位并列出，不猜前缀" → 显式枚举 stage1..4 / layer1..4 后按数字排序。
    import re as _re
    ordered = sorted({n for n in backbone_prefixes
                      if _re.fullmatch(r'(layer|stage)\d+', n)},
                     key=lambda x: int(_re.search(r'\d+', x).group()))
    assert ordered, '未能定位 HRNet 的 stage 序列，需人工确认'
    last_stage = ordered[-1]
    rep["hrnet_stage_sequence"] = ordered
    rep["j1_last_stage_module"] = last_stage
    print("→ HRNet stage 序列（按代码实际）= %s" % ordered, flush=True)
    print("→ J1 解冻的最后 stage = %s" % last_stage, flush=True)

    def build_arm(include_backbone_stage):
        m = build_segmentor(copy.deepcopy(cfg.model))
        load_checkpoint(m, D_CK, map_location="cpu", strict=False)
        m = m.cuda()
        for p in m.parameters():
            p.requires_grad_(False)
        for p in m.decode_head.parameters():
            p.requires_grad_(True)                      # J0/J1 共同：decoder
        if include_backbone_stage:
            for nm, p in m.backbone.named_parameters():
                if nm.split(".")[0] == last_stage:
                    p.requires_grad_(True)              # J1 额外：最后 stage
        # BN 运行统计始终冻结
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eval()
        return m

    m0 = build_arm(False).train()
    m1 = build_arm(True).train()

    def tr(m):
        return sorted(n for n, p in m.named_parameters() if p.requires_grad)

    t0, t1 = tr(m0), tr(m1)
    rep["J0_trainable"] = {"n": len(t0), "sample": t0[:8],
                           "all_in_decode_head": all(x.startswith("decode_head.") for x in t0)}
    rep["J1_trainable"] = {"n": len(t1), "sample": t1[:8],
                           "extra_vs_J0": sorted(set(t1) - set(t0))[:8],
                           "n_extra": len(set(t1) - set(t0)),
                           "all_expected": all(x.startswith("decode_head.") or
                                               x.split(".")[0] == "backbone" and
                                               x.split(".")[1] == last_stage for x in t1)}
    print("J0 trainable = %d (全在 decode_head: %s)" % (len(t0), rep["J0_trainable"]["all_in_decode_head"]), flush=True)
    print("J1 trainable = %d，比 J0 多 %d 个（%s…）"
          % (len(t1), rep["J1_trainable"]["n_extra"], rep["J1_trainable"]["extra_vs_J0"][:3]), flush=True)

    opt_params = {
        "J0": [p for p in m0.parameters() if p.requires_grad],
        "J1": [p for p in m1.parameters() if p.requires_grad],
    }
    opt = {k: torch.optim.AdamW(v, lr=1e-4) for k, v in opt_params.items()}
    rep["optimizer_param_counts"] = {k: len(v) for k, v in opt_params.items()}
    rep["optimizer_ids_subset_of_trainable"] = {
        "J0": {id(p) for p in opt_params["J0"]} <= {id(p) for p in m0.parameters() if p.requires_grad},
        "J1": {id(p) for p in opt_params["J1"]} <= {id(p) for p in m1.parameters() if p.requires_grad},
    }

    # ---------- 2) 一轮真实更新 ----------
    from mmseg.datasets import build_dataset
    d = copy.deepcopy(cfg.data["train"]); d.test_mode = False
    ds = build_dataset(d)
    s = ds[0]

    def unwrap(x):
        n = 0
        while hasattr(x, "data") and n < 8:
            x = x.data; n += 1
        return x

    img = unwrap(s["img"]).unsqueeze(0).cuda()
    _im = unwrap(s["img_metas"])
    meta = [_im] if isinstance(_im, dict) else [unwrap(e) for e in _im]
    gt = unwrap(s["gt_semantic_seg"]).cuda()
    while gt.dim() < 4:
        gt = gt.unsqueeze(0)

    upd = {}
    for tag, m in (("J0", m0), ("J1", m1)):
        before = fp(m)
        before_bn = bfp(m)
        before_stage = {n: p.detach().clone() for n, p in m.backbone.named_parameters()
                        if n.split(".")[0] == last_stage}
        losses = m.forward_train(img, meta, gt)
        loss = sum(losses[k] for k in losses if "loss" in k)
        opt[tag].zero_grad(set_to_none=True)
        loss.backward()
        g_stage = sum(float(p.grad.abs().sum()) for n, p in m.backbone.named_parameters()
                      if n.split(".")[0] == last_stage and p.grad is not None)
        g_base_other = sum(float(p.grad.abs().sum()) for n, p in m.backbone.named_parameters()
                           if n.split(".")[0] != last_stage and p.grad is not None)
        opt[tag].step()
        after_bn = bfp(m)
        changed_stage = [n for n, p in m.backbone.named_parameters()
                         if n.split(".")[0] == last_stage
                         and not torch.equal(p.detach(), before_stage[n])]
        upd[tag] = {"loss": float(loss), "finite": bool(torch.isfinite(loss)),
                    "grad_last_stage": g_stage, "grad_backbone_other": g_base_other,
                    "n_last_stage_params_changed": len(changed_stage),
                    "bn_buffers_unchanged": before_bn == after_bn,
                    "params_changed_overall": before != fp(m)}
        print("[%s] loss=%.4f 末stage梯度=%.4f 其他backbone梯度=%.6f 末stage已变参数=%d BN buffers不变=%s"
              % (tag, loss, g_stage, g_base_other, len(changed_stage), before_bn == after_bn), flush=True)

    rep["one_real_update"] = upd
    rep["verdict"] = {
        "J0_decoder_only_trains": bool(upd["J0"]["params_changed_overall"]
                                       and upd["J0"]["n_last_stage_params_changed"] == 0),
        "J1_last_stage_trains": bool(upd["J1"]["n_last_stage_params_changed"] > 0),
        "J1_earlier_stages_untouched": bool(upd["J1"]["grad_backbone_other"] == 0.0),
        "BN_buffers_frozen_both": bool(upd["J0"]["bn_buffers_unchanged"] and upd["J1"]["bn_buffers_unchanged"]),
        "trainable_sets_differ": bool(set(t1) - set(t0)),
    }
    rep["status"] = "PASS" if all(rep["verdict"].values()) else "FAIL"
    print("\n判定:", rep["verdict"], flush=True)
    print("STATUS =", rep["status"], flush=True)

    p = os.path.join(args.out_dir, "experiments", "J0J1_interface_preflight.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("写出:", p)


if __name__ == "__main__":
    main()
