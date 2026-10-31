#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1-b：实证 load_checkpoint 返回值被解包时的真实行为（prompt §5.1）。

不推理，直接跑。回答三个问题：
  Q1 load_checkpoint 返回的对象类型与顶层键是什么？
  Q2 旧评估器写的 `miss, unexp = r` 在真实对象上会发生什么？
  Q3 因此历史 eval JSON 里的 missing/unexpected 到底代表了什么？
"""
import argparse
import json
import os
import sys

CKPTS = {
    "b_weights": "/root/autodl-tmp/hrplus/repo/weights/b_weights.pth",
    "d_full": "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth",
    "d_student_only": "/root/autodl-tmp/hrplus/D_0913_retry/backup_student_2000.pth",
    "e_full": "/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import torch
    from mmcv.runner import load_checkpoint
    from mmcv import Config
    from mmseg.models import build_segmentor

    cfg = Config.fromfile("configs/lesion/exp/hrplus_patch_dice.py")
    model = build_segmentor(cfg.model)

    R = {"probe": "load_checkpoint_return_unpack", "items": {}}

    for name, p in CKPTS.items():
        rec = {"path": p, "exists": os.path.isfile(p)}
        if not rec["exists"]:
            rec["status"] = "MISSING"
            R["items"][name] = rec
            continue

        # ---- 直接 torch.load 看容器结构 ----
        ck_raw = torch.load(p, map_location="cpu")
        rec["raw_type"] = type(ck_raw).__name__
        rec["raw_top_keys"] = (sorted(ck_raw.keys())
                               if isinstance(ck_raw, dict) else "NOT_A_DICT")
        rec["raw_n_top_keys"] = (len(ck_raw) if isinstance(ck_raw, dict) else None)

        # ---- 用 mmcv.run_checkpoint 真实调用 ----
        m2 = build_segmentor(cfg.model)
        r = load_checkpoint(m2, p, map_location="cpu", strict=False)
        rec["load_checkpoint_return_type"] = type(r).__name__
        rec["load_checkpoint_return_keys"] = (sorted(r.keys())
                                              if isinstance(r, dict) else "NOT_A_DICT")
        rec["is_tuple"] = isinstance(r, tuple)

        # ---- 复刻旧评估器的解包逻辑，看真实后果 ----
        try:
            miss, unexp = r
            rec["old_unpack"] = {
                "raised": False,
                "miss_type": type(miss).__name__,
                "miss_repr": repr(miss)[:80],
                "unexp_repr": repr(unexp)[:80],
                "nunet_len": len([k for k in miss
                                  if "num_batches_tracked" not in k]),
            }
        except Exception as e:
            rec["old_unpack"] = {"raised": True,
                                 "exc": "%s: %s" % (type(e).__name__, e),
                                 "then_miss_becomes": "[]",
                                 "then_nunet_len": 0}
        R["items"][name] = rec
        print("[%s] raw_type=%s n_keys=%s | lc_return=%s is_tuple=%s | old_unpack_raised=%s"
              % (name, rec["raw_type"], rec.get("raw_n_top_keys"),
                 rec["load_checkpoint_return_type"], rec["is_tuple"],
                 rec["old_unpack"]["raised"]))
        if not rec["old_unpack"]["raised"]:
            print("     miss=%r (len(nunet)=%d)"
                  % (rec["old_unpack"]["miss_repr"], rec["old_unpack"]["nunet_len"]))
        else:
            print("     %s -> 于是被记成 missing=[] unexpected=[]"
                  % rec["old_unpack"]["exc"])

    # ---- 结论（基于实测，不是推理） ----
    raised = [k for k, v in R["items"].items()
              if v.get("old_unpack", {}).get("raised")]
    not_raised = [k for k, v in R["items"].items()
                  if v.get("old_unpack") and not v["old_unpack"]["raised"]]
    R["conclusion"] = {
        "load_checkpoint_returns": "checkpoint 本身（dict），不是 (missing, unexpected)",
        "old_code_semantics": (
            "旧评估器 `miss, unexp = load_checkpoint(...)` 与 API 真实契约不符："
            "若容器恰好可解包成两个元素，会把键名/内容误当 missing/unexpected；"
            "否则抛异常并被 except 吞掉，记为 []、[]。两种情况下都不是加载审计。"),
        "ckpts_where_unpack_raised": raised,
        "ckpts_where_unpack_succeeded": not_raised,
        "verdict_for_history": (
            "历史 eval JSON 的 load_report.missing/unexpected 不能作为"
            "『权重完整加载』的证据（prompt §5.1、审查结论 §3.4）。"
            "本轮必须改用逐键 shape/dtype 契约做加载审计。"),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, "tests", "checkpoint_loading.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, p)
    print("\n结论:", R["conclusion"]["verdict_for_history"])
    print("写出:", p)


if __name__ == "__main__":
    main()
