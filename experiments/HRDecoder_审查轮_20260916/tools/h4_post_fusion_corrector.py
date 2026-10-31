#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H4/H6：在最终融合之后修正某个类别通道（prompt L4）。--cls 选择 EX/HE/SE/MA。

特征 7 通道：RGB(3) + logit(P_D_MA) + logit(P_s_MA) + logit(P_r_MA)
            + (P_s_MA - P_r_MA)
头：Conv3x3(7->16) -> ReLU -> Conv3x3(16->16) -> ReLU -> Conv1x1(16->1)，输出层零初始化
delta = 6 * tanh(head(features))
P_new_MA = sigmoid(logit(clamp(P_D_MA)) + delta)
损失：0.5*BCEWithLogits + 0.5*Dice（仅 MA 通道）
优化器：Adam 1e-3，只传头参数（prompt L8：不统一限 2000 步，2000 只是观察点）
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "/root/autodl-tmp/hrplus/repo"
os.chdir(REPO)
sys.path.insert(0, REPO)


def unwrap(x):
    n = 0
    while hasattr(x, "data") and n < 8:
        x = x.data
        n += 1
    return x


def as_img(x):
    x = unwrap(x)
    while isinstance(x, (list, tuple)):
        x = unwrap(x[0])
    return x


def as_meta(x):
    x = unwrap(x)
    if isinstance(x, dict):
        return [x]
    out = []
    for e in x:
        e = unwrap(e)
        out.extend([e] if isinstance(e, dict) else [unwrap(e)])
    return out


def logit(p, eps=1e-4):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def soft_dice(p, g, eps=1.0):
    num = 2 * (p * g).sum()
    den = (p * p).sum() + (g * g).sum() + eps
    return 1 - num / den


class MACorrectorHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat):
        return 6.0 * torch.tanh(self.net(feat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--cls", default="MA", choices=["EX","HE","SE","MA"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bce-w", type=float, default=0.5)
    ap.add_argument("--dice-w", type=float, default=0.5)
    args = ap.parse_args()

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    CIDX = {'EX': 0, 'HE': 1, 'SE': 2, 'MA': 3}[args.cls]
    LBL = CIDX + 1
    cfg = Config.fromfile(args.config)
    dtr = copy.deepcopy(cfg.data["train"]); dtr.test_mode = False
    ds_tr = build_dataset(dtr)
    dva = copy.deepcopy(cfg.data["val"]); dva.test_mode = True
    ds_va = build_dataset(dva)
    gt_va = ds_va.get_gt_seg_maps()
    print("train=%d val=%d" % (len(ds_tr), len(ds_va)), flush=True)

    model = build_segmentor(cfg.model)
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    head = MACorrectorHead().cuda()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.PolyLR(opt, power=1.0,
                                            max_iters=args.iters) if False else None
    # 简单 poly 衰减
    base_lr = args.lr

    # ---- 验证函数：val 上 MA IoU（student-only / fused / fused+head）----
    def val_eval():
        agg = {"student": [0, 0, 0], "fused": [0, 0, 0], "corrected": [0, 0, 0]}
        with torch.no_grad():
            for i in range(len(ds_va)):
                s = ds_va[i]
                img = as_img(s["img"]).unsqueeze(0).cuda()
                meta = as_meta(s["img_metas"])
                gt = np.asarray(gt_va[i])
                ori = meta[0]["ori_shape"][:2]
                if gt.shape != tuple(ori):
                    import cv2
                    gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                                    interpolation=cv2.INTER_NEAREST)
                g = torch.from_numpy(gt).cuda() == LBL
                ig = torch.from_numpy(gt).cuda() == 255
                ps = model._student_prob(img, meta, True)[0]      # (4,H,W) prob
                pr = model._reference_prob(img, meta, True)[0]
                pd = 0.5 * ps + 0.5 * pr
                # 特征
                img_f = torch.nn.functional.interpolate(
                    img, size=tuple(ori), mode='bilinear', align_corners=False)
                f7 = torch.cat([img_f[0],
                                logit(pd[CIDX:CIDX+1]), logit(ps[CIDX:CIDX+1]), logit(pr[CIDX:CIDX+1]),
                                (ps[CIDX:CIDX+1] - pr[CIDX:CIDX+1])], dim=0).unsqueeze(0)
                delta = head(f7)[0, 0]
                new_logit = logit(pd[CIDX:CIDX+1][0]) + delta
                pnew = torch.sigmoid(new_logit)
                for tag, pr_ma in (("student", ps[CIDX]), ("fused", pd[CIDX]),
                                   ("corrected", pnew)):
                    p = pr_ma > 0.5
                    agg[tag][0] += int((p & g & ~ig).sum())
                    agg[tag][1] += int((p & ~g & ~ig).sum())
                    agg[tag][2] += int((~p & g).sum())
        res = {}
        for k, (tp, fp, fn) in agg.items():
            den = tp + fp + fn
            res[k] = tp / den if den else 0.0
        return res

    print("== 零步基线（val MA IoU）==", flush=True)
    base = val_eval()
    print("  student=%.4f fused=%.4f corrected=%.4f" % (
        base["student"], base["fused"], base["corrected"]), flush=True)

    hist = [{"step": 0, **base}]
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    best = {"step": 0, "corrected": base["corrected"]}
    it = 0
    while it < args.iters:
        s = ds_tr[rng.randint(0, len(ds_tr))]
        img = as_img(s["img"]).unsqueeze(0).cuda()
        meta = as_meta(s["img_metas"])
        gt = unwrap(s["gt_semantic_seg"]).cuda()
        while gt.dim() < 4:
            gt = gt.unsqueeze(0)
        gt_ma = (gt.squeeze(1) == 4).float()

        with torch.no_grad():
            logit_s = model.encode_decode(img, meta)          # (1,4,1024,1024)
            ps = torch.sigmoid(logit_s)
            pr = model.reference.inference(img, meta, rescale=False)
            if isinstance(pr, tuple):
                pr = pr[0]
        pd = 0.5 * ps + 0.5 * pr
        f7 = torch.cat([img, logit(pd[:, CIDX:CIDX+1]), logit(ps[:, CIDX:CIDX+1]),
                        logit(pr[:, CIDX:CIDX+1]), (ps[:, CIDX:CIDX+1] - pr[:, CIDX:CIDX+1])], dim=1)
        delta = head(f7)
        base_logit = logit(pd[:, CIDX:CIDX+1])
        new_logit = base_logit + delta
        p_new = torch.sigmoid(new_logit)

        tgt = torch.where(gt == 255, torch.full_like(gt, 0), gt)
        y = (tgt.squeeze(1) == LBL).float().unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(new_logit, y)
        dice = soft_dice(p_new, y)
        loss = args.bce_w * bce + args.dice_w * dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        # poly 衰减
        lr = base_lr * (1 - it / args.iters)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        it += 1

        if it % args.eval_every == 0 or it == args.iters:
            head.eval(); model.eval()
            r = val_eval()
            hist.append({"step": it, **r,
                         "loss": float(loss), "bce": float(bce), "dice": float(dice),
                         "elapsed": round(time.time() - t0, 1)})
            if r["corrected"] > best["corrected"]:
                best = {"step": it, "corrected": r["corrected"]}
                torch.save({"head": head.state_dict(), "step": it},
                           os.path.join(args.out_dir, "h4_best_%s.pth" % args.cls))
            print("  step=%d loss=%.4f(bce=%.4f dice=%.4f) val: student=%.4f fused=%.4f corrected=%.4f%s"
                  % (it, loss, bce, dice, r["student"], r["fused"], r["corrected"],
                     " *best*" if r["corrected"] == best["corrected"] else ""), flush=True)
            head.train(); model.eval()

    out = {"config": args.config, "cls": args.cls, "bce_w": args.bce_w, "dice_w": args.dice_w, "ckpt": args.ckpt, "iters": args.iters, "lr": args.lr,
           "zero_step_baseline": base, "history": hist,
           "best": best,
           "gain_vs_fused": best["corrected"] - base["fused"],
           "gain_vs_student": best["corrected"] - base["student"],
           "status": "PASS" if best["corrected"] > base["fused"] else "NO_IMPROVEMENT"}
    p = os.path.join(args.out_dir, "experiments", "h4_post_fusion_corrector.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\nH4 状态:", out["status"],
          " best corrected=%.4f @step%d (fused=%.4f, student=%.4f)"
          % (best["corrected"], best["step"], base["fused"], base["student"]))
    print("写出:", p)


if __name__ == "__main__":
    main()
