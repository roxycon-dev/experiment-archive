#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H12：整图级监督的 MA 校正头（针对 §9.1 的 train/inference 不一致）。

与 H4/H6 的唯一区别 = **监督分辨率**：
  H4/H6：在 1024 crop 上算特征与损失（与推理的整图+重叠平铺不一致）
  H12  ：在推理同款的整图尺寸上算特征与损失（Resize(2048) 管线 + 整图 gt）

其余全部保持不变：同样的 7 通道特征、同样的头结构、同样的损失、
同样的 ensemble（D×3+E×2+B+E4 + 翻转），所以是干净的单变量对照。
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
IDX = {"EX": 0, "HE": 1, "SE": 2, "MA": 3}


def logit(p, eps=1e-4):
    q = p.clamp(eps, 1 - eps)
    return torch.log(q / (1 - q))


def soft_dice(p, g, eps=1.0):
    return 1 - (2 * (p * g).sum()) / ((p * p).sum() + (g * g).sum() + eps)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return 6.0 * torch.tanh(self.net(f))


def unwrap(x):
    n = 0
    while hasattr(x, "data") and n < 8:
        x = x.data; n += 1
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--cls", default="MA")
    ap.add_argument("--bce-w", type=float, default=0.5)
    ap.add_argument("--dice-w", type=float, default=0.5)
    ap.add_argument("--models", required=True)
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()
    C = IDX[args.cls]; LBL = C + 1

    from mmcv import Config
    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmcv.runner import load_checkpoint

    base = Config.fromfile("configs/lesion/exp/hrplus_patch_dice.py")
    # 训练用「推理同款」管线（Resize(2048) + Normalize），只是换成 train 的图
    dtr = copy.deepcopy(base.data["train"])
    # 取 train 的「读图 + 读标签」，其余换成 test 的「Resize(2048)+Normalize+ToTensor+Collect」
    keep = [p for p in copy.deepcopy(base.data["train"]["pipeline"])
            if p.get("type") in ("LoadImageFromFile", "LoadAnnotations")]
    geo = copy.deepcopy(base.data["test"]["pipeline"])
    for p in geo:                       # 让 Collect 也带上标签，否则 test_mode 下没有 gt
        if p.get("type") == "Collect":
            keys = list(p.get("keys", []))
            if "gt_semantic_seg" not in keys:
                keys.append("gt_semantic_seg")
            p["keys"] = keys
    dtr["pipeline"] = keep + geo
    dtr.test_mode = False               # 需要 gt，不能走 test_mode 的裁剪逻辑
    ds_tr = build_dataset(dtr)
    dva = copy.deepcopy(base.data["val"]); dva.test_mode = True
    ds_va = build_dataset(dva)
    gt_va = ds_va.get_gt_seg_maps()
    gt_tr = ds_tr.get_gt_seg_maps()   # 直接取标签，不依赖 pipeline 的 Collect 键
    print("train=%d val=%d （整图监督管线）" % (len(ds_tr), len(ds_va)), flush=True)

    dm = build_segmentor(base.model)
    load_checkpoint(dm, "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth",
                    map_location="cpu", strict=False)
    dm = dm.cuda().eval()
    for p in dm.parameters():
        p.requires_grad_(False)

    specs = []
    for sp in args.models.split(";"):
        sp = sp.strip()
        if not sp:
            continue
        cfgp, rest = sp.split("=", 1)
        ck, w = (rest.rsplit(":", 1) + ["1.0"])[:2] if ":" in rest else (rest, "1.0")
        specs.append((cfgp, ck, float(w)))
    members = []
    for cfgp, ck, w in specs:
        c = Config.fromfile(cfgp)
        m = build_segmentor(c.model)
        load_checkpoint(m, ck, map_location="cpu", strict=False)
        members.append((m.cuda().eval(), w))
    flip = not args.no_flip
    tot = sum(w for _, w in members) * (2 if flip else 1)
    print("集成成员 =", [s[2] for s in specs], " flip =", flip, flush=True)

    head = Head().cuda()
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    base_lr = args.lr

    def feat_of(img, meta, ori, pf):
        """在整图上构造 7 通道特征（与推理同分辨率）。"""
        with torch.no_grad():
            ps = dm._student_prob(img, meta, True)[0]
            pr = dm._reference_prob(img, meta, True)[0]
        img_f = F.interpolate(img, size=tuple(ori), mode="bilinear",
                              align_corners=False)[0]
        return torch.cat([img_f, logit(pf[C:C + 1]), logit(ps[C:C + 1]),
                          logit(pr[C:C + 1]), (ps[C:C + 1] - pr[C:C + 1])],
                         dim=0).unsqueeze(0), ps, pr

    def ensemble_prob(img, meta, ori):
        acc = None
        with torch.no_grad():
            for m, w in members:
                p = m(return_loss=False, rescale=True, img=[img], img_metas=[meta])[0]
                p = np.asarray(p[0] if isinstance(p, tuple) else p, dtype=np.float32)
                acc = p * w if acc is None else acc + p * w
                if flip:
                    imf = img.flip(-1)
                    mf = copy.deepcopy(meta)
                    mf[0]["flip"] = True; mf[0]["flip_direction"] = "horizontal"
                    p2 = m(return_loss=False, rescale=True, img=[imf], img_metas=[mf])[0]
                    p2 = np.asarray(p2[0] if isinstance(p2, tuple) else p2, dtype=np.float32)
                    acc = acc + p2 * w
        return torch.from_numpy(acc / tot).cuda()

    def val_eval():
        agg = {"fused": [0, 0, 0], "corrected": [0, 0, 0]}
        with torch.no_grad():
            for i in range(len(ds_va)):
                s = ds_va[i]
                img = as_img(s["img"]).unsqueeze(0).cuda()
                meta = as_meta(s["img_metas"])
                gt = np.asarray(gt_va[i]); ori = meta[0]["ori_shape"][:2]
                if gt.shape != tuple(ori):
                    import cv2
                    gt = cv2.resize(gt.astype(np.uint8), (ori[1], ori[0]),
                                    interpolation=cv2.INTER_NEAREST)
                gtf = torch.from_numpy(gt).cuda()
                pf = ensemble_prob(img, meta, ori)
                f7, _, _ = feat_of(img, meta, ori, pf)
                pnew = torch.sigmoid(logit(pf[C:C + 1]) + head(f7)).squeeze(0).squeeze(0)
                g = gtf == LBL; ig = gtf == 255
                for tag, pr in (("fused", pf[C]), ("corrected", pnew)):
                    p = pr > 0.5
                    agg[tag][0] += int((p & g & ~ig).sum())
                    agg[tag][1] += int((p & ~g & ~ig).sum())
                    agg[tag][2] += int((~p & g).sum())
        out = {}
        for k, (tp, fp, fn) in agg.items():
            den = tp + fp + fn
            out[k] = tp / den if den else 0.0
        return out

    print("== 零步基线 ==", flush=True)
    b = val_eval()
    print("  fused=%.4f corrected=%.4f" % (b["fused"], b["corrected"]), flush=True)
    hist = [{"step": 0, **b}]
    best = {"step": 0, "corrected": b["corrected"]}
    rng = np.random.RandomState(0)
    t0 = time.time()
    for it in range(1, args.iters + 1):
        idx = int(rng.randint(0, len(ds_tr)))
        s = ds_tr[idx]
        img = as_img(s["img"]).unsqueeze(0).cuda()
        meta = as_meta(s["img_metas"])
        gt = torch.from_numpy(np.asarray(gt_tr[idx])).long().cuda()
        while gt.dim() < 2:
            gt = gt.unsqueeze(0)
        ori = meta[0]["ori_shape"][:2]
        if gt.shape[-2:] != tuple(ori):
            import cv2
            gt = torch.from_numpy(cv2.resize(
                gt.cpu().numpy().astype(np.uint8), (ori[1], ori[0]),
                interpolation=cv2.INTER_NEAREST)).long().cuda()

        pf = ensemble_prob(img, meta, ori)
        f7, ps, pr = feat_of(img, meta, ori, pf)
        delta = head(f7)
        new_logit = logit(pf[C:C + 1]) + delta
        p_new = torch.sigmoid(new_logit)
        g3 = gt.squeeze(0) if gt.dim() == 3 else gt
        y = (g3 == LBL).float().unsqueeze(0).unsqueeze(0)
        ign = (g3 == 255).float().unsqueeze(0).unsqueeze(0)
        valid = 1.0 - ign
        bce = (F.binary_cross_entropy_with_logits(new_logit, y, reduction="none")
               * valid).sum() / valid.sum().clamp(min=1)
        dice = soft_dice(p_new * valid, y * valid)
        loss = args.bce_w * bce + args.dice_w * dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        for g_ in opt.param_groups:
            g_["lr"] = base_lr * (1 - it / args.iters)
        opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            head.eval()
            r = val_eval()
            head.train()
            hist.append({"step": it, **r, "loss": float(loss),
                         "elapsed": round(time.time() - t0, 1)})
            if r["corrected"] > best["corrected"]:
                best = {"step": it, "corrected": r["corrected"]}
                torch.save({"head": head.state_dict(), "step": it},
                           os.path.join(args.out_dir, "h12_best_%s.pth" % args.cls))
            print("  step=%-5d loss=%.4f fused=%.4f corrected=%.4f%s"
                  % (it, loss, r["fused"], r["corrected"],
                     " *best*" if r["corrected"] == best["corrected"] else ""),
                  flush=True)

    out = {"probe": "H12_whole_image_supervision", "cls": args.cls,
           "iters": args.iters, "lr": args.lr, "bce_w": args.bce_w, "dice_w": args.dice_w,
           "flip": flip, "zero_step": b, "history": hist, "best": best,
           "gain_vs_fused": best["corrected"] - b["fused"],
           "status": "PASS" if best["corrected"] > b["fused"] else "NO_IMPROVEMENT"}
    os.makedirs(os.path.join(args.out_dir, "experiments"), exist_ok=True)
    p = os.path.join(args.out_dir, "experiments", "h12_whole_image.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\nH12 状态:", out["status"], " best=%.4f @%d (fused=%.4f)"
          % (best["corrected"], best["step"], b["fused"]))
    print("写出:", p)


if __name__ == "__main__":
    main()
