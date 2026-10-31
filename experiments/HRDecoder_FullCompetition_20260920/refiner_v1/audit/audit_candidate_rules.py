#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""audit_candidate_rules — GO/NO-GO audit for the frozen-champion MA/HE candidate refiner.

Read-only, val only, never test.  Writes exactly one JSON report.

What it establishes
-------------------
1. Rebuilds the EXACT frozen champion (runtime.ChampionEnsemble: members
   D x3 + E x2 + B x1 + E4 x1, hflip TTA, H4 post-fusion MA head) and evaluates
   val with the same mmseg test pipeline the historical export used.
2. For every val image it derives a PRE-DECLARED grid of candidate rules
       seeds = (P_MA > t) | (P_HE > t)            # low-threshold, frozen champion
       grow  = cv2 dilate by (d + margin) px       # bridges + box margin
       boxes = connected-component bboxes of `grow`
       cand  = rasterised union of boxes
   and measures the CANDIDATE RECALL UPPER BOUND
       recall_c = |GT_c inside cand| / |GT_c|      c in {MA, HE}
   A refiner that only edits inside `cand` can never exceed this number.
   `t` in {0.05, 0.10, 0.20} is the ONLY scanned axis and is used solely for
   the GO/NO-GO gate, per the pre-declared protocol.
3. False-positive separability: champion binary components (p>0.5) are split
   into TP-components (touch GT) and FP-components (touch no GT).  Shape
   (area, PCA elongation, aspect, fill, solidity) and green-channel texture
   (inside mean/std, ring contrast) features get a rank AUC for TP vs FP.
   This answers "can the vessel-like / dark-texture false positives be told
   apart by a fixed hand-crafted rule?".
4. Storage/throughput accounting for the candidate patches.

Nothing is written except --out JSON.  No weights, images or masks are saved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The rule engine is imported from the SHIPPING module so that "what was
# audited" and "what runs at inference" cannot drift apart.  There is no
# second implementation.
from refiner import candidate as RULE                      # noqa: E402

candidate_seeds = RULE.candidate_seeds
grow_mask = RULE.grow_mask
boxes_from_mask = RULE.boxes_from_mask
tile_patch = RULE.tile_patch
rasterise = RULE.mask_from_tiles      # union of boxes == union of tiles here
THRESHOLDS = RULE.SEED_THRESHOLDS
DILATIONS = RULE.SEED_DILATIONS
MARGIN = RULE.MARGIN
TILE = RULE.TILE
TILE_STRIDE = RULE.TILE_STRIDE

DATA_ROOT = "/root/autodl-tmp/hr_exec_20260919/data/DDR_split_613"
REPO = "/root/autodl-tmp/hr_exec_20260919/repo"
SRC = "/root/autodl-tmp/hr_exec_20260919/code"
SETTINGS = ("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/analysis/"
            "expert_eval_settings.json")
OUT = ("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/refiner_v1/audit/"
       "candidate_audit.json")

CLASSES = RULE.CLASSES

# ---- PRE-DECLARED grid (frozen before looking at any number) --------------
# THRESHOLDS / DILATIONS / MARGIN / TILE / TILE_STRIDE come from
# refiner.candidate (imported above) so the audit and the shipping rule are
# literally the same code.
GREEN_RING = 8               # px, ring width for green-channel contrast
MAX_COMP_PER_IMAGE = 300     # cap on per-image components analysed for shape


# ---------------------------------------------------------------------------
# rule engine (pure numpy/cv2; unit-tested on CPU in tests/test_candidate_rules.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# shape / texture features (bbox-local, no full-image scans)
# ---------------------------------------------------------------------------

def comp_features(lab, stats, ids, img_bgr, ring=GREEN_RING):
    import cv2
    H, W = lab.shape
    g = img_bgr[:, :, 1].astype(np.float32)      # BGR -> G is index 1
    feats = {}
    for cid in ids:
        x, y, w, h = (int(stats[cid, k]) for k in range(4))
        a = float(stats[cid, 4])
        if a < 1:
            continue
        sub = (lab[y:y + h, x:x + w] == cid)
        ys, xs = np.nonzero(sub)
        gx = xs + x
        gy = ys + y
        pts = np.stack([gx.astype(np.float64), gy.astype(np.float64)], 1)
        mu = pts.mean(0)
        d = pts - mu
        cov = (d.T @ d) / max(a - 1.0, 1.0)
        ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
        elong = float(np.sqrt((ev[0] + 1e-6) / (ev[1] + 1e-6)))
        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        fill = a / float(w * h)
        try:
            hull = cv2.convexHull(np.stack([gx, gy], 1).astype(np.int32))
            harea = float(cv2.contourArea(hull))
            solidity = a / harea if harea > 0 else 0.0
        except Exception:
            solidity = 0.0
        # green ring contrast, computed on a bbox-local canvas
        pad = ring
        y0 = max(0, y - pad); x0 = max(0, x - pad)
        y1 = min(H, y + h + pad); x1 = min(W, x + w + pad)
        small = (lab[y0:y1, x0:x1] == cid).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1,) * 2)
        ringm = cv2.dilate(small, k, 1).astype(bool) & (~small.astype(bool))
        gs = g[y0:y1, x0:x1]
        gin = gs[small.astype(bool)]
        ringv = gs[ringm]
        contrast = float(gin.mean() - ringv.mean()) if ringv.size else 0.0
        feats[cid] = {"area": a, "elong": elong, "aspect": aspect, "fill": fill,
                      "solidity": solidity, "bbox_w": w, "bbox_h": h,
                      "g_in_mean": float(gin.mean()), "g_in_std": float(gin.std()),
                      "g_ring_mean": float(ringv.mean()) if ringv.size else 0.0,
                      "g_contrast": contrast}
    return feats


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC of a feature as a TP-vs-FP separator (mid-ranks for ties)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rp = ranks[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def q(a):
    if len(a) == 0:
        return None
    a = np.asarray(a, np.float64)
    return {"n": int(a.size), "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "mean": float(a.mean())}


FEATURE_KEYS = ("area", "elong", "aspect", "fill", "solidity",
                "g_in_mean", "g_in_std", "g_ring_mean", "g_contrast")


def _agg(tp, fp):
    out = {"n_tp_comp": len(tp), "n_fp_comp": len(fp)}
    for key in FEATURE_KEYS:
        pv = np.array([f[key] for f in tp], np.float64)
        nv = np.array([f[key] for f in fp], np.float64)
        out[key] = {"TP": q(pv), "FP": q(nv), "auc_tp_vs_fp": rank_auc(pv, nv)}
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    # IMPORTANT: runtime.inject_repo() chdir()s into settings['repo'], so any
    # relative --out would silently land there.  Resolve it to an absolute path
    # before the runtime is imported.
    args.out = os.path.abspath(args.out)
    if "test" in args.out.lower():
        raise SystemExit("refusing an out path containing 'test'")

    raw = json.load(open(SETTINGS, encoding="utf-8"))
    import torch
    import torch.nn.functional as F
    import cv2
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    for d_ in (SRC, REPO):
        if d_ not in sys.path:
            sys.path.insert(0, d_)
    rt = importlib.import_module("runtime")
    rp = importlib.import_module("run_protocol")

    settings = rp.validate_settings(SETTINGS)
    if settings.get("_pending"):
        raise SystemExit("settings pending assets: %s" % settings["_pending"])
    cfg = rt.load_config(settings["repo"], settings["config"])
    cfg.model["source_checkpoint"] = settings["_abs"]["reference_checkpoint"]
    settings["_cfg"] = cfg
    rt.inject_repo(settings["repo"])
    from mmseg.datasets import build_dataset
    d = copy.deepcopy(cfg.data["val"])
    d.test_mode = True
    d.data_root = DATA_ROOT
    d.img_dir = os.path.join(DATA_ROOT, "images", "val")
    d.ann_dir = os.path.join(DATA_ROOT, "labels", "val")
    ds = build_dataset(d)
    if len(ds) != 76:
        raise SystemExit("val must have 76 images, got %d" % len(ds))
    print("[audit] val dataset ok n=%d" % len(ds), flush=True)

    champ = rt.ChampionEnsemble(settings, settings["champion"])
    print("[audit] champion built", flush=True)

    keys = ["t%.2f_d%d" % (t, dl) for t in THRESHOLDS for dl in DILATIONS]
    grid = {k: {"ma_hit": 0, "ma_gt": 0, "he_hit": 0, "he_gt": 0, "cand_px": 0,
                "tot_px": 0, "n_clusters": 0, "n_tiles": 0, "tile_px": 0}
            for k in keys}
    shape_tp = {"MA": [], "HE": []}
    shape_fp = {"MA": [], "HE": []}
    per_img = []
    champ_counts = {"MA": [0, 0, 0], "HE": [0, 0, 0]}

    limit = len(ds) if args.limit is None else min(args.limit, len(ds))
    t0 = time.time()
    for i in range(limit):
        s = ds[i]
        meta = rt._unwrap_meta(s["img_metas"])
        ori = tuple(int(x) for x in meta[0]["ori_shape"][:2])
        stem = os.path.splitext(os.path.basename(meta[0]["ori_filename"]))[0]
        if "test" in stem.lower():
            raise SystemExit("stem contains 'test': %s" % stem)
        img = rt._unwrap_img(s["img"]).unsqueeze(0).cuda()

        with torch.no_grad():
            pf, ps, pr = champ.pre_head_probs(img, meta)
            pf = pf.float(); ps = ps.float(); pr = pr.float()
            img_f = F.interpolate(img, size=ori, mode="bilinear", align_corners=False)[0]
            p_ma = champ.corrected_class_prob(
                3, pf, ps, pr, img_f, champ.heads["MA"]).float()
            p_he = pf[1].float()
        del img, ps, pr, img_f
        torch.cuda.empty_cache()
        p_ma = np.ascontiguousarray(p_ma.cpu().numpy(), np.float32)
        p_he = np.ascontiguousarray(p_he.cpu().numpy(), np.float32)
        if tuple(p_ma.shape) != ori or tuple(p_he.shape) != ori:
            raise SystemExit("prob shape %r/%r != ori %r" % (p_ma.shape, p_he.shape, ori))

        lb = cv2.imread(os.path.join(DATA_ROOT, "labels", "val", stem + ".png"),
                        cv2.IMREAD_UNCHANGED)
        if lb is None:
            raise SystemExit("missing GT for %s" % stem)
        if lb.shape[:2] != ori:
            lb = cv2.resize(lb, (ori[1], ori[0]), interpolation=cv2.INTER_NEAREST)
        gt = {"MA": (lb == 4), "HE": (lb == 2)}

        rec = {"stem": stem, "ori": list(ori),
               "gt_ma_px": int(gt["MA"].sum()), "gt_he_px": int(gt["HE"].sum()),
               "rules": {}}
        for t in THRESHOLDS:
            seed = candidate_seeds(p_ma, p_he, t)
            for dl in DILATIONS:
                g = grow_mask(seed, dl + MARGIN)
                bx = boxes_from_mask(g)
                cand = rasterise(bx, ori) if bx else np.zeros(ori, bool)
                tiles = []
                for b in bx:
                    tiles.extend(tile_patch(b))
                tp_px = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in tiles)
                k = "t%.2f_d%d" % (t, dl)
                G = grid[k]
                G["ma_gt"] += int(gt["MA"].sum()); G["he_gt"] += int(gt["HE"].sum())
                G["ma_hit"] += int(np.count_nonzero(gt["MA"] & cand))
                G["he_hit"] += int(np.count_nonzero(gt["HE"] & cand))
                G["cand_px"] += int(np.count_nonzero(cand))
                G["tot_px"] += ori[0] * ori[1]
                G["n_clusters"] += len(bx); G["n_tiles"] += len(tiles)
                G["tile_px"] += tp_px
                rec["rules"][k] = {
                    "ma_recall": int(np.count_nonzero(gt["MA"] & cand)) / max(1, int(gt["MA"].sum())),
                    "he_recall": int(np.count_nonzero(gt["HE"] & cand)) / max(1, int(gt["HE"].sum())),
                    "cand_frac": float(np.count_nonzero(cand)) / (ori[0] * ori[1]),
                    "n_clusters": len(bx), "n_tiles": len(tiles)}
                del cand
            del seed

        # ---- champion binary TP/FP/FN + component shape / texture ----
        bgr = cv2.imread(os.path.join(DATA_ROOT, "images", "val", stem + ".jpg"),
                         cv2.IMREAD_COLOR)
        for cname, p in (("MA", p_ma), ("HE", p_he)):
            pred = p > 0.5
            gtc = gt[cname]
            champ_counts[cname][0] += int(np.count_nonzero(pred & gtc))
            champ_counts[cname][1] += int(np.count_nonzero(pred & ~gtc))
            champ_counts[cname][2] += int(np.count_nonzero(~pred & gtc))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(
                pred.astype(np.uint8), connectivity=8)
            if n <= 1:
                continue
            ids = list(range(1, n))
            if len(ids) > MAX_COMP_PER_IMAGE:
                areas = stats[1:, cv2.CC_STAT_AREA]
                ids = list(np.argsort(-areas)[:MAX_COMP_PER_IMAGE] + 1)
            ov = np.bincount(lab[gtc].ravel(), minlength=n) if gtc.any() else np.zeros(n, np.int64)
            tp_ids = [c for c in ids if ov[c] > 0]
            fp_ids = [c for c in ids if ov[c] == 0]
            feats = comp_features(lab, stats, tp_ids + fp_ids, bgr)
            shape_tp[cname].extend(feats[c] for c in tp_ids if c in feats)
            shape_fp[cname].extend(feats[c] for c in fp_ids if c in feats)
            del lab, stats
        del bgr, p_ma, p_he
        per_img.append(rec)
        del meta
        if (i + 1) % 5 == 0 or i + 1 == limit:
            print("[audit] %d/%d %.1fs" % (i + 1, limit, time.time() - t0), flush=True)

    report = {"schema": "hrdecoder-refiner-candidate-audit/v1", "split": "val",
              "n_images": limit, "data_root": DATA_ROOT,
              "settings": SETTINGS,
              "settings_sha256": hashlib.sha256(open(SETTINGS, "rb").read()).hexdigest(),
              "predeclared_grid": {"thresholds": list(THRESHOLDS),
                                   "dilations": list(DILATIONS), "margin": MARGIN,
                                   "tile": TILE, "tile_stride": TILE_STRIDE},
              "champion_binary_counts": {
                  c: {"tp": champ_counts[c][0], "fp": champ_counts[c][1],
                      "fn": champ_counts[c][2],
                      "iou": champ_counts[c][0] / max(1, champ_counts[c][0] +
                                                      champ_counts[c][1] +
                                                      champ_counts[c][2])}
                  for c in ("MA", "HE")},
              "fp_separability": {c: _agg(shape_tp[c], shape_fp[c])
                                  for c in ("MA", "HE")},
              "n_comp_analysed": {c: {"tp": len(shape_tp[c]), "fp": len(shape_fp[c])}
                                  for c in ("MA", "HE")},
              "rule_table": {}, "per_image": per_img,
              "elapsed_s": time.time() - t0}
    for k in keys:
        G = grid[k]
        report["rule_table"][k] = {
            "threshold": float(k.split("_")[0][1:]), "dilate": int(k.split("_d")[1]),
            "ma_recall": G["ma_hit"] / max(1, G["ma_gt"]),
            "he_recall": G["he_hit"] / max(1, G["he_gt"]),
            "ma_gt_px": G["ma_gt"], "he_gt_px": G["he_gt"],
            "cand_frac_mean": G["cand_px"] / max(1, G["tot_px"]),
            "clusters_per_image": G["n_clusters"] / max(1, limit),
            "tiles_per_image": G["n_tiles"] / max(1, limit),
            "tile_px_per_image": G["tile_px"] / max(1, limit)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[audit] wrote %s (%.1fs)" % (args.out, report["elapsed_s"]))
    for k in keys:
        v = report["rule_table"][k]
        print("  %-10s MA=%.4f HE=%.4f candfrac=%.4f clust/img=%.1f tiles/img=%.1f"
              % (k, v["ma_recall"], v["he_recall"], v["cand_frac_mean"],
                 v["clusters_per_image"], v["tiles_per_image"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
