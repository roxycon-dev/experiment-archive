#!/usr/bin/env python
"""Read-only probe: image/label geometry + GT connected-component statistics.

Never touches test.  Writes a JSON report only.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

DR = "/root/autodl-tmp/hr_exec_20260919/data/DDR_split_613"
CLASSES = ("EX", "HE", "SE", "MA")          # label value c+1
OUT = "/root/autodl-tmp/HRDecoder_FullCompetition_20260920/refiner_v1/audit/input_stats.json"


def cc_stats(binary: np.ndarray) -> dict:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64) if n > 1 else np.zeros(0, np.int64)
    return {"n_cc": int(n - 1),
            "areas": areas,
            "bbox_w": stats[1:, cv2.CC_STAT_WIDTH].astype(np.int64) if n > 1 else np.zeros(0, np.int64),
            "bbox_h": stats[1:, cv2.CC_STAT_HEIGHT].astype(np.int64) if n > 1 else np.zeros(0, np.int64)}


def pct(a: np.ndarray) -> dict:
    if a.size == 0:
        return {"n": 0}
    q = np.percentile(a, [50, 90, 99]).tolist()
    return {"n": int(a.size), "min": int(a.min()), "p50": float(q[0]),
            "p90": float(q[1]), "p99": float(q[2]), "max": int(a.max()),
            "mean": float(a.mean())}


def main() -> int:
    rep = {"data_root": DR, "splits": {}}
    for split in ("train", "val"):
        img_dir = os.path.join(DR, "images", split)
        lab_dir = os.path.join(DR, "labels", split)
        stems = sorted(os.path.splitext(f)[0]
                       for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
        labs = sorted(os.path.splitext(f)[0]
                      for f in os.listdir(lab_dir) if f.endswith(".png"))
        if stems != labs:
            raise RuntimeError("%s: image/label stem mismatch" % split)

        hw, hs, ws = [], [], []
        lab_vals = set()
        per_class = {c: {"pos_images": 0, "pixel_total": 0, "cc_per_image": [],
                         "cc_areas": [], "cc_w": [], "cc_h": []} for c in CLASSES}
        for k, stem in enumerate(stems):
            im = cv2.imread(os.path.join(img_dir, stem + ".jpg"), cv2.IMREAD_COLOR)
            if im is None:
                raise RuntimeError("cannot read %s" % stem)
            hs.append(im.shape[0]); ws.append(im.shape[1])
            hw.append((im.shape[0], im.shape[1]))
            lb = cv2.imread(os.path.join(lab_dir, stem + ".png"), cv2.IMREAD_UNCHANGED)
            lab_vals.update(np.unique(lb).tolist())
            if lb.shape[:2] != im.shape[:2]:
                # label may be stored at a different size; record it
                hw[-1] = (hw[-1], "MISMATCH")
            tot_px = im.shape[0] * im.shape[1]
            for ci, cname in enumerate(CLASSES):
                b = (lb == ci + 1)
                n_px = int(b.sum())
                st = cc_stats(b) if n_px > 0 else {"n_cc": 0, "areas": np.zeros(0, np.int64),
                                                   "bbox_w": np.zeros(0, np.int64),
                                                   "bbox_h": np.zeros(0, np.int64)}
                per_class[cname]["pixel_total"] += n_px
                if n_px > 0:
                    per_class[cname]["pos_images"] += 1
                    per_class[cname]["cc_per_image"].append(st["n_cc"])
                    per_class[cname]["cc_areas"].append(st["areas"])
                    per_class[cname]["cc_w"].append(st["bbox_w"])
                    per_class[cname]["cc_h"].append(st["bbox_h"])
            del im, lb
        out = {"n_images": len(stems),
               "label_unique_values": sorted(int(v) for v in lab_vals),
               "H": {"min": int(min(hs)), "p50": float(np.percentile(hs, 50)),
                     "max": int(max(hs))},
               "W": {"min": int(min(ws)), "p50": float(np.percentile(ws, 50)),
                     "max": int(max(ws))},
               "px_per_image": {"p50": float(np.median([h * w for h, w in hw]))},
               "per_class": {}}
        for cname in CLASSES:
            d = per_class[cname]
            areas = np.concatenate(d["cc_areas"]) if d["cc_areas"] else np.zeros(0, np.int64)
            cw = np.concatenate(d["cc_w"]) if d["cc_w"] else np.zeros(0, np.int64)
            ch = np.concatenate(d["cc_h"]) if d["cc_h"] else np.zeros(0, np.int64)
            cpi = np.asarray(d["cc_per_image"], dtype=np.int64) if d["cc_per_image"] else np.zeros(0, np.int64)
            out["per_class"][cname] = {
                "pos_images": d["pos_images"],
                "pixel_total": d["pixel_total"],
                "cc_total": int(areas.size),
                "cc_per_image": pct(cpi),
                "cc_area": pct(areas),
                "cc_bbox_w": pct(cw),
                "cc_bbox_h": pct(ch),
                "frac_cc_area_le_10": float((areas <= 10).mean()) if areas.size else None,
                "frac_cc_area_le_50": float((areas <= 50).mean()) if areas.size else None,
                "frac_cc_area_ge_500": float((areas >= 500).mean()) if areas.size else None,
            }
        rep["splits"][split] = out
        print("[done] %s n=%d" % (split, len(stems)), flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print("[wrote] %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
