#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verify_ragged_fix — real-image evidence for the ragged-patch fix.

Runs on ONE real val image (champion + frozen candidate rule + the real stage-1
checkpoint) and re-derives, on shipped data, everything the unit tests assert on
synthetic tiles:

  1. legacy_repro      the shipped eval statement `np.stack(patches)` still
                       raises ValueError on this image's patches (the exact
                       reported failure), and the fixed path returns them ragged
  2. cache_geometry    the extraction window the evaluator recomputes is the
                       SAME rectangle the frozen patch cache stored (tiles and
                       rects compared element-wise against the cached npz)
  3. identity          a zero-init refiner reproduces the champion BITWISE on
                       the full assembled map
  4. bucket_vs_perpatch  shape-bucketed batch-8 inference == one-patch-at-a-time
                       reference (max abs diff + exact-equal element count), on
                       the trained checkpoint
  5. invariants        EX/SE changed=0, outside-candidate changed=0, no NaN/Inf,
                       strict state_dict load

Writes a JSON report (default: logs/verify_ragged_fix_<stem>.json).
Nothing is written outside that report; no test data is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from refiner import assemble as A                        # noqa: E402
from refiner import candidate as C                        # noqa: E402
from refiner import guards as G                           # noqa: E402
from refiner import tiled_infer as TI                     # noqa: E402
from refiner.model import HaarHeteroRefiner               # noqa: E402
from refiner.reference import (FrozenChampion, DEFAULT_DATA_ROOT,  # noqa: E402
                               DEFAULT_REPO)

CACHE_ROOT = ("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/"
              "refiner_v1/patch_cache")
RUN_ROOT = ("/root/autodl-tmp/HRDecoder_FullCompetition_20260920/"
            "refiner_v1/runs/formal_20260921")
CONTRACT = {"mIoU": 0.42669428454584546, "MA": 0.259672, "HE": 0.437583}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(_HERE),
                                                     "configs", "refiner_v1.json"))
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--ckpt", default=os.path.join(RUN_ROOT, "stage1", "nodes",
                                                   "node_ep200.pth"))
    ap.add_argument("--cache-root", default=CACHE_ROOT)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.data_root = os.path.abspath(args.data_root)

    cfg = json.load(open(args.config, encoding="utf-8"))
    G.assert_val_only_inference(args.split, args.config, args.data_root,
                                args.cache_root, args.ckpt)
    t = float(cfg["candidate"]["threshold"])
    dl = int(cfg["candidate"]["dilate"])
    ctx = int(cfg["cache"]["context"])
    maxp = int(cfg["cache"]["max_patch"])

    import cv2
    champ = FrozenChampion(repo=DEFAULT_REPO, data_root=args.data_root,
                           device=args.device)
    ds = champ.dataset(args.split)
    s = ds[args.index]
    meta = champ.rt._unwrap_meta(s["img_metas"])
    ori = tuple(int(x) for x in meta[0]["ori_shape"][:2])
    stem = os.path.splitext(os.path.basename(meta[0]["ori_filename"]))[0]
    H, W = ori
    img = champ.rt._unwrap_img(s["img"]).unsqueeze(0).to(args.device)
    p4 = champ.champion_prob4(img, meta)
    del img
    torch.cuda.empty_cache()
    tiles, boxes = C.candidate_tiles(p4[C.CLS_INDEX["MA"]], p4[C.CLS_INDEX["HE"]],
                                     t, dl)
    wins = C.windows_for(tiles, H, W, ctx, maxp)
    rep = {"schema": "hrdecoder-refiner-ragged-fix-verify/v1",
           "split": args.split, "index": args.index, "stem": stem,
           "image_hw": [H, W], "n_tiles": len(tiles), "n_boxes": len(boxes),
           "candidate": {"threshold": t, "dilate": dl, "margin": C.MARGIN},
           "extraction": {"context": ctx, "max_patch": maxp},
           "device": args.device, "ckpt": args.ckpt,
           "window_shapes": sorted({(w[3] - w[1], w[2] - w[0]) for w in wins}),
           "n_distinct_window_shapes": len({(w[3] - w[1], w[2] - w[0])
                                            for w in wins})}

    bgr = cv2.imread(os.path.join(args.data_root, "images", args.split,
                                 stem + ".jpg"), cv2.IMREAD_COLOR)
    if bgr.shape[:2] != ori:
        bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb_full = np.ascontiguousarray(bgr[:, :, ::-1])
    ma_he = np.ascontiguousarray(np.stack([p4[C.CLS_INDEX["MA"]],
                                           p4[C.CLS_INDEX["HE"]]], 0))

    # ---- 2. cache geometry: same rectangle as the frozen training cache ----
    npz_p = os.path.join(args.cache_root, "patch", args.split, stem + ".npz")
    z = np.load(npz_p, allow_pickle=False)
    ct = np.asarray(z["tiles"], np.int64)
    cr = np.asarray(z["rects"], np.int64)
    cs = np.asarray(z["shape"], np.int64)
    n_cand = int((np.asarray(z["role"]) != 3).sum())
    same_tiles = bool(n_cand == len(tiles)) and bool(np.array_equal(
        ct[:n_cand], np.asarray(tiles, np.int64)))
    same_rects = bool(np.array_equal(cr[:n_cand], np.asarray(wins, np.int64)))
    same_shapes = bool(np.array_equal(
        cs[:n_cand], np.asarray([[w[3] - w[1], w[2] - w[0]] for w in wins], np.int64)))
    rep["cache_geometry"] = {"npz": npz_p, "cached_candidate_tiles": n_cand,
                            "tiles_identical": same_tiles,
                            "rects_identical": same_rects,
                            "shapes_identical": same_shapes,
                            "note": "the evaluator cuts exactly the rectangle the "
                                    "training cache stored"}
    assert same_tiles and same_rects and same_shapes, rep["cache_geometry"]
    del z, ct, cr, cs

    # ---- 3. zero-init refiner: bitwise identity over ragged patches ---------
    m0 = HaarHeteroRefiner(width=int(cfg["model"]["width"]),
                           residual_scale=float(cfg["model"]["residual_scale"]))
    m0 = m0.to(args.device).eval()
    for p in m0.parameters():
        p.requires_grad_(False)
    pat0, _w0, st0 = TI.refine_tiles(m0, rgb_full, ma_he, tiles, H, W, context=ctx,
                                     max_patch=maxp, batch=args.batch,
                                     device=args.device)
    out0 = A.assemble(p4, tiles, pat0)
    rep["identity_zero_init"] = {
        "assembled_bitwise_champion": bool(np.array_equal(out0, p4)),
        "max_abs_diff": float(np.abs(out0 - p4).max()),
        "n_buckets": st0["n_buckets"], "n_forward": st0["n_forward"]}
    del m0, pat0, out0

    # ---- 1. the legacy statement must still fail (it is the reported bug) ---
    m = HaarHeteroRefiner(width=int(cfg["model"]["width"]),
                          residual_scale=float(cfg["model"]["residual_scale"]))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = ck.get("model", ck)
    missing, unexpected = m.load_state_dict(state, strict=True)
    m = m.to(args.device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    rep["strict_load"] = {"missing": list(missing), "unexpected": list(unexpected),
                          "stage": ck.get("stage"), "step": ck.get("step"),
                          "ok": not missing and not unexpected}

    pat_b, _wb, stb = TI.refine_tiles(m, rgb_full, ma_he, tiles, H, W, context=ctx,
                                      max_patch=maxp, batch=args.batch,
                                      device=args.device)
    try:
        np.stack(pat_b)
        legacy = "NO ERROR (unexpected)"
    except ValueError as e:
        legacy = "ValueError: %s" % e
    rep["legacy_repro"] = {"statement": "np.stack(patches)",
                           "result": legacy,
                           "fixed_path_shapes": [list(p.shape) for p in pat_b[:6]],
                           "fixed_path_is_ragged": len({p.shape for p in pat_b}) > 1}
    rep["bucket_stats"] = stb

    pat_1, _w1, st1 = TI.refine_tiles(m, rgb_full, ma_he, tiles, H, W, context=ctx,
                                      max_patch=maxp, batch=1, device=args.device)
    per_tile = []
    for i, tl in enumerate(tiles):
        want = (len(C.REFINED_CLASSES), tl[3] - tl[1], tl[2] - tl[0])
        d = np.abs(pat_b[i] - pat_1[i])
        per_tile.append({"tile": [int(v) for v in tl],
                         "shape": list(pat_b[i].shape),
                         "shape_ok": list(pat_b[i].shape) == list(want),
                         "max_abs_diff": float(d.max()),
                         "n_exact": int(np.count_nonzero(d == 0)),
                         "n_elements": int(d.size)})
    rep["bucket_vs_per_patch"] = {
        "per_tile": per_tile,
        "max_abs_diff": max(x["max_abs_diff"] for x in per_tile),
        "all_shapes_ok": all(x["shape_ok"] for x in per_tile),
        "n_exact_elements": sum(x["n_exact"] for x in per_tile),
        "n_elements": sum(x["n_elements"] for x in per_tile)}

    out_b = A.assemble(p4, tiles, pat_b)
    out_1 = A.assemble(p4, tiles, pat_1)
    rb = A.assert_bitwise_unchanged(out_b, p4, tiles)
    rep["invariants"] = {
        "ex_se_changed": int(rb["EX_total_changed"] + rb["SE_total_changed"]),
        "outside_cand_changed": int(sum(rb["%s_changed_outside" % c]
                                        for c in C.CLASSES)),
        "changed_inside_candidate": int(rb["MA_changed_inside"] + rb["HE_changed_inside"]),
        "nonfinite_refined": int(np.count_nonzero(~np.isfinite(out_b[
            C.CLS_INDEX["MA"]])) + np.count_nonzero(~np.isfinite(out_b[
                C.CLS_INDEX["HE"]]))),
        "assembled_bucket_vs_perpatch_max_abs_diff": float(np.abs(out_b - out_1).max()),
        "assembled_bucket_vs_perpatch_n_diff": int(np.count_nonzero(out_b != out_1)),
        "n_candidate_px": int(C.mask_from_tiles(tiles, ori).sum()),
    }
    mae = float(np.abs(out_b[C.CLS_INDEX["MA"]] - p4[C.CLS_INDEX["MA"]]).max())
    rep["residual_bound_ok"] = bool(mae <= 0.5 + 1e-6)
    rep["max_abs_residual_MA"] = mae
    rep["verdict"] = bool(
        rep["strict_load"]["ok"] and rep["identity_zero_init"]["assembled_bitwise_champion"]
        and rep["bucket_vs_per_patch"]["all_shapes_ok"]
        and rep["bucket_vs_per_patch"]["max_abs_diff"] < 1e-5
        and rep["invariants"]["ex_se_changed"] == 0
        and rep["invariants"]["outside_cand_changed"] == 0
        and rep["invariants"]["nonfinite_refined"] == 0
        and legacy.startswith("ValueError")
        and rep["cache_geometry"]["tiles_identical"]
        and rep["cache_geometry"]["rects_identical"])

    out_p = args.out or os.path.join(_HERE, "logs",
                                     "verify_ragged_fix_%s.json" % stem)
    out_p = os.path.abspath(out_p)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("[verify] %s tiles=%d distinct_shapes=%d buckets=%d forwards=%d"
          % (stem, len(tiles), rep["n_distinct_window_shapes"],
             stb["n_buckets"], stb["n_forward"]))
    print("[verify] identity bitwise=%s  legacy np.stack -> %s"
          % (rep["identity_zero_init"]["assembled_bitwise_champion"], legacy))
    print("[verify] bucket vs per-patch: max|d|=%.3e exact=%d/%d"
          % (rep["bucket_vs_per_patch"]["max_abs_diff"],
             rep["bucket_vs_per_patch"]["n_exact_elements"],
             rep["bucket_vs_per_patch"]["n_elements"]))
    print("[verify] invariants: EX/SE changed=%d outside=%d nonfinite=%d "
          "changed_inside=%d"
          % (rep["invariants"]["ex_se_changed"],
             rep["invariants"]["outside_cand_changed"],
             rep["invariants"]["nonfinite_refined"],
             rep["invariants"]["changed_inside_candidate"]))
    print("[verify] cache geometry: tiles=%s rects=%s shapes=%s"
          % (same_tiles, same_rects, same_shapes))
    print("[verify] VERDICT=%s  wrote %s" % (rep["verdict"], out_p))
    return 0 if rep["verdict"] else 3


if __name__ == "__main__":
    sys.exit(main())
