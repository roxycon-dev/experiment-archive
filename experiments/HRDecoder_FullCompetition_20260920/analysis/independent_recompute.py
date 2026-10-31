# -*- coding: utf-8 -*-
"""独立复算：不 import 分析器/契约模块，从 npz 原始二进制重新解包并重算 TP/FP/FN。

目的：
  A) 逐文件核对 contract.per_image[stem].file_sha256 与磁盘实际内容；
  B) 独立重算逐类 TP/FP/FN/IoU，与各模型 contract.global_counts / metrics 对比；
  C) 目录间无覆盖检查（同 stem 在不同模型目录必须内容不同，除已知的三通道共享）；
  D) 与 analysis/expert_complementarity.json 的 scalar 结果交叉核对；
  E) 逐类 OR/AND oracle 与"单类替换"独立复算。
输出：JSON 到 stdout（同时落一份 /root/logs_run/independent_recompute.json）。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

OUT = "/root/autodl-tmp/HRDecoder_FullCompetition_20260920/expert_outputs"
CLASSES = ["EX", "HE", "SE", "MA"]
DENOM_GUARD = 1e-12


def unpad_unpack(packed_row: np.ndarray, n_pixels: int, hw) -> np.ndarray:
    """packed_row: uint8 1-D 或 bool；返回 (h,w) bool。"""
    bits = np.unpackbits(np.asarray(packed_row, dtype=np.uint8), bitorder="big")
    bits = bits[: int(n_pixels)]
    return bits.reshape(int(hw[0]), int(hw[1])).astype(bool)


def load_mask(path):
    with np.load(path) as z:
        packed = np.asarray(z["packed"])
        npx = int(np.asarray(z["n_pixels"]).reshape(-1)[0])
        hw = tuple(int(x) for x in np.asarray(z["orig_hw"]).reshape(-1)[:2])
    c = packed.shape[0]
    out = np.empty((c, hw[0], hw[1]), dtype=bool)
    for k in range(c):
        out[k] = unpad_unpack(packed[k], npx, hw)
    return out, npx, hw


def load_gt(path):
    with np.load(path) as z:
        packed = np.asarray(z["packed"])
        igp = np.asarray(z["ignore_packed"])
        npx = int(np.asarray(z["n_pixels"]).reshape(-1)[0])
        hw = tuple(int(x) for x in np.asarray(z["orig_hw"]).reshape(-1)[:2])
    c = packed.shape[0]
    gt = np.empty((c, hw[0], hw[1]), dtype=bool)
    for k in range(c):
        gt[k] = unpad_unpack(packed[k], npx, hw)
    ignore = unpad_unpack(igp, npx, hw)
    return gt, ignore, npx, hw


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def counts(pred, gt, ignore):
    valid = ~ignore
    tp = int(np.count_nonzero(pred & gt & valid))
    fp = int(np.count_nonzero(pred & ~gt & valid))
    fn = int(np.count_nonzero(~pred & gt & valid))
    return tp, fp, fn


def iou(tp, fp, fn):
    u = tp + fp + fn
    return (tp / u) if u else float("nan")


def main():
    models = []
    for fam in ("hrplus", "fullmodel"):
        base = os.path.join(OUT, fam)
        if not os.path.isdir(base):
            continue
        for m in sorted(os.listdir(base)):
            if os.path.isdir(os.path.join(base, m)):
                models.append((f"{fam}/{m}", os.path.join(base, m)))

    gt_dir = os.path.join(OUT, "_shared", "gt")
    stems = sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".npz"))

    # ---- GT 载入一次 ----
    GT = {}
    for s in stems:
        gt, ig, npx, hw = load_gt(os.path.join(gt_dir, s + ".npz"))
        GT[s] = (gt, ig, npx, hw)

    # GT 总量（用于 oracle 分母参考）
    out = {"stems": stems, "n_stems": len(stems), "models": {}}
    gt_pos = {c: 0 for c in CLASSES}
    for s in stems:
        gt, ig, _, _ = GT[s]
        valid = ~ig
        for ci, c in enumerate(CLASSES):
            gt_pos[c] += int(np.count_nonzero(gt[ci] & valid))
    out["gt_positive_pixels"] = gt_pos

    masks_cache = {}
    for name, d in models:
        files = sorted(f for f in os.listdir(d) if f.endswith(".npz"))
        rec = {"n_files": len(files), "per_class": {}, "contract_file_sha_mismatch": [],
               "files_not_in_gt": [], "gt_missing": []}
        # A) 文件名集合 == GT stems
        fstems = [f[:-4] for f in files]
        rec["files_not_in_gt"] = sorted(set(fstems) - set(stems))
        rec["gt_missing"] = sorted(set(stems) - set(fstems))
        # contract
        cpath = os.path.join(d, "contract.json")
        contract = json.load(open(cpath, encoding="utf-8")) if os.path.exists(cpath) else None
        agg = {c: [0, 0, 0] for c in CLASSES}
        per_image_counts = {}
        for f in files:
            s = f[:-4]
            p = os.path.join(d, f)
            # A) sha 核对
            if contract is not None:
                exp = contract.get("per_image", {}).get(s, {}).get("file_sha256")
                if exp:
                    if sha256_file(p) != exp:
                        rec["contract_file_sha_mismatch"].append(s)
            pred, npx, hw = load_mask(p)
            gt, ig, _, _ = GT[s]
            assert pred.shape == gt.shape, (name, s, pred.shape, gt.shape)
            masks_cache[(name, s)] = (pred, ig)
            pc = {}
            for ci, c in enumerate(CLASSES):
                tp, fp, fn = counts(pred[ci], gt[ci], ig)
                agg[c][0] += tp
                agg[c][1] += fp
                agg[c][2] += fn
                pc[c] = [tp, fp, fn]
            per_image_counts[s] = pc
        mi = [iou(*agg[c]) for c in CLASSES]
        rec["per_class"] = {c: {"tp": agg[c][0], "fp": agg[c][1], "fn": agg[c][2],
                                "iou": iou(*agg[c])} for c in CLASSES}
        rec["mIoU"] = float(np.mean(mi))
        rec["totals"] = {"tp": sum(agg[c][0] for c in CLASSES),
                         "fp": sum(agg[c][1] for c in CLASSES),
                         "fn": sum(agg[c][2] for c in CLASSES)}
        rec["per_image_counts"] = per_image_counts
        # B) 与 contract 对比
        if contract is not None:
            gc = contract.get("global_counts", {})
            diffs = {}
            for c in CLASSES:
                if c in gc:
                    ct, cf, cn = [int(x) for x in gc[c]]
                    if [ct, cf, cn] != agg[c]:
                        diffs[c] = {"contract": [ct, cf, cn], "recomputed": agg[c]}
            rec["contract_global_counts_diff"] = diffs
            rec["contract_mIoU"] = contract.get("metrics", {}).get("mIoU_lesion4")
            rec["contract_mIoU_delta"] = (
                None if rec["contract_mIoU"] is None else rec["mIoU"] - rec["contract_mIoU"])
            rec["contract_n_images"] = contract.get("n_images")
            rec["contract_split"] = contract.get("split")
            rec["contract_prob_payload_saved"] = contract.get("eval_contract", {}).get(
                "prob_payload_saved")
        out["models"][name] = rec

    # C) 目录间无覆盖：同 stem 的两两比较
    names = [n for n, _ in models]
    pair_diff = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n_diff_any = 0
            n_diff_nonma = 0
            n_identical_all = 0
            for s in stems:
                pa, ia = masks_cache[(a, s)]
                pb, ib = masks_cache[(b, s)]
                d = pa != pb
                if d.any():
                    n_diff_any += 1
                else:
                    n_identical_all += 1
                if d[:3].any():          # EX/HE/SE 通道
                    n_diff_nonma += 1
            pair_diff[f"{a}|{b}"] = {"stems_with_any_channel_diff": n_diff_any,
                                     "stems_identical_all_channels": n_identical_all,
                                     "stems_with_nonMA_channel_diff": n_diff_nonma}
    out["pairwise_overwrite_check"] = pair_diff

    # D) 与 analysis JSON 交叉核对
    aj = "/root/autodl-tmp/HRDecoder_FullCompetition_20260920/expert_outputs/analysis/expert_complementarity.json"
    cross = {}
    if os.path.exists(aj):
        A = json.load(open(aj, encoding="utf-8"))
        out["analysis_json_keys"] = sorted(A.keys())
        models_json = A.get("model_metrics") or {}
        for name in names:
            mk = name.split("/")[-1]
            if mk in models_json:
                jm = models_json[mk].get("mIoU_lesion4")
                jc = models_json[mk].get("global_counts") or {}
                dc = {}
                for c in CLASSES:
                    if c in jc and [int(x) for x in jc[c]] != [
                            out["models"][name]["per_class"][c]["tp"],
                            out["models"][name]["per_class"][c]["fp"],
                            out["models"][name]["per_class"][c]["fn"]]:
                        dc[c] = {"json": [int(x) for x in jc[c]],
                                 "recomputed": [out["models"][name]["per_class"][c][k]
                                                for k in ("tp", "fp", "fn")]}
                cross[mk] = {"analysis_json_mIoU": jm,
                             "recomputed_mIoU": out["models"][name]["mIoU"],
                             "delta": None if jm is None else out["models"][name]["mIoU"] - jm,
                             "global_counts_diff": dc}
        out["analysis_json_mIoU_cross"] = cross

        # E) OR/AND + 单类替换独立复算（相对 champion），与 analysis pairwise 对比
        pw = A.get("pairwise") or {}
        or_and = {}
        for name in names:
            mk = name.split("/")[-1]
            if mk not in pw:
                continue
            entry = {}
            for ci, c in enumerate(CLASSES):
                tp = fp = fn = 0
                or_tp = or_fp = or_fn = 0
                and_tp = and_fp = and_fn = 0
                lone_tp = lone_fp = 0
                ch_only_tp = 0
                repl_tp = repl_fp = repl_fn = 0
                for s in stems:
                    pc, ig = masks_cache[("hrplus/champion", s)]
                    pd, _ = masks_cache[(name, s)]
                    gt, igg, _, _ = GT[s]
                    valid = ~igg
                    A_ = pc[ci] & valid
                    B_ = pd[ci] & valid
                    G_ = gt[ci] & valid
                    o = A_ | B_
                    a_ = A_ & B_
                    # 单类替换：把 champion 的第 c 类换成候选的第 c 类，其余类保持 champion
                    for pred, sink in ((B_, "cand"), (o, "or"), (a_, "and"), (B_, "repl")):
                        T = int(np.count_nonzero(pred & G_))
                        F = int(np.count_nonzero(pred & ~G_))
                        N = int(np.count_nonzero(~pred & G_))
                        if sink == "cand":
                            tp += T; fp += F; fn += N
                        elif sink == "or":
                            or_tp += T; or_fp += F; or_fn += N
                        elif sink == "and":
                            and_tp += T; and_fp += F; and_fn += N
                        else:
                            repl_tp += T; repl_fp += F; repl_fn += N
                    lone_tp += int(np.count_nonzero(B_ & G_ & ~A_))
                    lone_fp += int(np.count_nonzero(B_ & ~G_ & ~A_))
                    ch_only_tp += int(np.count_nonzero(A_ & G_ & ~B_))
                jp = (pw[mk].get("per_class") or {}).get(c, {})
                entry[c] = {
                    "candidate_iou": iou(tp, fp, fn),
                    "or_iou": iou(or_tp, or_fp, or_fn),
                    "and_iou": iou(and_tp, and_fp, and_fn),
                    "candidate_replace_iou": iou(repl_tp, repl_fp, repl_fn),
                    "common_tp": tp - lone_tp,
                    "candidate_only_tp": lone_tp,
                    "candidate_only_fp": lone_fp,
                    "champion_only_tp": ch_only_tp,
                    "json_candidate_iou": jp.get("candidate_iou"),
                    "json_or_iou": jp.get("or_iou"),
                    "json_candidate_only_tp": jp.get("candidate_only_tp"),
                    "json_candidate_only_fp": jp.get("candidate_only_fp"),
                    "json_champion_only_tp": jp.get("champion_only_tp"),
                }
            or_and[mk] = entry
        out["or_and_recompute_vs_champion"] = or_and

    # F) 单类替换的"整体 mIoU"（标量）：champion 的第 c 类换成候选的第 c 类，其余类保持 champion
    repl = {}
    chname = "hrplus/champion"
    if chname in names:
        for name in names:
            if name == chname:
                continue
            mk = name.split("/")[-1]
            entry = {}
            for ci, c in enumerate(CLASSES):
                # 其余 3 类沿用 champion 的累计计数；第 c 类用候选的
                acc = {k: [0, 0, 0] for k in CLASSES}
                for s in stems:
                    pc, ig = masks_cache[(chname, s)]
                    pd, _ = masks_cache[(name, s)]
                    gt, igg, _, _ = GT[s]
                    valid = ~igg
                    for cj, cc in enumerate(CLASSES):
                        pred = (pd[ci] if cj == ci else pc[cj]) & valid
                        g = gt[cj] & valid
                        acc[cc][0] += int(np.count_nonzero(pred & g))
                        acc[cc][1] += int(np.count_nonzero(pred & ~g))
                        acc[cc][2] += int(np.count_nonzero(~pred & g))
                per = {cc: iou(*acc[cc]) for cc in CLASSES}
                entry[c] = {"replace_mIoU": float(np.mean(list(per.values()))),
                            "per_class_iou": per}
            entry["_champion_mIoU"] = out["models"][chname]["mIoU"]
            repl[mk] = entry
        out["single_class_replacement_mIoU"] = repl

    txt = json.dumps(out, ensure_ascii=False, indent=1)
    os.makedirs("/root/logs_run", exist_ok=True)
    with open("/root/logs_run/independent_recompute.json", "w", encoding="utf-8") as fh:
        fh.write(txt)
    # 摘要
    print("=== SUMMARY (independent recompute) ===")
    for name in names:
        r = out["models"][name]
        print("%-18s n=%d mIoU=%.9f  %s" % (
            name, r["n_files"], r["mIoU"],
            " ".join("%s=%.6f" % (c, r["per_class"][c]["iou"]) for c in CLASSES)))
        if r.get("contract_mIoU_delta") is not None:
            print("    contract mIoU=%r delta=%.3e sha_mismatch=%d counts_diff=%s" % (
                r["contract_mIoU"], r["contract_mIoU_delta"],
                len(r["contract_file_sha_mismatch"]),
                list(r.get("contract_global_counts_diff", {}).keys())))
    print("\n=== cross vs analysis JSON ===")
    for k, v in out.get("analysis_json_mIoU_cross", {}).items():
        print("  %-12s json=%r recomputed=%r delta=%.3e" % (
            k, v["analysis_json_mIoU"], v["recomputed_mIoU"], v["delta"]))
    print("\n=== overwrite check (spot) ===")
    for k, v in list(out["pairwise_overwrite_check"].items())[:8]:
        print("  %-40s nonMA_diff_stems=%d identical_all=%d" % (
            k, v["stems_with_nonMA_channel_diff"], v["stems_identical_all_channels"]))
    print("\n=== single-class replacement mIoU (scalar, vs champion %.6f) ===" % out["models"]["hrplus/champion"]["mIoU"])
    for mk, e in out.get("single_class_replacement_mIoU", {}).items():
        print("  %-12s %s" % (mk, "  ".join(
            "%s=%.6f(Δ%+.6f)" % (c, e[c]["replace_mIoU"],
                                e[c]["replace_mIoU"] - e["_champion_mIoU"]) for c in CLASSES)))
    print("\nwrote /root/logs_run/independent_recompute.json")


if __name__ == "__main__":
    main()
