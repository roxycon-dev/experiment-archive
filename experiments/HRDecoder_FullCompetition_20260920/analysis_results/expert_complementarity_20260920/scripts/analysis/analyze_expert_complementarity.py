#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_expert_complementarity — val 上的**逐像素**专家互补性分析。

输入：`export_hrplus_expert_masks.py` + `export_fullmodel_masks.py` 产出的
packbits 掩膜目录（同一 `<out_root>` 下的共享 GT + 各模型 contract.json）。

做法（严格逐像素，绝不拼接各模型独立算出的聚合数字）：
- 先按 contract 完整校验：schema / split=val / 76 张 / 样本 ID 集合 / 逐图
  orig_hw & n_pixels / 文件 sha256 / 语义 sha256 / 共享 GT 哈希 / 评估契约逐字段；
- 再**解包位图**，重新计算每模型的逐图逐类 TP/FP/FN，并要求与契约里存的
  逐图计数、全局计数完全一致（不一致即拒绝）；
- 然后才做布尔集合运算：共同 TP、champ-only TP、cand-only TP、cand-only FP、
  champ-only FP、both_fp、both_fn、OR、AND、纯替换（=候选自身，作为锚点）；
- 每个集合的 OR/AND IoU 由**同一批像素**算出的 tp/fp/fn 得出；
- 逐图差值、按 GT 连通域面积（小/中/大）粗分的召回；
- 四类"直接专家组合"= 每类取 val 上该类 IoU 最高的模型（含 champion），
  并明确标注这是 val 选择结果、不是无偏估计。

只处理 val。split=test / 路径含 test 分量 / 图像数 != 76 一律拒绝。
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import expert_mask_contract as C  # noqa: E402

SCHEMA = "hrdecoder-expert-complementarity/v1"
DEFAULT_CHAMPION = "champion"
DEFAULT_MODELS = ("champion", "BASE_0", "D_blend", "C2000", "MIX_C2000", "S4000", "U0", "H0")
KNOWN_FAMILIES = ("hrplus", "fullmodel")
BUCKET_NAMES = ("small", "medium", "large")
CAVEATS = [
    "本分析只用 val（n=76）。历史配对 bootstrap 95% 区间为 [-0.5454,+0.8119]"
    "（ROUND_CS）与 [+0.5718,+1.1815]（H4−BASE_0），任何 < ~0.8 点的融合增益都落在噪声内。",
    "test 曾被查看 ≥5 次，已非独立泛化估计；本分析严禁使用 test，也不得用本结果去选 test 上的方案。",
    "直接四类专家组合是**在 val 上逐类 argmax 选出来的**，属于选择偏差样本，不是无偏泛化估计。",
    "历史先例：val 锁定的 0.75×champion+0.25×C@2000 在 val +0.1963，却在 test 回退 −0.4052。",
    "OR（并集）在提升召回的同时必然引入 FP；candidate_only_fp 必须与 candidate_only_tp 一起读。",
    "逐像素互补性只说明误差结构，不构成上线建议；上线需独立验证协议。",
]


# ---------------------------------------------------------------------------
# 输入装载与校验
# ---------------------------------------------------------------------------

def load_settings(path: str | None):
    if not path:
        return {}
    p = C.safe_path(path, "--settings")
    if not os.path.isfile(p):
        raise C.PreflightBlocked("settings 文件不存在: %s" % p)
    import json
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_out_root(raw: dict, cli_out_root: str | None) -> str:
    if cli_out_root:
        return C.safe_path(cli_out_root, "--out-root")
    exp = raw.get("export") or {}
    if not exp.get("out_root"):
        raise C.ContractError("缺少 out_root：请给 --out-root 或在 settings.export.out_root 中声明")
    return C.safe_path(exp["out_root"], "export.out_root")


def discover_models(out_root: str, names, *, allow_missing=()) -> dict:
    """在 <out_root>/{hrplus,fullmodel}/<name> 下发现并校验模型目录。"""
    allow_missing = set(allow_missing)
    found, missing = {}, []
    for name in names:
        hits = []
        for fam in KNOWN_FAMILIES:
            d = os.path.join(out_root, fam, name)
            if os.path.isfile(os.path.join(d, C.CONTRACT_FILENAME)):
                hits.append((fam, d))
        if not hits:
            missing.append(name)
            continue
        if len(hits) > 1:
            raise C.ContractError("模型 %s 在两个 family 目录下同时存在: %s"
                                  % (name, [h[0] for h in hits]))
        fam, d = hits[0]
        m = C.read_model_dir(d, expected_family=None)
        if m["contract"]["pipeline_family"] != fam:
            raise C.ContractError("%s: 目录 family=%s 但 contract.pipeline_family=%r"
                                  % (name, fam, m["contract"]["pipeline_family"]))
        found[name] = dict(m, family=fam)
    if missing and set(missing) - allow_missing:
        raise C.ContractError(
            "以下模型的 contract.json 缺失（未完成导出或路径不对）: %s。"
            "如确要跳过，显式传 --allow-missing NAME（会被记录到输出里）"
            % sorted(set(missing) - allow_missing))
    return found, sorted(missing)


def cross_check(found: dict, gt_man: dict) -> dict:
    """跨模型 + 与共享 GT 的一致性校验（样本 ID / 形状 / GT 哈希）。"""
    gt_stems = list(gt_man["stems"])
    gt_fp = gt_man["gt_semantic_fingerprint"]
    gt_sha = gt_man["gt_contract_sha256"]
    gt_per = gt_man["per_image"]
    info = {}
    for name, m in found.items():
        if list(m["stems"]) != gt_stems:
            raise C.ContractError(
                "%s: 样本 ID 集合/顺序与共享 GT 不一致（missing=%s extra=%s）"
                % (name,
                   sorted(set(gt_stems) - set(m["stems"]))[:5],
                   sorted(set(m["stems"]) - set(gt_stems))[:5]))
        if m["contract"]["gt"].get("gt_contract_sha256") != gt_sha:
            raise C.ContractError("%s: 记录的 gt_contract_sha256 与共享 GT 不符" % name)
        if m["contract"]["gt"].get("gt_semantic_fingerprint") != gt_fp:
            raise C.ContractError("%s: 记录的 gt_semantic_fingerprint 与共享 GT 不符" % name)
        for s in gt_stems:
            a = tuple(m["per_image"][s]["orig_hw"])
            b = tuple(gt_per[s]["orig_hw"])
            if a != b:
                raise C.ContractError("%s[%s]: orig_hw %r != GT %r（形状不一致）"
                                      % (name, s, a, b))
            if int(m["per_image"][s]["n_pixels"]) != int(gt_per[s]["n_pixels"]):
                raise C.ContractError("%s[%s]: n_pixels 与 GT 不一致" % (name, s))
        fam = m["family"]
        info[name] = {
            "family": fam,
            "dir": m["dir"],
            "contract_path": m["contract_path"],
            "contract_sha256": C.sha256_file(m["contract_path"]),
            "eval_contract": m["contract"]["eval_contract"],
            "pipeline_definition": m["contract"]["pipeline_definition"],
            "identity": m["contract"]["identity"],
            "self_checks": m["contract"]["self_checks"],
            "stored_global_counts": m["contract"]["global_counts"],
            "stored_metrics": m["contract"]["metrics"],
        }
    return info


def gt_bucket_thresholds(gt_dir: str, gt_man: dict, *, mode: str,
                         absolute=None, log) -> dict:
    """仅读 GT，算逐类连通域面积 → 小/中/大阈值与规模统计。"""
    import cv2

    per_class_areas = {c: [] for c in C.CLASSES}
    gt_px = {c: 0 for c in C.CLASSES}
    n_comp = {c: 0 for c in C.CLASSES}
    for s in gt_man["stems"]:
        p = os.path.join(gt_dir, C.MASK_FILENAME % s)
        g = C.load_gt_npz(p, where="gt[%s]" % s)
        if g["file_sha256"] != gt_man["per_image"][s]["file_sha256"]:
            raise C.ContractError("GT[%s]: 文件 sha256 与 GT 契约不符" % s)
        for ci, cname in enumerate(C.CLASSES):
            mask = g["bits"][ci].astype(np.uint8)
            gt_px[cname] += int(mask.sum())
            n, _lab, stats, _cen = cv2.connectedComponentsWithStats(mask, 8)
            if n > 1:
                areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
                per_class_areas[cname].append(areas)
                n_comp[cname] += int(areas.size)
    out = {"mode": mode, "per_class": {}}
    degenerate = []
    for cname in C.CLASSES:
        areas = (np.concatenate(per_class_areas[cname])
                 if per_class_areas[cname] else np.zeros(0, dtype=np.int64))
        if mode == "absolute":
            if not absolute or len(absolute) != 2:
                raise C.ContractError("area_mode=absolute 需要 area_thresholds=[small_max, medium_max]")
            t1, t2 = int(absolute[0]), int(absolute[1])
            if not (0 < t1 < t2):
                raise C.ContractError("area_thresholds 必须满足 0 < small_max < medium_max")
        elif mode == "tertile":
            if areas.size < 3:
                t1 = t2 = int(areas.max()) if areas.size else 0
                degenerate.append(cname)
            else:
                t1 = int(np.floor(np.quantile(areas, 1.0 / 3.0)))
                t2 = int(np.floor(np.quantile(areas, 2.0 / 3.0)))
                if not (t1 < t2):
                    t2 = t1 + 1
        else:
            raise C.ContractError("area_mode=%r 非法（tertile/absolute）" % mode)
        bidx = np.searchsorted([t1, t2], areas, side="right")
        counts = np.bincount(bidx, minlength=3) if areas.size else np.zeros(3, int)
        px = np.bincount(bidx, weights=areas, minlength=3) if areas.size else np.zeros(3)
        out["per_class"][cname] = {
            "thresholds": [t1, t2],
            "gt_positive_pixels": gt_px[cname],
            "n_components": n_comp[cname],
            "buckets": [
                {"name": BUCKET_NAMES[b],
                 "area_lo": (1 if b == 0 else (t1 + 1 if b == 1 else t2 + 1)),
                 "area_hi": (t1 if b == 0 else (t2 if b == 1 else None)),
                 "n_components": int(counts[b]),
                 "gt_pixels": int(px[b])}
                for b in range(3)
            ],
        }
        log("GT 面积分布 %s: n_comp=%d gt_px=%d thresholds=%s"
            % (cname, n_comp[cname], gt_px[cname], [t1, t2]))
    return out


def bucket_index(areas: np.ndarray, t1: int, t2: int) -> np.ndarray:
    return np.searchsorted(np.asarray([t1, t2]), areas, side="right")


# ---------------------------------------------------------------------------
# 主分析：单次遍历（每图每模型只解包一次）
# ---------------------------------------------------------------------------

def analyse(found: dict, gt_dir: str, gt_man: dict, *, champion: str,
            buckets: dict, log) -> dict:
    import cv2

    names = [champion] + [n for n in found if n != champion]
    for n in names:
        if n not in found:
            raise C.ContractError("模型 %r 未在导出目录中找到" % n)
    gt_stems = list(gt_man["stems"])

    # ---- 累加器 ----
    gcounts = {n: [[0, 0, 0] for _ in range(4)] for n in names}
    picounts = {n: {} for n in names}
    pair = {n: {c: _new_pair_acc() for c in C.CLASSES} for n in names if n != champion}
    pair_img = {n: {} for n in names if n != champion}
    bucket_tp = {n: {c: np.zeros(3, np.int64) for c in C.CLASSES} for n in names}
    bucket_gt_px = {c: np.zeros(3, np.int64) for c in C.CLASSES}
    t0 = time.time()

    for i, stem in enumerate(gt_stems):
        g = C.load_gt_npz(os.path.join(gt_dir, C.MASK_FILENAME % stem),
                          where="gt[%s]" % stem)
        if g["file_sha256"] != gt_man["per_image"][stem]["file_sha256"]:
            raise C.ContractError("GT[%s]: 文件 sha256 与契约不符" % stem)
        h, w = g["orig_hw"]
        npx = g["n_pixels"]
        gt2d = g["bits"].reshape(4, h, w)
        ignore2d = g["ignore"]
        # 逐类：GT 正样本的展平索引 + 连通域编号/面积（只对 GT 正像素算，
        # 代价 = GT 正像素数，而非整图，故 8 个模型复用同一份即可）
        gt_flat_idx, comp_id, comp_areas = {}, {}, {}
        gt_flat = g["bits"].reshape(4, -1)
        for ci, cname in enumerate(C.CLASSES):
            n, lab, stats, _c = cv2.connectedComponentsWithStats(
                gt2d[ci].astype(np.uint8), 8)
            sel = np.flatnonzero(gt2d[ci].reshape(-1))
            gt_flat_idx[cname] = sel
            comp_id[cname] = lab.reshape(-1)[sel] if sel.size else np.zeros(0, np.int32)
            comp_areas[cname] = (stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
                                 if n > 1 else np.zeros(0, np.int64))
            t1, t2 = buckets["per_class"][cname]["thresholds"]
            bidx = bucket_index(comp_areas[cname], t1, t2)
            bucket_gt_px[cname] += (
                np.bincount(bidx, weights=comp_areas[cname].astype(np.float64),
                            minlength=3).astype(np.int64)
                if comp_areas[cname].size else np.zeros(3, np.int64))

        champion_bits = None
        for name in names:
            m = found[name]
            mm = C.load_verified_mask(m, stem)
            if tuple(mm["orig_hw"]) != (h, w) or mm["n_pixels"] != npx:
                raise C.ContractError("%s[%s]: 掩膜形状与 GT 不一致" % (name, stem))
            bits = mm["bits"].reshape(4, h, w)
            cnt = C.counts_from_bits(bits.reshape(4, -1), gt_flat, ignore2d.reshape(-1))
            stored = m["per_image"][stem]["counts"]
            for ci, cname in enumerate(C.CLASSES):
                if [int(x) for x in stored[cname]] != cnt[ci]:
                    raise C.ContractError(
                        "%s[%s]: 契约存计数 %s=%r != 位图重算 %r"
                        % (name, stem, cname, stored[cname], cnt[ci]))
                for j in range(3):
                    gcounts[name][ci][j] += int(cnt[ci][j])
            picounts[name][stem] = {c: list(cnt[ci]) for ci, c in enumerate(C.CLASSES)}
            bits_flat = bits.reshape(4, -1)
            for ci, cname in enumerate(C.CLASSES):
                sel = gt_flat_idx[cname]
                t1, t2 = buckets["per_class"][cname]["thresholds"]
                if sel.size:
                    tp_per_comp = np.bincount(
                        comp_id[cname][bits_flat[ci][sel]],
                        minlength=comp_areas[cname].size + 1)[1:].astype(np.int64)
                    bidx = bucket_index(comp_areas[cname], t1, t2)
                    bucket_tp[name][cname] += np.bincount(
                        bidx, weights=tp_per_comp.astype(np.float64),
                        minlength=3).astype(np.int64)

            if name == champion:
                champion_bits = bits.copy()
                continue
            # ---- 逐像素互补（champion vs 该候选）----
            cb = champion_bits
            row_img = {}
            for ci, cname in enumerate(C.CLASSES):
                p = bits[ci].reshape(-1)        # candidate
                q = cb[ci].reshape(-1)          # champion
                gt = gt2d[ci].reshape(-1)
                valid = ~ignore2d.reshape(-1)
                neg = (~gt) & valid
                acc = pair[name][cname]
                common_tp = int(np.count_nonzero(p & q & gt))
                champ_only_tp = int(np.count_nonzero(q & ~p & gt))
                cand_only_tp = int(np.count_nonzero(p & ~q & gt))
                both_fn = int(np.count_nonzero(~p & ~q & gt))
                champ_only_fp = int(np.count_nonzero(q & ~p & neg))
                cand_only_fp = int(np.count_nonzero(p & ~q & neg))
                both_fp = int(np.count_nonzero(p & q & neg))
                acc["common_tp"] += common_tp
                acc["champion_only_tp"] += champ_only_tp
                acc["candidate_only_tp"] += cand_only_tp
                acc["champion_only_fp"] += champ_only_fp
                acc["candidate_only_fp"] += cand_only_fp
                acc["both_fp"] += both_fp
                acc["both_fn"] += both_fn
                # 恒等式（构造性校验：任意一项不成立说明布尔链有 bug）
                ch_tp = common_tp + champ_only_tp
                ch_fp = champ_only_fp + both_fp
                ch_fn = both_fn + cand_only_tp
                ca_tp = common_tp + cand_only_tp
                ca_fp = cand_only_fp + both_fp
                ca_fn = both_fn + champ_only_tp
                if ([ch_tp, ch_fp, ch_fn] != [int(x) for x in picounts[champion][stem][cname]]
                        or [ca_tp, ca_fp, ca_fn] != [int(x) for x in picounts[name][stem][cname]]):
                    raise C.ContractError(
                        "互补恒等式失败 %s vs %s / %s[%s]" % (champion, name, stem, cname))
                or_counts = [common_tp + champ_only_tp + cand_only_tp,
                             champ_only_fp + cand_only_fp + both_fp, both_fn]
                and_counts = [common_tp, both_fp, both_fn + champ_only_tp + cand_only_tp]
                acc["or"] = _add3(acc["or"], or_counts)
                acc["and"] = _add3(acc["and"], and_counts)
                row_img[cname] = {
                    "champion": [ch_tp, ch_fp, ch_fn],
                    "candidate": [ca_tp, ca_fp, ca_fn],
                    "or": or_counts,
                    "and": and_counts,
                    "common_tp": common_tp,
                    "champion_only_tp": champ_only_tp,
                    "candidate_only_tp": cand_only_tp,
                    "champion_only_fp": champ_only_fp,
                    "candidate_only_fp": cand_only_fp,
                    "both_fp": both_fp,
                    "both_fn": both_fn,
                }
            pair_img[name][stem] = row_img
        del champion_bits, gt2d, comp_id, comp_areas, gt_flat_idx, gt_flat
        if (i + 1) % 10 == 0 or (i + 1) == len(gt_stems):
            log("analyse %d/%d (%.1fs)" % (i + 1, len(gt_stems), time.time() - t0))

    return {"names": names, "gcounts": gcounts, "picounts": picounts,
            "pair": pair, "pair_img": pair_img,
            "bucket_tp": bucket_tp, "bucket_gt_px": bucket_gt_px}


def _new_pair_acc():
    return {k: [0, 0, 0] if k in ("or", "and") else 0
            for k in ("common_tp", "champion_only_tp", "candidate_only_tp",
                      "champion_only_fp", "candidate_only_fp", "both_fp",
                      "both_fn", "or", "and")}


def _add3(a, b):
    return [int(a[0]) + int(b[0]), int(a[1]) + int(b[1]), int(a[2]) + int(b[2])]


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def iou_of(counts):
    tp, fp, fn = (int(x) for x in counts)
    union = tp + fp + fn
    return None if union == 0 else tp / union


def summarise(res: dict, *, champion: str) -> dict:
    names = res["names"]
    model_metrics = {}
    for n in names:
        miou, per, detail = C.iou_from_counts(res["gcounts"][n])
        model_metrics[n] = {"mIoU_lesion4": miou, "per_class_iou": per,
                            "per_class_detail": detail,
                            "global_counts": {C.CLASSES[c]: list(res["gcounts"][n][c])
                                              for c in range(4)}}
    pairwise = {}
    for n in names:
        if n == champion:
            continue
        cls_out = {}
        for cname in C.CLASSES:
            a = res["pair"][n][cname]
            gt_pos = int(model_metrics[champion]["per_class_detail"][cname]["tp"]
                         + model_metrics[champion]["per_class_detail"][cname]["fn"])
            ch_tiou = iou_of([a["common_tp"] + a["champion_only_tp"],
                              a["champion_only_fp"] + a["both_fp"],
                              a["both_fn"] + a["candidate_only_tp"]])
            ca_tiou = iou_of([a["common_tp"] + a["candidate_only_tp"],
                              a["candidate_only_fp"] + a["both_fp"],
                              a["both_fn"] + a["champion_only_tp"]])
            or_iou = iou_of(a["or"])
            and_iou = iou_of(a["and"])
            uniq = a["candidate_only_tp"] + a["candidate_only_fp"]
            cls_out[cname] = {
                "gt_positive_pixels": gt_pos,
                "champion_iou": ch_tiou,
                "candidate_iou": ca_tiou,
                "candidate_replace_iou": ca_tiou,
                "delta_candidate_minus_champion": (None if (ch_tiou is None or ca_tiou is None)
                                                   else ca_tiou - ch_tiou),
                "or_iou": or_iou,
                "delta_or_minus_champion": (None if (ch_tiou is None or or_iou is None)
                                            else or_iou - ch_tiou),
                "and_iou": and_iou,
                "common_tp": a["common_tp"],
                "champion_only_tp": a["champion_only_tp"],
                "candidate_only_tp": a["candidate_only_tp"],
                "champion_only_fp": a["champion_only_fp"],
                "candidate_only_fp": a["candidate_only_fp"],
                "both_fp": a["both_fp"],
                "both_fn": a["both_fn"],
                "candidate_unique_activation_pixels": uniq,
                "candidate_unique_precision": (None if uniq == 0
                                               else a["candidate_only_tp"] / uniq),
                "added_tp_fraction_of_gt": (None if gt_pos == 0
                                            else a["candidate_only_tp"] / gt_pos),
                "lost_tp_fraction_of_gt": (None if gt_pos == 0
                                           else a["champion_only_tp"] / gt_pos),
            }
        or_miou = _mean([cls_out[c]["or_iou"] for c in C.CLASSES])
        and_miou = _mean([cls_out[c]["and_iou"] for c in C.CLASSES])
        pairwise[n] = {
            "per_class": cls_out,
            "or_mIoU_lesion4": or_miou,
            "and_mIoU_lesion4": and_miou,
            "delta_or_mIoU_minus_champion": (
                None if or_miou is None
                else or_miou - model_metrics[champion]["mIoU_lesion4"]),
            "pooled": {
            "or": _sum_pooled(res["pair"][n], "or"),
            "and": _sum_pooled(res["pair"][n], "and"),
            "common_tp": sum(res["pair"][n][c]["common_tp"] for c in C.CLASSES),
            "champion_only_tp": sum(res["pair"][n][c]["champion_only_tp"] for c in C.CLASSES),
            "candidate_only_tp": sum(res["pair"][n][c]["candidate_only_tp"] for c in C.CLASSES),
            "candidate_only_fp": sum(res["pair"][n][c]["candidate_only_fp"] for c in C.CLASSES),
            "champion_only_fp": sum(res["pair"][n][c]["champion_only_fp"] for c in C.CLASSES),
            "both_fp": sum(res["pair"][n][c]["both_fp"] for c in C.CLASSES),
            "both_fn": sum(res["pair"][n][c]["both_fn"] for c in C.CLASSES),
            },
        }
    return {"model_metrics": model_metrics, "pairwise": pairwise}


def _sum_pooled(pair_cls, key):
    tot = [0, 0, 0]
    for c in C.CLASSES:
        tot = _add3(tot, pair_cls[c][key])
    return tot


def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def direct_combination(model_metrics: dict, *, champion: str) -> dict:
    """每类取 val 上该类 IoU 最高的模型（并列优先 champion，保证确定性）。

    明确：这是 val 选择结果，不是无偏估计（见 caveats）。
    """
    picks, counts = {}, [[0, 0, 0] for _ in range(4)]
    per_class_iou, per_model_iou = {}, {}
    for ci, cname in enumerate(C.CLASSES):
        best_name, best_iou = champion, model_metrics[champion]["per_class_iou"][cname]
        order = [champion] + sorted(k for k in model_metrics if k != champion)
        for n in order:
            v = model_metrics[n]["per_class_iou"][cname]
            if v is None:
                continue
            if best_iou is None or v > best_iou + 1e-12:
                best_name, best_iou = n, v
        picks[cname] = {"model": best_name, "val_iou": best_iou}
        per_class_iou[cname] = best_iou
        for j in range(3):
            counts[ci][j] = int(model_metrics[best_name]["global_counts"][cname][j])
        per_model_iou[cname] = {n: model_metrics[n]["per_class_iou"][cname]
                                for n in model_metrics}
    miou = _mean(list(per_class_iou.values()))
    return {
        "note": "每类在 val 上逐类 argmax 选模型（并列优先 champion）；"
                "这是选择偏差样本，不是无偏估计",
        "picks": picks,
        "per_class_iou": per_class_iou,
        "mIoU_lesion4": miou,
        "delta_vs_champion": (None if miou is None
                              else miou - model_metrics[champion]["mIoU_lesion4"]),
        "global_counts": {C.CLASSES[c]: list(counts[c]) for c in range(4)},
        "all_models_per_class_iou": per_model_iou,
        "distinct_models_used": sorted({v["model"] for v in picks.values()}),
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _csv_text(header, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def write_outputs(out_dir: str, *, res, summ, buckets, direct, models_info,
                  gt_man, champion, skipped, log) -> dict:
    C.ensure_dir(out_dir)
    names = res["names"]
    r6 = lambda x: None if x is None else round(float(x), 6)

    # ---- per-class CSV ----
    header = ["pair", "class", "gt_positive_pixels", "champion_iou", "candidate_iou",
              "delta_candidate_minus_champion", "or_iou", "delta_or_minus_champion",
              "and_iou", "candidate_replace_iou", "common_tp", "champion_only_tp",
              "candidate_only_tp", "candidate_only_fp", "champion_only_fp",
              "both_fp", "both_fn", "candidate_unique_activation_pixels",
              "candidate_unique_precision", "added_tp_fraction_of_gt",
              "lost_tp_fraction_of_gt"]
    rows = []
    for n in names:
        if n == champion:
            continue
        for cname in C.CLASSES:
            e = summ["pairwise"][n]["per_class"][cname]
            rows.append([n, cname, e["gt_positive_pixels"], r6(e["champion_iou"]),
                         r6(e["candidate_iou"]),
                         r6(e["delta_candidate_minus_champion"]), r6(e["or_iou"]),
                         r6(e["delta_or_minus_champion"]), r6(e["and_iou"]),
                         r6(e["candidate_replace_iou"]), e["common_tp"],
                         e["champion_only_tp"], e["candidate_only_tp"],
                         e["candidate_only_fp"], e["champion_only_fp"],
                         e["both_fp"], e["both_fn"],
                         e["candidate_unique_activation_pixels"],
                         r6(e["candidate_unique_precision"]),
                         r6(e["added_tp_fraction_of_gt"]),
                         r6(e["lost_tp_fraction_of_gt"])])
    C.atomic_write_text(os.path.join(out_dir, "expert_complementarity_per_class.csv"),
                        _csv_text(header, rows))

    # ---- per-image CSV ----
    header = ["pair", "stem", "class", "orig_h", "orig_w",
              "champion_tp", "champion_fp", "champion_fn",
              "candidate_tp", "candidate_fp", "candidate_fn",
              "or_tp", "or_fp", "or_fn", "and_tp", "and_fp", "and_fn",
              "champion_iou", "candidate_iou", "delta_iou", "or_iou", "delta_or",
              "common_tp", "champion_only_tp", "candidate_only_tp",
              "candidate_only_fp"]
    rows = []
    for n in names:
        if n == champion:
            continue
        for stem in gt_man["stems"]:
            hw = found_hw(gt_man, stem)
            for cname in C.CLASSES:
                e = res["pair_img"][n][stem][cname]
                ch, ca, orc, andc = e["champion"], e["candidate"], e["or"], e["and"]
                ch_iou, ca_iou = iou_of(ch), iou_of(ca)
                or_iou = iou_of(orc)
                rows.append([n, stem, cname, hw[0], hw[1],
                             ch[0], ch[1], ch[2], ca[0], ca[1], ca[2],
                             orc[0], orc[1], orc[2], andc[0], andc[1], andc[2],
                             r6(ch_iou), r6(ca_iou),
                             r6(None if (ch_iou is None or ca_iou is None) else ca_iou - ch_iou),
                             r6(or_iou),
                             r6(None if (ch_iou is None or or_iou is None) else or_iou - ch_iou),
                             e["common_tp"], e["champion_only_tp"],
                             e["candidate_only_tp"], e["candidate_only_fp"]])
    C.atomic_write_text(os.path.join(out_dir, "expert_complementarity_per_image.csv"),
                        _csv_text(header, rows))

    # ---- area buckets CSV ----
    header = (["class", "bucket", "area_lo", "area_hi", "n_components", "gt_pixels",
               "gt_pixel_fraction"]
              + ["%s_recall" % n for n in names])
    rows = []
    for cname in C.CLASSES:
        b = buckets["per_class"][cname]
        total = max(int(b["gt_positive_pixels"]), 1)
        for bi, bk in enumerate(b["buckets"]):
            gp = int(res["bucket_gt_px"][cname][bi])
            row = [cname, bk["name"], bk["area_lo"], bk["area_hi"],
                   bk["n_components"], gp, r6(gp / total)]
            for n in names:
                row.append(r6(None if gp == 0
                              else int(res["bucket_tp"][n][cname][bi]) / gp))
            rows.append(row)
    C.atomic_write_text(os.path.join(out_dir, "expert_complementarity_area_buckets.csv"),
                        _csv_text(header, rows))

    payload = {
        "schema": SCHEMA,
        "split": C.SPLIT,
        "champion": champion,
        "models": names,
        "skipped_models": skipped,
        "n_images": len(gt_man["stems"]),
        "input_contracts": models_info,
        "gt": {
            "gt_contract_sha256": gt_man["gt_contract_sha256"],
            "gt_semantic_fingerprint": gt_man["gt_semantic_fingerprint"],
            "stems": gt_man["stems"],
            "label_files_sha256": gt_man["label_files_sha256"],
        },
        "gt_global_positive_pixels": {
            c: int(buckets["per_class"][c]["gt_positive_pixels"]) for c in C.CLASSES},
        "model_metrics": summ["model_metrics"],
        "pairwise": summ["pairwise"],
        "area_buckets": buckets,
        "area_bucket_recall": {
            n: {c: [None if int(res["bucket_gt_px"][c][b]) == 0
                    else int(res["bucket_tp"][n][c][b]) / int(res["bucket_gt_px"][c][b])
                    for b in range(3)] for c in C.CLASSES}
            for n in names},
        "direct_four_class_combination": direct,
        "caveats": CAVEATS,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    C.atomic_write_json(os.path.join(out_dir, "expert_complementarity.json"), payload)
    C.atomic_write_text(os.path.join(out_dir, "EXPERT_COMPLEMENTARITY.md"),
                        markdown_report(payload, buckets, direct, summ, res,
                                        champion=champion, models_info=models_info))
    log("输出目录: %s" % out_dir)
    return payload


def found_hw(gt_man, stem):
    return tuple(int(x) for x in gt_man["per_image"][stem]["orig_hw"])


def markdown_report(payload, buckets, direct, summ, res, *, champion, models_info) -> str:
    names = payload["models"]
    pct = lambda x: "—" if x is None else "%.4f" % (100.0 * x)
    spct = lambda x: "—" if x is None else "%+.4f" % (100.0 * x)
    L = []
    L.append("# 逐像素专家互补性分析（DDR %s，n=%d）\n" % (C.SPLIT, payload["n_images"]))
    L.append("生成时间：%s · 冠军口径：`%s` · 图像数：%d · 通道序 EX/HE/SE/MA · "
             "严格 `p > 0.5`\n" % (payload["created_at"], champion, payload["n_images"]))
    L.append("输入契约：`gt_contract_sha256=%s…` · `gt_semantic_fingerprint=%s…`\n"
             % (payload["gt"]["gt_contract_sha256"][:16],
                payload["gt"]["gt_semantic_fingerprint"][:16]))
    if payload["skipped_models"]:
        L.append("\n> **注意**：以下模型被显式跳过（--allow-missing）：%s\n"
                 % ", ".join(payload["skipped_models"]))

    L.append("\n## 1. 各模型 val 指标（位图重算，非直接引用）\n")
    L.append("| 模型 | family | mIoU | EX | HE | SE | MA | TP/FP/FN 总 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n in names:
        mm = payload["model_metrics"][n]
        d = mm["per_class_detail"]
        tot = {k: sum(d[c][k] for c in C.CLASSES) for k in ("tp", "fp", "fn")}
        L.append("| %s | %s | %s | %s | %s | %s | %s | %d/%d/%d |" % (
            n, models_info[n]["family"], pct(mm["mIoU_lesion4"]),
            *[pct(mm["per_class_iou"][c]) for c in C.CLASSES],
            tot["tp"], tot["fp"], tot["fn"]))
    L.append("\n（TP/FP/FN 为四类之和；逐类见 JSON/CSV。）")

    L.append("\n## 2. 与冠军的逐像素互补（每格：候选 IoU / Δ候选 / OR IoU / ΔOR / 候选独有 TP / 候选独有 FP）\n")
    L.append("| 候选 | EX | HE | SE | MA | OR mIoU | ΔOR mIoU |")
    L.append("|---|---|---|---|---|---|---|")
    for n in names:
        if n == champion:
            continue
        row = ["| %s" % n]
        for c in C.CLASSES:
            e = payload["pairwise"][n]["per_class"][c]
            row.append("| %s / %+.4f / %s / %s / %d / %d" % (
                pct(e["candidate_iou"]), 100.0 * e["delta_candidate_minus_champion"],
                pct(e["or_iou"]), pct(e["delta_or_minus_champion"]),
                e["candidate_only_tp"], e["candidate_only_fp"]))
        row.append("| %s" % pct(payload["pairwise"][n]["or_mIoU_lesion4"]))
        row.append("| %s |" % spct(payload["pairwise"][n]["delta_or_mIoU_minus_champion"]))
        L.append(" ".join(row))

    L.append("\n## 3. 按 GT 连通域面积粗分的小/中/大召回\n")
    L.append("阈值模式：`%s`（逐类阈值见 CSV/JSON）。召回 = 该桶 GT 像素被检出的比例。\n"
             % buckets["mode"])
    L.append("| 类别 | 桶 | 面积范围 | 连通域数 | GT 像素 | " + " | ".join(names) + " |")
    L.append("|---|---|---|---|---|" + "---|" * len(names))
    for c in C.CLASSES:
        b = buckets["per_class"][c]
        for bi, bk in enumerate(b["buckets"]):
            gp = int(res["bucket_gt_px"][c][bi])
            rng = "%s..%s" % (bk["area_lo"], "∞" if bk["area_hi"] is None else bk["area_hi"])
            cells = [("%s" % pct(None if gp == 0
                                 else int(res["bucket_tp"][n][c][bi]) / gp))
                     for n in names]
            L.append("| %s | %s | %s | %d | %d | %s |"
                     % (c, bk["name"], rng, bk["n_components"], gp, " | ".join(cells)))

    L.append("\n## 4. 直接四类专家组合（仅基于 val）\n")
    L.append("| 类别 | 选中模型 | 该类 val IoU |")
    L.append("|---|---|---|")
    for c in C.CLASSES:
        p = direct["picks"][c]
        L.append("| %s | %s | %s |" % (c, p["model"], pct(p["val_iou"])))
    L.append("\n组合 mIoU = **%s**（Δ vs 冠军 = %s），使用到的模型：%s。\n"
             % (pct(direct["mIoU_lesion4"]), pct(direct["delta_vs_champion"]),
                ", ".join(direct["distinct_models_used"])))
    L.append("\n> %s\n" % direct["note"])

    L.append("\n## 5. 结论与限制\n")
    for c in CAVEATS:
        L.append("- %s" % c)
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="val 逐像素专家互补性分析（严格 p>0.5、逐像素布尔运算；test 一律拒绝）")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--champion", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--allow-missing", action="append", default=[],
                    help="允许缺失的模型名（会被写进输出的 skipped_models）")
    ap.add_argument("--area-mode", default=None, choices=("tertile", "absolute"))
    ap.add_argument("--area-thresholds", nargs=2, type=int, default=None,
                    metavar=("SMALL_MAX", "MEDIUM_MAX"))
    args = ap.parse_args()
    log = C.log

    raw = load_settings(args.settings)
    out_root = resolve_out_root(raw, args.out_root)
    ana = raw.get("analysis") or {}
    champion = args.champion or ana.get("champion") or DEFAULT_CHAMPION
    models = list(args.models or ana.get("models") or DEFAULT_MODELS)
    if champion not in models:
        models = [champion] + models
    area_mode = args.area_mode or ana.get("area_mode") or "tertile"
    area_abs = args.area_thresholds or ana.get("area_thresholds")
    out_dir = C.safe_path(
        args.out_dir or ana.get("out_dir") or os.path.join(out_root, "analysis"),
        "out_dir")
    C.assert_val_only(C.SPLIT, paths=(out_root, out_dir), what="analyse")

    gt_dir = os.path.join(out_root, "_shared", "gt")
    gt_man = C.load_gt_contract(gt_dir)
    C.assert_val_only(C.SPLIT, n_images=gt_man["contract"]["n_images"], what="gt contract")
    log("共享 GT 校验通过：%d 张，fingerprint=%s…"
        % (gt_man["contract"]["n_images"],
           gt_man["contract"]["gt_semantic_fingerprint"][:12]))

    found, missing = discover_models(out_root, models, allow_missing=args.allow_missing)
    if len(found) < 2:
        raise C.ContractError("至少需要 2 个模型（含冠军）才能做互补分析，当前 %d 个"
                              % len(found))
    models_info = cross_check(found, gt_man["contract"])
    log("模型契约校验通过：%s" % sorted(found))

    buckets = gt_bucket_thresholds(gt_dir, gt_man["contract"], mode=area_mode,
                                   absolute=area_abs, log=log)
    res = analyse(found, gt_dir, gt_man["contract"], champion=champion,
                  buckets=buckets, log=log)
    summ = summarise(res, champion=champion)
    direct = direct_combination(summ["model_metrics"], champion=champion)
    payload = write_outputs(out_dir, res=res, summ=summ, buckets=buckets,
                            direct=direct, models_info=models_info,
                            gt_man=gt_man["contract"], champion=champion,
                            skipped=missing, log=log)

    ch_miou = payload["model_metrics"][champion]["mIoU_lesion4"]
    log("冠军 %s mIoU=%.6f" % (champion, ch_miou))
    for n, e in payload["pairwise"].items():
        cand_miou = payload["model_metrics"][n]["mIoU_lesion4"]
        log("  vs %-10s mIoU=%s (Δ=%s)  OR mIoU=%s (Δ=%s)"
            % (n,
               "—" if cand_miou is None else "%.6f" % cand_miou,
               "—" if cand_miou is None else "%+.6f" % (cand_miou - ch_miou),
               "—" if e["or_mIoU_lesion4"] is None else "%.6f" % e["or_mIoU_lesion4"],
               "—" if e["delta_or_mIoU_minus_champion"] is None
               else "%+.6f" % e["delta_or_mIoU_minus_champion"]))
    log("直接四类组合 mIoU=%s（val 选择，非无偏）"
        % payload["direct_four_class_combination"]["mIoU_lesion4"])
    return 0


def main_entry():
    try:
        return main()
    except C.PreflightBlocked as e:
        C.log("BLOCKED: %s" % e)
        return 3
    except C.ContractError as e:
        C.log("CONTRACT FAILURE: %s" % e)
        return 2
    except BaseException:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main_entry())
