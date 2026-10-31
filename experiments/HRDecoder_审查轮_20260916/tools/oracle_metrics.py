#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立指标 oracle（prompt §8.1）+ ignore 行为实证（§2 的 S11）。

两条实现严格分离：

1. `oracle_*` —— **独立实现**。用最直接的逐像素集合计数（TP/FP/FN 各自独立
   统计），不 import、不调用 `lesion_metric.py` 的任何函数，也不复用它的
   "累加 P/TP/FN 再相除" 的代数路径。这是所谓"oracle"。

2. `legacy_replica_*` —— **忠实复刻**被测实现的计算路径（含
   `label = raw_label == i`、`P = Σ(pred)`、忽略 ignore_index 的行为），
   用于把"ignore 未生效"这一静态推断变成可复现的实验事实。

设计约束（prompt §4.1）：
  - 必检用例集合固定；缺一个不是 PASS。
  - 任何断言失败 / 异常 → 非零退出码。
  - JSON 只写有限数值；不输出 NaN/Inf 冒充合法结果。
  - 失败报告与成功报告文件名不同，不混名。
  - 临时文件 + 原子替换。

用法：
  python oracle_metrics.py --out-dir <audit_runs/<run_id>>/tests
"""
import argparse
import json
import math
import os
import sys

import numpy as np

CLASSES = ["EX", "HE", "SE", "MA"]   # 通道 0..3 ↔ 标签值 1..4
BG_LABEL = 0
IGNORE_LABEL = 255
THRESH = 0.5                          # 冻结值；比较符为严格 >
# 必检用例集合（缺任何一个 -> 非 PASS）
REQUIRED_CASES = ["S%d" % i for i in range(1, 13)]


# ----------------------------------------------------------------------------
# 独立 oracle：直接集合计数，与被测的代数路径无关
# ----------------------------------------------------------------------------
def oracle_counts(probs, gts):
    """probs: list of (4,H,W) float; gts: list of (H,W) int.
    返回 {class: (TP, FP, FN)}，直接用布尔集合计数。"""
    tp = {c: 0 for c in CLASSES}
    fp = {c: 0 for c in CLASSES}
    fn = {c: 0 for c in CLASSES}
    for p, g in zip(probs, gts):
        assert p.shape[0] == len(CLASSES), p.shape
        assert p.shape[1:] == g.shape, (p.shape, g.shape)
        for ci, cname in enumerate(CLASSES):
            pos = p[ci] > THRESH            # 严格 >
            gt_pos = (g == (ci + 1))
            gt_ig = (g == IGNORE_LABEL)
            tp[cname] += int(np.sum(pos & gt_pos))
            fp[cname] += int(np.sum(pos & ~gt_pos & ~gt_ig))
            fn[cname] += int(np.sum(~pos & gt_pos))
    return tp, fp, fn


def oracle_iou(tp, fp, fn):
    out = {}
    for c in CLASSES:
        den = tp[c] + fp[c] + fn[c]
        out[c] = (tp[c] / den) if den > 0 else float("nan")
    return out


def oracle_miou(probs, gts):
    tp, fp, fn = oracle_counts(probs, gts)
    iou = oracle_iou(tp, fp, fn)
    vals = [0.0 if (isinstance(iou[c], float) and math.isnan(iou[c])) else iou[c]
            for c in CLASSES]
    return sum(vals) / 4.0, iou, (tp, fp, fn)


# ----------------------------------------------------------------------------
# 忠实复刻被测实现（用于 S11；不是 oracle）
# ----------------------------------------------------------------------------
def legacy_replica_counts(probs, gts, num_classes=5):
    """复刻 lesion_metric.sigmoid_confused_matrix + sigmoid_metrics 的算法：
       累加 P = Σ(pred)、TP、FN，然后 IoU = TP / (P + FN)。
       注意：完全忽略 ignore_index（真实实现就是如此）。"""
    total_p = np.zeros((num_classes,), dtype=np.float64)
    total_tp = np.zeros((num_classes,), dtype=np.float64)
    total_fn = np.zeros((num_classes,), dtype=np.float64)
    for p, raw_label in zip(probs, gts):
        for i in range(1, num_classes):
            pred = p[i - 1] > THRESH
            label = raw_label == i
            total_tp[i] += np.sum(label & pred)
            total_p[i] += np.sum(pred)
            total_fn[i] += np.sum(label) - np.sum(label & pred)
    return total_p, total_tp, total_fn


def legacy_replica_iou(probs, gts, num_classes=5):
    P, TP, FN = legacy_replica_counts(probs, gts, num_classes)
    with np.errstate(invalid="ignore"):
        iou = TP / (P + FN)
    return iou[1:5], (P[1:5], TP[1:5], FN[1:5])   # 去掉背景，对齐 EX..MA


# ----------------------------------------------------------------------------
# 用例
# ----------------------------------------------------------------------------
def mk(h, w, fill=0.0):
    return np.full((len(CLASSES), h, w), fill, dtype=np.float64)


def build_cases():
    """返回 {case_id: (probs, gts, checker)}，checker(miou, iou, counts) -> (ok, detail)"""
    C = {}

    # S1: TP=1, FP=1, FN=1  -> IoU = 1/3 （EX 类）
    # 形状 (H=2, W=3)；p 是 (C=4, H, W)，g 是 (H, W)
    p = mk(2, 3); g = np.zeros((2, 3), dtype=np.int64)
    p[0, 0, 0] = 0.9           # 预测阳性 & GT 阳性 -> TP
    p[0, 0, 1] = 0.9           # 预测阳性 & GT 阴性 -> FP
    g[0, 0] = 1
    g[1, 0] = 1                # GT 阳性 & 预测阴性 -> FN
    C["S1"] = ([p], [g], lambda m, i, c: (abs(i["EX"] - 1 / 3) < 1e-12, i["EX"]))

    # S2: 图1 TP=1,FP=0,FN=0；图2 TP=0,FP=9,FN=0 -> 累计 1/10（不是逐图平均 1/2）
    p1 = mk(1, 1); g1 = np.zeros((1, 1), dtype=np.int64); p1[0, 0, 0] = 0.9; g1[0, 0] = 1
    p2 = mk(3, 3); g2 = np.zeros((3, 3), dtype=np.int64); p2[0, :, :] = 0.9
    C["S2"] = ([p1, p2], [g1, g2],
               lambda m, i, c: (abs(i["EX"] - 0.1) < 1e-12, i["EX"]))

    # S3: GT 有阳性、预测全阴性 -> IoU = 0
    p = mk(2, 2); g = np.zeros((2, 2), dtype=np.int64); g[:, :] = 1
    C["S3"] = ([p], [g], lambda m, i, c: (i["EX"] == 0.0, i["EX"]))

    # S4: GT 全阴性、预测有阳性 -> IoU = 0
    p = mk(2, 2); p[0, :, :] = 0.9; g = np.zeros((2, 2), dtype=np.int64)
    C["S4"] = ([p], [g], lambda m, i, c: (i["EX"] == 0.0, i["EX"]))

    # S5: GT 与预测全阴性 -> 分母 0 -> 冻结规则记 0
    p = mk(2, 2); g = np.zeros((2, 2), dtype=np.int64)
    C["S5"] = ([p], [g],
               lambda m, i, c: (math.isnan(i["EX"]) and m == 0.0, (i["EX"], m)))

    # S6: 单个 MA 像素且预测完全正确 -> MA IoU = 1
    p = mk(2, 2); g = np.zeros((2, 2), dtype=np.int64)
    p[3, 1, 1] = 0.9; g[1, 1] = 4
    C["S6"] = ([p], [g], lambda m, i, c: (i["MA"] == 1.0, i["MA"]))

    # S7: 同一像素两个独立通道均为阳性 -> 两通道各自计数（多标签）
    p = mk(1, 1); g = np.zeros((1, 1), dtype=np.int64)
    p[0, 0, 0] = 0.9; p[1, 0, 0] = 0.9; g[0, 0] = 1
    C["S7"] = ([p], [g],
               lambda m, i, c: (i["EX"] == 1.0 and i["HE"] == 0.0, (i["EX"], i["HE"])))

    # S8: 概率恰好 0.5 -> 判阴（严格 >）
    p = mk(1, 1); p[0, 0, 0] = 0.5; g = np.zeros((1, 1), dtype=np.int64); g[0, 0] = 1
    C["S8"] = ([p], [g], lambda m, i, c: (i["EX"] == 0.0, i["EX"]))

    # S9: 0.5±eps
    # pe 形状 (C=4, H=1, W=2)；g 形状 (H=1, W=2)
    #   像素(0,0): pred=0.5+eps > 0.5 -> 阳；gt=1 -> TP
    #   像素(0,1): pred=0.5-eps > 0.5 -> 阴；gt=1 -> FN
    # 故 TP=1, FP=0, FN=1 -> IoU = 1/(1+0+1) = 0.5
    pe = mk(1, 2); pe[0, 0, 0] = 0.5 + 1e-6; pe[0, 0, 1] = 0.5 - 1e-6
    g = np.zeros((1, 2), dtype=np.int64); g[0, 0] = 1; g[0, 1] = 1
    C["S9"] = ([pe], [g],
               lambda m, i, c: (abs(i["EX"] - 0.5) < 1e-12
                                and c[0]["EX"] == 1 and c[1]["EX"] == 0
                                and c[2]["EX"] == 1, i["EX"]))

    # S10: 四类分别只在四个不同位置阳性 -> 四类 IoU 均为 1
    p = mk(4, 4); g = np.zeros((4, 4), dtype=np.int64)
    for k in range(4):
        p[k, k, k] = 0.9; g[k, k] = k + 1
    C["S10"] = ([p], [g],
                lambda m, i, c: (all(i[x] == 1.0 for x in CLASSES), i))

    # S11: ignore(255) 位置改变预测 -> oracle: TP/FN 不变、FP 不计；
    #      legacy replica: FP 计入 -> 两者必然不同
    p_a = mk(3, 3); p_b = mk(3, 3)
    g = np.zeros((3, 3), dtype=np.int64)
    g[0, 0] = 1                    # 真阳性像素
    p_a[0, 0, 0] = 0.9; p_b[0, 0, 0] = 0.9
    g[1, 1] = IGNORE_LABEL         # ignore 像素
    p_a[0, 1, 1] = 0.1             # 阴性
    p_b[0, 1, 1] = 0.9             # 阳性（在 ignore 区）
    C["S11"] = ([p_a, p_b], [g, g], None)   # 特殊判定，见 run_case

    # S12: 图像顺序重排 -> 累计计数不变
    p1 = mk(1, 1); g1 = np.zeros((1, 1), dtype=np.int64); p1[0, 0, 0] = 0.9; g1[0, 0] = 1
    p2 = mk(1, 1); g2 = np.zeros((1, 1), dtype=np.int64); p2[0, 0, 0] = 0.9
    C["S12"] = ([p1, p2], [g1, g2], None)   # 特殊判定，见 run_case
    return C


def _finite(x):
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, float):
        return x if math.isfinite(x) else None      # NaN/Inf -> null，不冒充数值
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cases = build_cases()
    results = []
    ok_ids, fail_ids, missing = [], [], []

    for cid in REQUIRED_CASES:
        if cid not in cases:
            missing.append(cid)
            results.append({"test_id": cid, "required": True,
                            "status": "NOT_RUN", "reason": "case not implemented"})
            continue
        probs, gts, checker = cases[cid]
        rec = {"test_id": cid, "required": True, "input_case": cid}
        try:
            if cid == "S11":
                # 三次判定：(a) oracle 对 a/b 的计数是否不变（TP/FN 不变、FP 不计 ignore）
                tpA, fpA, fnA = oracle_counts([probs[0]], [gts[0]])
                tpB, fpB, fnB = oracle_counts([probs[1]], [gts[1]])
                oracle_invariant = (tpA == tpB and fnA == fnB)
                # (b) legacy replica 的 IoU 是否随 ignore 区预测变化
                iouA, _ = legacy_replica_iou([probs[0]], [gts[0]])
                iouB, _ = legacy_replica_iou([probs[1]], [gts[1]])
                legacy_changes = not np.allclose(iouA, iouB)
                rec["expected"] = {
                    "oracle_counts_invariant": True,
                    "legacy_replica_changes_when_ignore_prediction_changes": True,
                    "interpretation": ("若 legacy 变化而 oracle 不变，"
                                       "则证实 ignore_index 未生效（§2）")}
                rec["actual"] = {
                    "oracle_TP_FP_FN_A": _finite({"TP": tpA, "FP": fpA, "FN": fnA}),
                    "oracle_TP_FP_FN_B": _finite({"TP": tpB, "FP": fpB, "FN": fnB}),
                    "legacy_iou_A": _finite({c: float(v) for c, v in zip(CLASSES, iouA)}),
                    "legacy_iou_B": _finite({c: float(v) for c, v in zip(CLASSES, iouB)}),
                    "oracle_counts_invariant": oracle_invariant,
                    "legacy_changes": bool(legacy_changes)}
                rec["status"] = "PASS" if (oracle_invariant and legacy_changes) else "FAIL"
                if rec["status"] == "FAIL":
                    rec["note"] = ("legacy 未随 ignore 区预测变化，"
                                   "则 §2 的静态推断需要重新审查")
            elif cid == "S12":
                m_ab, _, c_ab = oracle_miou(probs, gts)
                m_ba, _, c_ba = oracle_miou(list(reversed(probs)), list(reversed(gts)))
                same = (c_ab == c_ba) and abs(m_ab - m_ba) < 1e-15
                rec["expected"] = {"order_invariant": True}
                rec["actual"] = {"counts_ab": _finite(c_ab), "counts_ba": _finite(c_ba),
                                 "miou_ab": m_ab, "miou_ba": m_ba,
                                 "order_invariant": same}
                rec["status"] = "PASS" if same else "FAIL"
            else:
                m, iou, counts = oracle_miou(probs, gts)
                ok, detail = checker(m, iou, counts)
                rec["expected"] = {k: (None if not math.isfinite(float(v)) else v)
                                   for k, v in iou.items()}
                rec["actual"] = {"miou": m,
                                 "iou": _finite(dict(iou)),
                                 "counts_TP_FP_FN": _finite(
                                     {"TP": counts[0], "FP": counts[1], "FN": counts[2]}),
                                 "checker_detail": _finite(detail)}
                rec["status"] = "PASS" if ok else "FAIL"
        except Exception as e:
            rec["status"] = "FAIL"
            rec["exception"] = "%s: %s" % (type(e).__name__, e)
        results.append(rec)

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_notrun = sum(1 for r in results if r["status"] == "NOT_RUN")
    overall = "PASS" if (n_fail == 0 and n_notrun == 0
                         and not missing) else "FAIL"

    out = {
        "suite": "oracle_metrics",
        "required_cases": REQUIRED_CASES,
        "n_required": len(REQUIRED_CASES),
        "n_pass": n_pass, "n_fail": n_fail, "n_not_run": n_notrun,
        "missing_cases": missing,
        "frozen_constants": {"thresh": THRESH, "comparison": "strict_>",
                            "classes": CLASSES, "aggregation": "dataset_cumulative",
                            "background_in_mean": False,
                            "empty_union": "nan->0"},
        "overall_status": overall,
        "results": results,
    }

    name = "loss_ignore_metrics.json" if overall == "PASS" \
        else "loss_ignore_metrics.FAILED.json"
    p = os.path.join(args.out_dir, name)
    # 清掉对向名字的历史报告，避免"成功/失败"两份同时存在造成误读
    other = os.path.join(args.out_dir,
                         "loss_ignore_metrics.FAILED.json"
                         if overall == "PASS" else "loss_ignore_metrics.json")
    if os.path.isfile(other):
        os.remove(other)
        print("  (已清理上一轮的对向报告: %s)" % os.path.basename(other))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_finite(out), f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)

    print("== oracle 用例 ==")
    for r in results:
        mark = {"PASS": "✓", "FAIL": "✗", "NOT_RUN": "-"}[r["status"]]
        extra = ""
        if r["test_id"] == "S11" and r["status"] in ("PASS", "FAIL"):
            a = r["actual"]
            extra = ("  oracle不变=%s  legacy随ignore变化=%s"
                     % (a["oracle_counts_invariant"], a["legacy_changes"]))
        print("  %s %-4s %s%s" % (mark, r["test_id"], r["status"], extra))
        if r["status"] == "FAIL" and "checker_detail" in r.get("actual", {}):
            print("        期望 %s / 实际 %s"
                  % (r.get("expected"), r["actual"]["checker_detail"]))
        if "exception" in r:
            print("        异常:", r["exception"])
    print("\n  %d/%d PASS overall=%s" % (n_pass, len(REQUIRED_CASES), overall))
    print("  写出:", p)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
