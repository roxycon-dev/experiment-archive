#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1 阶段：本轮所有臂的汇总 + 配对显著性检验 + 结论表（只读产物，不训练）。

输入（全部是各臂已经落盘的产物，不存在就跳过并标注缺失）：
  * T 线（train_T0T1.py 补丁版）：$RUN/val_curve_{variant}_s{seed}.json
                                    $RUN/val_counts_{variant}_s{seed}_it{iters}.json（逐图 TP/FP/FN）
  * V1：$RUN/V1/s{seed}_{variant}/ 下的同名产物
  * E1：$RUN/E1/ 下的同名产物
  * J 线（mmseg 训练）：workdir/{J0,J1}/*.log.json 里的 mIoU 记录（**无逐图计数**）
  * E1 判定门：$RUN/experiments/E1_gate.json

输出：
  * ROUND_SUMMARY.md —— 人类可读结论表
  * round_summary.json —— 机器可读

统计方法（与 e1_gate.py / R0 完全一致，不新造）：
  图像级配对 bootstrap，2000 次，seed 20260917，两臂共用重采样索引，
  先累计 TP/FP/FN 再求 IoU，95% 百分位区间，空并集记 0。
  ⚠ 该区间刻画的是"图像级不确定性"（给定这两个模型），**不是 seed 不确定性**；
    seed 不确定性只能由 V1 的 3 seed 离散度描述 —— 报告里分开列。
"""
import argparse
import glob
import json
import os

import numpy as np

B = 2000
SEED = 20260917


def iou(t):
    den = t[0] + t[1] + t[2]
    return t[0] / den if den else 0.0


def load_counts(p):
    d = json.load(open(p))
    rows = d["per_image"]
    return d, rows


def agg(rows, idx, tag):
    tp = fp = fn = 0
    for i in idx:
        a, b, c = rows[i][tag]
        tp += a; fp += b; fn += c
    return tp, fp, fn


def paired_bootstrap(ra, rb, tag="head"):
    """两臂逐图计数配对比 IoU：返回 (点估计Δ, CI95, 单侧5%分位, 两侧点估计)。"""
    n = min(len(ra), len(rb))
    ia, ib = iou(agg(ra, range(n), tag)), iou(agg(rb, range(n), tag))
    rng = np.random.RandomState(SEED)
    dl = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        dl[b] = iou(agg(rb, idx, tag)) - iou(agg(ra, idx, tag))
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"iou_a": ia, "iou_b": ib, "delta": ib - ia,
            "ci95": [float(lo), float(hi)], "crosses_zero": bool(lo <= 0 <= hi),
            "q05_one_sided": float(np.percentile(dl, 5.0)), "n": n}


def within_file_bootstrap(rows, tag_a="base0", tag_b="head"):
    """同一文件内的自对照：head 相对 base0 的提升是否显著（配对，共用图像索引）。

    这是每个臂最该先回答的问题 —— "这个小头/这条改造到底有没有把 base0 抬起来"，
    与跨臂对比（T1−T0）是两回事，不能混。
    """
    n = len(rows)
    ia, ib = iou(agg(rows, range(n), tag_a)), iou(agg(rows, range(n), tag_b))
    rng = np.random.RandomState(SEED)
    dl = np.empty(B)
    for b in range(B):
        idx = rng.randint(0, n, n)
        dl[b] = iou(agg(rows, idx, tag_b)) - iou(agg(rows, idx, tag_a))
    lo, hi = np.percentile(dl, [2.5, 97.5])
    return {"iou_base0": ia, "iou_head": ib, "delta": ib - ia,
            "ci95": [float(lo), float(hi)], "crosses_zero": bool(lo <= 0 <= hi),
            "q05_one_sided": float(np.percentile(dl, 5.0)), "n": n}


def find_counts(root, variant, seed, iters):
    """只接受**精确 iter** 的产物。

    ⚠ 2026-09-17 子代理审计：初版会在找不到 it8000 时回落到 it*（取最高），
    于是 T1 尚未跑完时会静默拿 it2000 去和 T0 的 it8000 对比 —— 把不同训练阶段
    当成同阶段配对，结论直接错。递归 glob 只为发现 V1/E1 子目录，不再放宽 iter。
    """
    g = sorted(glob.glob(os.path.join(
        root, "**", "val_counts_%s_s%d_it%d.json" % (variant, seed, iters)), recursive=True))
    return g[-1] if g else None


def find_curve(root, variant, seed):
    g = sorted(glob.glob(os.path.join(root, "**", "val_curve_%s_s%d.json" % (variant, seed)),
                         recursive=True))
    return g[-1] if g else None


def parse_mmseg_logjson(workdir):
    """从 mmseg 的 *.log.json 抽出 mIoU 曲线（J 线用）。"""
    out = []
    for p in sorted(glob.glob(os.path.join(workdir, "*.log.json"))):
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "mIoU" in d:
                out.append({"iter": d.get("iter"), "mIoU": d["mIoU"],
                            "aAcc": d.get("aAcc"), "src": os.path.basename(p)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/root/autodl-tmp/audit/audit_20260916_A",
                    help="T/E1/V1 线的产物目录")
    ap.add_argument("--j-workdir", default="/root/box1_sync/workdir",
                    help="J 线的 mmseg workdir 根")
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    R = {}
    # ---------- 1) T 线：val 曲线 ----------
    arms = [("T0", 1), ("T1", 1), ("U0", 1)]
    for s in (2, 3):
        arms += [("T0", s), ("T1", s)]
    curves = {}
    for v, s in arms:
        p = find_curve(a.run_dir, v, s)
        if p:
            c = json.load(open(p))
            curves["%s_s%d" % (v, s)] = {"path": p, "curve": c.get("curve", []),
                                         "iters": c.get("iters")}
    # E1 续训臂
    for v in ("T0", "T1"):
        p = find_curve(os.path.join(a.run_dir, "E1"), v, 1)
        if p:
            c = json.load(open(p))
            curves["E1_%s_s1" % v] = {"path": p, "curve": c.get("curve", []),
                                      "iters": c.get("iters")}
    R["val_curves"] = {k: {"iters": val["iters"],
                           "points": [(h["iter"], h["base0_MA_IoU"], h["head_MA_IoU"])
                                      for h in val["curve"]]}
                       for k, val in curves.items()}

    # ---------- 2) 配对 bootstrap（有逐图计数的臂）----------
    cnt_files = {}
    for key in list(curves.keys()) + ["U0_s1"]:
        if key.startswith("E1_"):
            root = os.path.join(a.run_dir, "E1"); v = key[3:].split("_")[0]; s = 1
        elif key.startswith("U0_"):
            root = a.run_dir; v = "U0"; s = 1
        else:
            parts = key.split("_"); v = parts[0]; s = int(parts[1][1:])
            root = a.run_dir if s == 1 and v in ("T0", "T1", "U0") else a.run_dir
        p = find_counts(root, v, s, a.iters)
        if p:
            cnt_files[key] = p
    R["counts_available"] = cnt_files
    contrasts = [("T1_s1", "T0_s1", "采样效应：T1 对比 T0"),
                 ("U0_s1", "T1_s1", "P2 上下文网：U0 对比 T1")]
    for s in (2, 3):
        contrasts.append(("T1_s%d" % s, "T0_s%d" % s, "采样效应（seed %d）：T1 对比 T0" % s))
    if "E1_T0_s1" in cnt_files:
        contrasts.append(("E1_T0_s1", "T0_s1", "延长训练（T0 臂）：E1 对比 T0"))
    if "E1_T1_s1" in cnt_files:
        contrasts.append(("E1_T1_s1", "T1_s1", "延长训练（T1 臂）：E1 对比 T1"))
    boot = {}
    invalid = []
    for ka, kb, label in contrasts:
        if ka not in cnt_files or kb not in cnt_files:
            invalid.append({"contrast": "%s_vs_%s" % (ka, kb), "label": label,
                            "reason": "缺产物（%s 需要 it%d 的逐图计数）"
                                      % ("或".join(x for x in (ka, kb) if x not in cnt_files), a.iters),
                            "files": {ka: cnt_files.get(ka), kb: cnt_files.get(kb)}})
            continue
        da, ra = load_counts(cnt_files[ka])
        db, rb = load_counts(cnt_files[kb])
        # ---- e1_gate 式配对自检：不满足就不出数，标为 invalid ----
        why = []
        if len(ra) != len(rb):
            why.append("逐图行数不同 %d vs %d" % (len(ra), len(rb)))
        if da.get("seed") != db.get("seed"):
            why.append("seed 不同 %s vs %s" % (da.get("seed"), db.get("seed")))
        ia = iou(agg(ra, range(len(ra)), "base0")); ib = iou(agg(rb, range(len(rb)), "base0"))
        if abs(ia - ib) > 1e-12:
            why.append("两臂 base0 MA-IoU 不等 %.6f vs %.6f（val 集或 tiling 不一致）" % (ia, ib))
        if da.get("iter") != db.get("iter"):
            why.append("iter 不同 %s vs %s" % (da.get("iter"), db.get("iter")))
        if why:
            invalid.append({"contrast": "%s_vs_%s" % (ka, kb), "label": label,
                            "reason": "；".join(why), "files": {ka: cnt_files[ka], kb: cnt_files[kb]}})
            continue
        r = paired_bootstrap(ra, rb)
        # ⚠ 方向必须写死进标签：paired_bootstrap 的 Δ 恒为 IoU(b) − IoU(a)，
        # 而对比元组里 a=ka 在前 —— 初版标签文字写成"kb − ka"，与实际方向**相反**。
        r["label"] = "%s ｜ Δ = IoU(%s) − IoU(%s)" % (label, kb, ka)
        r["a_tag"], r["b_tag"] = ka, kb
        r["files"] = {ka: cnt_files[ka], kb: cnt_files[kb]}
        r["iter"] = da.get("iter")
        boot["%s_vs_%s" % (ka, kb)] = r
    R["bootstrap_invalid"] = invalid

    # ---------- 2b) 每个臂的自对照：head 相对 base0 ----------
    selfc = {}
    for key, p in sorted(cnt_files.items()):
        _, rows = load_counts(p)
        selfc[key] = within_file_bootstrap(rows)
        selfc[key]["file"] = p
    R["self_contrast_head_vs_base0"] = selfc
    R["bootstrap_contrasts"] = boot

    # ---------- 3) seed 离散度（V1 的意义所在）----------
    seed_spread = {}
    for v in ("T0", "T1"):
        vals = {}
        for s in (1, 2, 3):
            key = "%s_s%d" % (v, s)
            if key in cnt_files:
                _, rows = load_counts(cnt_files[key])
                vals[s] = iou(agg(rows, range(len(rows)), "head"))
        if len(vals) >= 2:
            seed_spread[v] = {"per_seed": vals,
                              "range": max(vals.values()) - min(vals.values()),
                              "mean": float(np.mean(list(vals.values())))}
    R["seed_spread"] = seed_spread

    # ---------- 4) J 线 ----------
    J = {}
    for arm in ("J0", "J1"):
        wd = os.path.join(a.j_workdir, arm)
        if os.path.isdir(wd):
            J[arm] = parse_mmseg_logjson(wd)
    R["J_line"] = J
    R["J_note"] = "J 线是 mmseg EvalHook 的聚合 mIoU，没有逐图计数 → 不能做图像级配对 bootstrap"

    # ---------- 5) E1 判定门 ----------
    gp = os.path.join(a.run_dir, "experiments", "E1_gate.json")
    R["E1_gate"] = json.load(open(gp)) if os.path.exists(gp) else None

    # ---------- 6) 报告 ----------
    L = ["# 审查轮汇总（自动生成，C1 阶段）", ""]
    L.append("## 1. val 曲线（MA-IoU）")
    L.append("")
    L.append("| 臂 | base0 | 终点 head | Δ vs base0 | 各 val 点 head |")
    L.append("|---|---|---|---|---|")
    for k, v in sorted(R["val_curves"].items()):
        pts = v["points"]
        if not pts:
            continue
        b0 = pts[0][1]
        end = pts[-1][2]
        seq = " ".join("it%d:%.4f" % (p[0], p[2]) for p in pts)
        L.append("| %s | %.6f | %.6f | %+.6f | %s |" % (k, b0, end, end - b0, seq))
    L.append("")
    L.append("## 2. 配对 bootstrap 对比（图像级，2000 次，seed %d）" % SEED)
    L.append("")
    L.append("> **Δ 的方向写在每行标签里**（Δ = IoU(后者) − IoU(前者)），不要凭『T1 对比 T0』猜正负。")
    L.append("> 跨零 = 是 ⇒ 这个差异与 0 无法区分，**不能说谁更好**。")
    L.append("")
    L.append("| 对比 | IoU_a | IoU_b | Δ | 95%CI | 跨零 | 单侧 5% 分位 |")
    L.append("|---|---|---|---|---|---|---|")
    for k, r in R["bootstrap_contrasts"].items():
        L.append("| %s | %.6f | %.6f | %+.6f | [%+.6f, %+.6f] | %s | %+.6f |"
                 % (r["label"], r["iou_a"], r["iou_b"], r["delta"], r["ci95"][0], r["ci95"][1],
                    "是" if r["crosses_zero"] else "否", r["q05_one_sided"]))
    for bad in R.get("bootstrap_invalid", []):
        L.append("| %s | — | — | — | — | **无效** | — |" % bad["label"])
        L.append("| ↳ 原因 | %s | | | | | |" % bad["reason"])
    L.append("")
    L.append("> 空行/无效行含义：对比所依赖的产物**不存在或未通过配对自检**（列数与行数不同、seed 不同、")
    L.append("> 两臂 base0 MA-IoU 不等、iter 不同）。**不会拿不同训练步数或不同 val 集硬比**。")
    if R.get("counts_available"):
        L.append("")
        L.append("实际用到的逐图计数文件：")
        for k, p in sorted(R["counts_available"].items()):
            L.append("- %s → %s" % (k, p))
    L.append("")
    L.append("## 2b. 每个臂的自对照：head 相对 base0（同一文件内配对，回答\"这条改造有没有把 base0 抬起来\"）")
    L.append("")
    L.append("| 臂 | base0 MA-IoU | head MA-IoU | Δ | 95%CI | 跨零 | 单侧 5% 分位 |")
    L.append("|---|---|---|---|---|---|---|")
    for k, r in sorted(R.get("self_contrast_head_vs_base0", {}).items()):
        L.append("| %s | %.6f | %.6f | %+.6f | [%+.6f, %+.6f] | %s | %+.6f |"
                 % (k, r["iou_base0"], r["iou_head"], r["delta"], r["ci95"][0], r["ci95"][1],
                    "是" if r["crosses_zero"] else "否", r["q05_one_sided"]))
    L.append("")
    L.append("## 3. seed 离散度（V1 的用途）")
    L.append("")
    for v, d in R["seed_spread"].items():
        L.append("- **%s**：各 seed head MA-IoU = %s，极差 = %.6f，均值 = %.6f"
                 % (v, {k: round(x, 6) for k, x in d["per_seed"].items()}, d["range"], d["mean"]))
    L.append("")
    L.append("## 4. J 线（mmseg 聚合 mIoU，无逐图计数）")
    L.append("")
    for arm, rows in J.items():
        L.append("- %s：%s" % (arm, [{"iter": r["iter"], "mIoU": r["mIoU"]} for r in rows] or "（暂无）"))
    if not J:
        L.append("- （未找到 J 线产物）")
    L.append("")
    L.append("## 5. E1 判定门")
    L.append("")
    if R["E1_gate"]:
        g = R["E1_gate"]
        L.append("- 结论：**%s**；Δ=%+.6f，CI95=[%+.6f, %+.6f]，单侧 5%% 分位=%+.6f"
                 % (g.get("verdict"), g.get("delta_point", float("nan")),
                    *g.get("delta_ci95_two_sided", [float("nan")] * 2),
                    g.get("delta_q05_one_sided", float("nan"))))
        L.append("- 局限：%s" % g.get("caveat", ""))
    else:
        L.append("- （未找到 E1_gate.json）")
    L.append("")
    L.append("## 6. 缺失产物清单（诚实披露：缺失 ≠ 该臂不存在，也可能是本 run-dir 不可访问）")
    L.append("")
    missing = []
    for key in sorted(set(["%s_s%d" % (v, s) for v, s in arms] + ["U0_s1", "E1_T0_s1", "E1_T1_s1"])):
        if key not in R["val_curves"]:
            missing.append("val 曲线缺失: %s" % key)
        if key not in cnt_files:
            missing.append("逐图计数缺失: %s（需要 it%d）" % (key, a.iters))
    if not os.path.isdir(a.run_dir):
        missing.insert(0, "!! run-dir 不可访问: %s（此时所有 T 线产物都读不到，不代表臂未完成）" % a.run_dir)
    L.append("")
    if missing:
        for m in missing:
            L.append("- %s" % m)
    else:
        L.append("- 无（全部产物齐备）")
    L.append("")
    L.append("## 7. 必须披露的统计局限")
    L.append("")
    L.append("- 第 2 节的 95%CI 只刻画**图像级**不确定性（给定这两个模型）；**不含 seed 不确定性** ——")
    L.append("  seed 不确定性只能由第 3 节的 3-seed 离散度描述。两者不可互相替代。")
    L.append("- J 线没有逐图计数，因此无法进入第 2 节的配对检验；若需要，应补一个逐图计数导出。")

    os.makedirs(a.out_dir, exist_ok=True)
    jp = os.path.join(a.out_dir, "round_summary.json")
    mp = os.path.join(a.out_dir, "ROUND_SUMMARY.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\n写出: %s\n      %s" % (mp, jp))


if __name__ == "__main__":
    main()
