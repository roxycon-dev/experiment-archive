#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase A4：数据边界检查（prompt §3.4）。

约束（严格遵守）：
  - 只做完整性审计：数量、不相交、重复 ID、图像-标签匹配、文件存在性。
  - **不利用 test 标签统计、病例图像或分层误差来指导任何设计。**
    因此 test 的逐类像素聚合被有意跳过，只记录图像数与"已跳过"标记。
  - 不重划划分，不合并 train+val。

用法：
  python phaseA4_data_boundary.py --split-assignment <json> --data-manifest <json> \
      --out-dir <audit_runs/<run_id>>
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))
CLASSES = ["EX", "HE", "SE", "MA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-assignment", required=True)
    ap.add_argument("--data-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rep = {"generated_at": datetime.now(CST).isoformat(),
           "phase": "A4_data_boundary",
           "constraint_note": ("test 标签统计被有意跳过，不用于指导设计（prompt §3.4）；"
                               "本文件仅作完整性审计记录。")}

    with open(args.split_assignment, encoding="utf-8") as f:
        sa = json.load(f)
    with open(args.data_manifest, encoding="utf-8") as f:
        dm = json.load(f)

    print("== split_assignment ==")
    print("  n_entries =", len(sa))
    rec_sa = {"n_entries": len(sa), "sha256": __import__("hashlib").sha256(
        open(args.split_assignment, "rb").read()).hexdigest()}

    # 结构
    fields = Counter()
    for v in sa.values():
        if isinstance(v, dict):
            fields.update(v.keys())
    rec_sa["field_presence"] = dict(fields)
    print("  字段出现次数 =", dict(fields))

    # split 计数
    split_counts = Counter(str(v.get("split")) for v in sa.values() if isinstance(v, dict))
    orig_counts = Counter(str(v.get("original_subset")) for v in sa.values() if isinstance(v, dict))
    tier_counts = Counter(str(v.get("tier")) for v in sa.values() if isinstance(v, dict))
    rec_sa["split_counts"] = dict(split_counts)
    rec_sa["original_subset_counts"] = dict(orig_counts)
    rec_sa["tier_counts"] = dict(tier_counts)
    print("  split 分布 =", dict(split_counts))
    print("  original_subset 分布 =", dict(orig_counts))
    print("  tier 分布 =", dict(tier_counts))

    # §3.4 关键：454/76/227
    expect = {"train": 454, "val": 76, "test": 227}
    check_counts = {k: {"expected": v, "actual": split_counts.get(k, 0),
                        "pass": split_counts.get(k, 0) == v}
                    for k, v in expect.items()}
    rec_sa["count_check_454_76_227"] = check_counts
    for k, v in check_counts.items():
        print("  %-6s 期望 %3d 实际 %3d  %s"
              % (k, v["expected"], v["actual"], "PASS" if v["pass"] else "FAIL"))
    total = sum(split_counts.values())
    rec_sa["total_entries"] = total
    print("  合计 = %d  (454+76+227 = %d)" % (total, 454 + 76 + 227))

    # ID 唯一性与不相交（键本身是唯一键，交叠只可能来自同一 ID 出现在多个 split）
    ids_by_split = defaultdict(set)
    for k, v in sa.items():
        if isinstance(v, dict):
            ids_by_split[str(v.get("split"))].add(k)
    pairs = {}
    names = sorted(ids_by_split)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = ids_by_split[names[i]] & ids_by_split[names[j]]
            pairs["%s∩%s" % (names[i], names[j])] = {
                "n_overlap": len(inter), "pass": len(inter) == 0,
                "sample": sorted(inter)[:5]}
    rec_sa["disjointness"] = pairs
    print("  split 两两不相交:")
    for k, v in pairs.items():
        print("    %-14s overlap=%d  %s" % (k, v["n_overlap"], "PASS" if v["pass"] else "FAIL"))

    # 重复 ID（同一 basename 出现多次，忽略扩展名）
    base = Counter(os.path.splitext(k)[0] for k in sa)
    dups = {k: c for k, c in base.items() if c > 1}
    rec_sa["duplicate_basenames"] = {"n": len(dups), "sample": dict(list(dups.items())[:10])}
    print("  重复 basename: %d 个" % len(dups))

    # 患者 ID 推断 —— 先验证规则是否可用，不可用就明确记为 INCONCLUSIVE
    def patient_of(fn):
        m = re.match(r"^(\d{3})-\d+-\d+\.", fn)
        if m:
            return "P" + m.group(1)
        return "TIMESTAMP"

    fn_pat = Counter()
    for k in sa:
        if re.match(r"^\d{3}-\d+-\d+\.", k):
            fn_pat["NNN-xxxx-NNN 形式"] += 1
        elif re.match(r"^\d{8,}\.", k):
            fn_pat["纯数字时间戳"] += 1
        else:
            fn_pat["其他"] += 1
    prefixes = sorted(set(k[:3] for k in sa if re.match(r"^\d{3}-", k)))

    pat = defaultdict(set)
    for k, v in sa.items():
        if isinstance(v, dict):
            pat[str(v.get("split"))].add(patient_of(k))
    overlap = {p: sorted(pat["train"] & pat[p]) for p in ("val", "test") if p in pat}

    # 规则可用性判定：只有当 NNN 前缀足够多样、且时间戳无法归组时才算可信
    rule_usable = (len(prefixes) > 3) and (fn_pat.get("纯数字时间戳", 0) == 0)
    rec_sa["patient_id_inference"] = {
        "rule": "文件名形如 NNN-xxxx-NNN.jpg 时取前 3 位为患者号；否则记为 TIMESTAMP",
        "filename_pattern_counts": dict(fn_pat),
        "distinct_nnn_prefixes": prefixes,
        "rule_usable": rule_usable,
        "verdict": ("BLOCKED: 患者 ID 无法从文件名可靠推断。"
                    "证据：%d 个 NNN 形式文件的前缀全部是 %s（批次编号而非患者号），"
                    "%d 个文件是纯数字时间戳无法归组。因此任何基于文件名的"
                    "'患者重叠'结论都是规则假象，不得作为患者级泄漏证据。"
                    "需要数据集自带的患者映射表才能做患者级审计。"
                    % (fn_pat.get("NNN-xxxx-NNN 形式", 0), prefixes,
                       fn_pat.get("纯数字时间戳", 0))),
        "raw_overlap_warning": ("以下数字为失效规则的产物，仅作反例留存，"
                                "不得引用为患者级重叠证据"),
        "raw_overlap_from_bad_rule": {k: v for k, v in overlap.items()},
        "n_patients_per_split_from_bad_rule": {k: len(v) for k, v in pat.items()},
    }
    print("  文件名模式:", dict(fn_pat), " NNN 前缀:", prefixes)
    print("  患者级审计: BLOCKED（规则不可用，见 JSON 的 verdict）")
    if not rule_usable:
        print("    ⚠ 撤回任何基于文件名的患者重叠结论")

    # class_pixels：只对 train / val 聚合（test 有意跳过）
    cls_agg = {}
    for sp in ("train", "val"):
        tot = {c: 0 for c in CLASSES}
        n = 0
        for k, v in sa.items():
            if isinstance(v, dict) and str(v.get("split")) == sp:
                cp = v.get("class_pixels") or {}
                for c in CLASSES:
                    tot[c] += int(cp.get(c, 0) or 0)
                n += 1
        cls_agg[sp] = {"n_images": n, "class_pixels": tot}
    rec_sa["class_pixels_train_val_only"] = cls_agg
    rec_sa["class_pixels_test"] = "SKIPPED_BY_DESIGN(prompt §3.4)"
    print("  逐类像素（仅 train/val，test 按 §3.4 有意跳过）:")
    for sp, v in cls_agg.items():
        print("    %-6s n=%d  %s" % (sp, v["n_images"], v["class_pixels"]))

    # imgsize/尺寸是否存在
    ks = set()
    for v in sa.values():
        if isinstance(v, dict):
            ks.update(v.keys())
    rec_sa["all_fields_seen"] = sorted(ks)

    # ---- data_manifest ----
    print("\n== data_manifest ==")
    rec_dm = {"n_entries": len(dm) if isinstance(dm, list) else None}
    if isinstance(dm, list):
        kinds = Counter()
        by_split = Counter()
        for e in dm:
            p = e.get("path", "")
            parts = p.split("/")
            if len(parts) >= 2:
                kinds[parts[-2]] += 1        # images / labels
            for sp in ("train", "val", "test"):
                if "/%s/" % sp in p:
                    by_split[(parts[-2], sp)] += 1
        rec_dm["kind_counts"] = dict(kinds)
        rec_dm["kind_x_split"] = {"%s/%s" % k: v for k, v in sorted(by_split.items())}
        rec_dm["has_sha256"] = all("sha256" in e for e in dm[:50])
        rec_dm["sample"] = dm[0]
        print("  目录类型计数 =", dict(kinds))
        for k, v in sorted(by_split.items()):
            print("    %-8s %-6s %4d" % (k[0], k[1], v))
        print("  has_sha256 =", rec_dm["has_sha256"])
        # 图像-标签配对
        img_base = set()
        lab_base = set()
        for e in dm:
            p = e.get("path", "")
            b = os.path.splitext(os.path.basename(p))[0]
            if "/images/" in p:
                img_base.add(b)
            elif "/labels/" in p:
                lab_base.add(b)
        rec_dm["image_label_pairing"] = {
            "n_images": len(img_base), "n_labels": len(lab_base),
            "images_without_label": sorted(img_base - lab_base)[:10],
            "labels_without_image": sorted(lab_base - img_base)[:10],
            "pass": img_base == lab_base and len(img_base) > 0,
        }
        pr = rec_dm["image_label_pairing"]
        print("  图像 %d / 标签 %d  配对 %s"
              % (pr["n_images"], pr["n_labels"], "PASS" if pr["pass"] else "FAIL"))
        if not pr["pass"]:
            print("    无标签的图像:", pr["images_without_label"])
            print("    无图像的标签:", pr["labels_without_image"])
        # 跨 split 重复哈希（完整性审计）
        h = defaultdict(list)
        for e in dm:
            if "/images/" in e.get("path", ""):
                h[e.get("sha256")].append(e.get("path"))
        dup = {k: v for k, v in h.items() if len(v) > 1}
        rec_dm["cross_split_duplicate_image_hashes"] = {
            "n": len(dup), "sample": dict(list(dup.items())[:5])}
        print("  跨文件重复图像哈希 = %d" % len(dup))
        for k, v in list(dup.items())[:3]:
            print("    ", k[:12], v)

    rep["split_assignment"] = rec_sa
    rep["data_manifest"] = rec_dm

    # 总判定
    hard = []
    if not all(v["pass"] for v in check_counts.values()):
        hard.append("454/76/227 计数不匹配")
    if any(not v["pass"] for v in pairs.values()):
        hard.append("split 存在交叠")
    if isinstance(dm, list) and not rec_dm["image_label_pairing"]["pass"]:
        hard.append("图像-标签不配对")
    rep["hard_failures"] = hard
    rep["status"] = "PASS" if not hard else "FAIL"
    print("\n=== A4 状态: %s ===" % rep["status"])
    if hard:
        for h in hard:
            print("  FAIL:", h)

    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, "phaseA4_data_boundary.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("写出:", p)
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
