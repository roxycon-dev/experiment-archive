# -*- coding: utf-8 -*-
"""inspect_ddr_contract.py — 只读解析固定 DDR split（DDR_split_613）与标签契约。

输出内容：
1. 类别映射：bg=0/EX=1/HE=2/SE=3/MA=4；评估通道序 EX/HE/SE/MA（通道 i ↔ class i+1）；
2. 数量：各 split 图像/标签数量、文件名对齐、split 互斥性；
3. 标签值域：train/val 融合 PNG 的取值域（应 ⊆ {0,1,2,3,4}）与各类阳性像素计数，
   并与 split_stats.csv 的原始结果交叉核对；
4. test 标签只做文件级检查（数量/文件名/头部尺寸），**不读取任何像素值**，
   绝不用于阈值、模型或训练配置决策（只读工具，也不修改任何数据）。

结构异常（数量不齐、文件名不对齐、split 泄漏、值域越界、与 split_stats 不一致）
→ 退出码非零（fail closed）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_common import (  # noqa: E402
    CHANNEL_INDEX_OF_LESION,
    DDR_CLASS_INDEX,
    DDR_CLASS_ID_OF_CHANNEL,
    LESION_CHANNEL_ORDER,
    ContractError,
    write_text_atomic,
)

try:
    import numpy as np
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


IMG_EXT = ".jpg"
LBL_EXT = ".png"


def stems_of(d: str, ext: str):
    if not os.path.isdir(d):
        return None
    out = set()
    for f in os.listdir(d):
        if f.lower().endswith(ext):
            out.add(os.path.splitext(f)[0])
    return out


def read_label_values(path: str):
    """读融合 PNG 的值域与逐类计数（仅 train/val；test 绝不调用本函数）。"""
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    vals = set(int(v) for v in np.unique(arr))
    per_class = {}
    for cls, cid in DDR_CLASS_INDEX.items():
        if cid == 0:
            continue
        per_class[cls] = int((arr == cid).sum())
    return vals, per_class


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="只读解析固定 DDR split 与标签契约")
    ap.add_argument("--ddr-root", default=r"D:\HRDecoder\data\DDR_split_613",
                    help="DDR_split_613 目录（只读）")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--max-label-pixels", type=int, default=0,
                    help=">0 时限制读取 train/val 标签张数（抽样模式），0=全量")
    args = ap.parse_args(argv)

    errors, warnings, checks = [], [], {}
    root = args.ddr_root
    if not os.path.isdir(root):
        print(f"[inspect_ddr_contract] FAIL: DDR root not found: {root}", file=sys.stderr)
        return 2

    # ---- 1. 类别映射（代码常量权威输出） ----
    checks["class_mapping"] = {
        "DDR_CLASS_INDEX": DDR_CLASS_INDEX,
        "channel_order": list(LESION_CHANNEL_ORDER),
        "channel_index": CHANNEL_INDEX_OF_LESION,
        "channel_to_ddr_class": {str(k): v for k, v in DDR_CLASS_ID_OF_CHANNEL.items()},
        "sigmoid_eval_rule": "channel i (0..3) <-> DDR class id i+1 (lesion_metric.py:71-77)",
    }

    # ---- 2. 数量与对齐 ----
    split_info = {}
    all_stems = {}
    for split in ("train", "val", "test"):
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        s_img = stems_of(img_dir, IMG_EXT)
        s_lbl = stems_of(lbl_dir, LBL_EXT)
        if s_img is None or s_lbl is None:
            errors.append(f"{split}: images/ 或 labels/ 目录缺失（{img_dir} / {lbl_dir}）")
            continue
        entry = {
            "images": len(s_img),
            "labels": len(s_lbl),
            "stems_aligned": s_img == s_lbl,
            "img_only": sorted(s_img - s_lbl)[:10],
            "lbl_only": sorted(s_lbl - s_img)[:10],
        }
        if s_img != s_lbl:
            errors.append(f"{split}: image/label stem 不对齐")
        all_stems[split] = s_img
        split_info[split] = entry
    # 互斥性
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and a in all_stems and b in all_stems:
                ov = all_stems[a] & all_stems[b]
                checks[f"disjoint_{a}_{b}"] = len(ov)
                if ov:
                    errors.append(f"split 泄漏：{a} ∩ {b} = {len(ov)} 张，例如 {sorted(ov)[:5]}")

    # ---- 3. split_assignment.json / split_report.md ----
    assign_path = os.path.join(root, "split_assignment.json")
    if os.path.isfile(assign_path):
        with open(assign_path, "r", encoding="utf-8") as f:
            assign = json.load(f)
        checks["split_assignment"] = {
            "seed": assign.get("seed"),
            "anomalies_count": len(assign.get("anomalies", [])),
            "anomalies_head": assign.get("anomalies", [])[:3],
        }
    else:
        errors.append(f"split_assignment.json 缺失: {assign_path}")
    checks["split_report_md_exists"] = os.path.isfile(os.path.join(root, "split_report.md"))
    checks["split_stats_csv_exists"] = os.path.isfile(os.path.join(root, "split_stats.csv"))

    # ---- 4. 每类 TIF 存档目录（train/val） ----
    perclass_tif = {}
    for split in ("train", "val"):
        for cls in LESION_CHANNEL_ORDER:
            d = os.path.join(root, "labels", split, cls)
            n = len([f for f in os.listdir(d) if f.lower().endswith(".tif")]) if os.path.isdir(d) else None
            perclass_tif[f"{split}/{cls}"] = n
            if n is None:
                warnings.append(f"labels/{split}/{cls}/ 目录缺失（仅存档用途，不影响训练接口）")
            elif split in split_info and n != split_info[split]["images"]:
                errors.append(
                    f"labels/{split}/{cls}/ TIF 数 {n} != 图像数 {split_info[split]['images']}"
                )
    checks["per_class_tif_counts"] = perclass_tif

    # ---- 5. train/val 标签值域与像素计数（不触碰 test 像素） ----
    stats_csv = {}
    stats_path = os.path.join(root, "split_stats.csv")
    if os.path.isfile(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stats_csv[(row["split"], row["class"])] = int(row["pos_pixels"])

    label_domain = {}
    for split in ("train", "val"):
        if split not in all_stems:
            continue
        lbl_dir = os.path.join(root, "labels", split)
        files = sorted(f for f in os.listdir(lbl_dir) if f.lower().endswith(LBL_EXT))
        if args.max_label_pixels > 0:
            files = files[: args.max_label_pixels]
            label_domain[f"{split}_sampled"] = len(files)
        union_vals = set()
        pos = {c: 0 for c in LESION_CHANNEL_ORDER}
        n_bad = 0
        for fn in files:
            try:
                vals, per_class = read_label_values(os.path.join(lbl_dir, fn))
            except Exception as e:
                errors.append(f"{split}: 标签不可读 {fn}: {e}")
                n_bad += 1
                continue
            union_vals |= vals
            for c in pos:
                pos[c] += per_class[c]
        bad_vals = sorted(v for v in union_vals if v not in set(DDR_CLASS_INDEX.values()))
        if bad_vals:
            errors.append(f"{split}: 标签值域越界 {bad_vals}（契约只允许 0..4）")
        label_domain[split] = {
            "unique_values": sorted(union_vals),
            "positive_pixels": pos,
        }
        for c in LESION_CHANNEL_ORDER:
            key = (split, c)
            if key in stats_csv and args.max_label_pixels == 0 and n_bad == 0:
                if stats_csv[key] != pos[c]:
                    errors.append(
                        f"{split}/{c}: 阳性像素 {pos[c]} != split_stats.csv 记录 {stats_csv[key]}（数据漂移？）"
                    )
    checks["label_domain"] = label_domain
    checks["test_policy"] = "test 标签仅做文件级检查（数量/文件名/头部尺寸），未读取任何像素值"

    # ---- 6. test 文件级尺寸抽查（只读头部，不读像素） ----
    if _HAS_PIL and "test" in all_stems:
        img_dir = os.path.join(root, "images", "test")
        lbl_dir = os.path.join(root, "labels", "test")
        mism = 0
        checked = 0
        for fn in sorted(os.listdir(img_dir))[:20] if args.max_label_pixels == 0 else []:
            stem = os.path.splitext(fn)[0]
            lp = os.path.join(lbl_dir, stem + LBL_EXT)
            if not os.path.isfile(lp):
                mism += 1
                continue
            with Image.open(os.path.join(img_dir, fn)) as im:
                iw, ih = im.size
            with Image.open(lp) as im:  # 仅头部尺寸
                lw, lh = im.size
            checked += 1
            if (iw, ih) != (lw, lh):
                mism += 1
        checks["test_header_size_check"] = {"checked": checked, "mismatch": mism}
        if mism:
            errors.append(f"test: {mism}/{checked} 图像与标签头部尺寸不一致")

    ok = not errors
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ddr_root": os.path.abspath(root),
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
    out_json = args.output_json or os.path.abspath(root) + ".ddr_contract_inspect.json"
    # 默认写进工作区 docs/，避免污染只读的 DDR 目录
    if os.path.isfile(out_json) is False and not os.access(os.path.dirname(out_json), os.W_OK):
        out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "ddr_contract_inspect.json")
    out_md = args.output_md or out_json.replace(".json", ".md")

    md = []
    md.append("# DDR 固定 split 契约检查（只读）")
    md.append("")
    md.append(f"- 生成时间：{report['generated_at']}")
    md.append(f"- DDR 根目录：`{os.path.abspath(root)}`")
    md.append(f"- **结论：{'PASS' if ok else 'FAIL'}**（errors={len(errors)}, warnings={len(warnings)}）")
    md.append("")
    md.append("## 类别映射（代码常量）")
    md.append(f"- `DDR_CLASS_INDEX = {json.dumps(DDR_CLASS_INDEX)}`")
    md.append(f"- 评估通道序 `{list(LESION_CHANNEL_ORDER)}`，通道 i ↔ DDR class i+1")
    if "split_assignment" in checks:
        sa = checks["split_assignment"]
        md.append("")
        md.append("## split 元数据")
        md.append(f"- seed：{sa['seed']}；anomalies 记录：{sa['anomalies_count']} 条")
    for split in ("train", "val", "test"):
        if split in split_info:
            e = split_info[split]
            md.append(f"- **{split}**：images={e['images']}, labels={e['labels']}, "
                      f"stems_aligned={e['stems_aligned']}")
    if "label_domain" in checks:
        for split in ("train", "val"):
            if split in checks["label_domain"]:
                ld = checks["label_domain"][split]
                md.append(f"- {split} 标签值域：{ld['unique_values']}；阳性像素：{ld['positive_pixels']}")
    if errors:
        md.append("\n## Errors")
        for e in errors:
            md.append(f"- {e}")
    if warnings:
        md.append("\n## Warnings")
        for w in warnings:
            md.append(f"- {w}")
    md.append("")

    write_text_atomic(out_json, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(out_md, "\n".join(md))
    print(f"[inspect_ddr_contract] {'PASS' if ok else 'FAIL'}: errors={len(errors)} warnings={len(warnings)}")
    print(f"[inspect_ddr_contract] report -> {out_json}")
    print(f"[inspect_ddr_contract] report -> {out_md}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
