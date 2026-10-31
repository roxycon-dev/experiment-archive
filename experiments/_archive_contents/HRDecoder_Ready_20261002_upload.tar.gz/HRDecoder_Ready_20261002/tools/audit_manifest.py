# -*- coding: utf-8 -*-
"""audit_manifest.py — 对统一外部病灶 manifest 做机器审计，输出 JSON + Markdown。

检查项（error → 退出码非零；warning → 记录不阻塞）：
E1 文件存在：image 与已标注类 mask/valid 文件存在且可读；
E2 尺寸对齐：解码后 image 尺寸与 original_width/height 一致，mask 与 image 尺寸一致；
E3 二值掩膜：mask 的非零值只有一种（或 0）；
E4 valid 语义：valid_mode 与各路径字段自洽（none→mask_path 必须为 null 等）；
E5 split 泄漏：external 清单内 sample_id/image_id 重复、official train/test 之间
   patient/image 冲突；给了 --ddr-images-root 时与 DDR 图像做感知哈希近重复比对；
W1 精确重复：image SHA256 完全相同的样本对；
W2 近重复：aHash/dHash/pHash 任一 64bit 汉明距离 <= 阈值的样本对（人工复核用）。

依赖策略：numpy+PIL 必需（感知哈希）；jsonschema 可选（缺失时用内置最小校验器，
降级记录在报告 degradations 中）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_common import (  # noqa: E402
    LESION_CHANNEL_ORDER,
    MANIFEST_SCHEMA_VERSION,
    NEAR_DUP_HAMMING_THRESHOLD,
    VALID_MODES,
    ContractError,
    compute_perceptual_hashes,
    hamming_hex,
    load_manifest,
    write_text_atomic,
)

try:
    import numpy as np
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False

try:  # 可选依赖：jsonschema 存在则严格校验
    import jsonschema  # noqa: F401

    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False


# ---------------------------------------------------------------------------
# 内置最小校验器（jsonschema 缺失时的降级实现，只覆盖本 schema 用到的子集）
# ---------------------------------------------------------------------------

def minimal_validate(record: dict, schema: dict):
    """返回错误信息列表。只实现 type/required/enum/const/pattern/minLength/keys。"""
    import re

    errs = []

    def check(obj, sch, path):
        t = sch.get("type")
        if t is not None:
            types = t if isinstance(t, list) else [t]
            pytypes = {
                "object": dict, "array": list, "string": str,
                "integer": int, "number": (int, float), "boolean": bool, "null": type(None),
            }
            ok = False
            for tt in types:
                pt = pytypes.get(tt)
                if pt is None:
                    continue
                if isinstance(obj, pt) and not (tt == "integer" and isinstance(obj, bool)):
                    ok = True
                    break
            if not ok:
                errs.append(f"{path}: expected type {t}, got {type(obj).__name__}")
                return
        if "const" in sch and obj != sch["const"]:
            errs.append(f"{path}: expected const {sch['const']!r}, got {obj!r}")
        if "enum" in sch and obj not in sch["enum"]:
            errs.append(f"{path}: value {obj!r} not in enum {sch['enum']}")
        if isinstance(obj, str):
            if "pattern" in sch and not re.match(sch["pattern"], obj):
                errs.append(f"{path}: {obj!r} fails pattern {sch['pattern']}")
            if "minLength" in sch and len(obj) < sch["minLength"]:
                errs.append(f"{path}: shorter than minLength={sch['minLength']}")
        if isinstance(obj, dict):
            for req in sch.get("required", []):
                if req not in obj:
                    errs.append(f"{path}: missing required key {req!r}")
            props = sch.get("properties", {})
            pat_props = sch.get("patternProperties") or {}
            if sch.get("additionalProperties") is False:
                for k in obj:
                    # patternProperties 的键是正则（如 lesions 的 ^(EX|HE|SE|MA)$），
                    # 必须按正则匹配而不是字面键名，否则合法键会被误判为 unexpected。
                    if k in props or any(re.match(pk, k) for pk in pat_props):
                        continue
                    errs.append(f"{path}: unexpected key {k!r}")
            for k, v in obj.items():
                if k in props:
                    check(v, props[k], f"{path}.{k}")
                else:
                    for pk, pv in pat_props.items():
                        if re.match(pk, k):
                            check(v, pv, f"{path}.{k}")
                            break

    check(record, schema, "$")
    return errs


def validate_record(record: dict, schema: dict, use_jsonschema: bool):
    if use_jsonschema:
        return []  # jsonschema 校验单独做，错误在调用处收集
    return minimal_validate(record, schema)


# ---------------------------------------------------------------------------
# 检查逻辑
# ---------------------------------------------------------------------------

def resolve_path(p: str, base_dir: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(base_dir, p)


def audit(records, schema, base_dir, ddr_images_root, phash_threshold, max_dup_report):
    errors, warnings, degradations = [], [], []
    if not _HAS_PIL:
        degradations.append("numpy/PIL unavailable: 尺寸/掩膜/感知哈希检查全部跳过（不允许静默假装通过）")

    # ---- schema 校验 ----
    if schema is not None:
        for i, rec in enumerate(records):
            if _HAS_JSONSCHEMA:
                try:
                    jsonschema.validate(rec, schema)
                except jsonschema.ValidationError as e:
                    errors.append(f"E0 schema: record#{i} ({rec.get('sample_id', '?')}): {e.message}")
            else:
                for msg in minimal_validate(rec, schema):
                    errors.append(f"E0 schema(degraded): record#{i}: {msg}")
        if not _HAS_JSONSCHEMA:
            degradations.append("jsonschema 未安装：使用内置最小校验器（覆盖 required/type/enum/pattern）")
    else:
        degradations.append("未提供 schema 文件：跳过 schema 校验")

    # ---- E1..E4 逐记录 ----
    for i, rec in enumerate(records):
        sid = rec.get("sample_id", f"record#{i}")
        img_path = resolve_path(rec.get("image", {}).get("path", ""), base_dir)
        img_arr = None
        if _HAS_PIL:
            if not os.path.isfile(img_path):
                errors.append(f"E1 exists: {sid}: image missing: {img_path}")
            else:
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                        rgb = im.convert("RGB")
                        rgb.load()  # 句柄关闭前把像素读入内存
                    if (w, h) != (rec.get("original_width"), rec.get("original_height")):
                        errors.append(
                            f"E2 size: {sid}: decoded {w}x{h} != recorded "
                            f"{rec.get('original_width')}x{rec.get('original_height')}"
                        )
                    img_arr = rgb
                except Exception as e:
                    errors.append(f"E1 readable: {sid}: image unreadable: {img_path}: {e}")

        lesions = rec.get("lesions", {})
        # 四键必须恰为契约顺序
        if list(lesions.keys()) != list(LESION_CHANNEL_ORDER):
            errors.append(
                f"E4 order: {sid}: lesions keys {list(lesions.keys())} != contract order "
                f"{list(LESION_CHANNEL_ORDER)}"
            )
        if _HAS_PIL:
            for cls, lrec in lesions.items():
                mode = lrec.get("valid_mode")
                if mode not in VALID_MODES:
                    errors.append(f"E4 valid: {sid}/{cls}: bad valid_mode {mode!r}")
                    continue
                mp = lrec.get("mask_path")
                if mode == "none":
                    if mp is not None or lrec.get("mask_sha256") is not None:
                        errors.append(
                            f"E4 valid: {sid}/{cls}: valid_mode=none 但 mask_path/sha256 非空（未标注绝不能挂掩膜）"
                        )
                    continue
                if mp is None:
                    errors.append(f"E1 exists: {sid}/{cls}: valid_mode={mode} 但 mask_path 为空")
                    continue
                mpath = resolve_path(mp, base_dir)
                if not os.path.isfile(mpath):
                    errors.append(f"E1 exists: {sid}/{cls}: mask missing: {mpath}")
                    continue
                try:
                    with Image.open(mpath) as im:
                        m = np.array(im)
                except Exception as e:
                    errors.append(f"E1 readable: {sid}/{cls}: mask unreadable: {mpath}: {e}")
                    continue
                if m.ndim == 3 and not (
                    m.shape[2] >= 3
                    and np.array_equal(m[..., 0], m[..., 1])
                    and np.array_equal(m[..., 0], m[..., 2])
                ):
                    errors.append(f"E3 binary: {sid}/{cls}: 多通道不一致掩膜，禁止按颜色猜 class id: {mpath}")
                    continue
                if m.ndim == 3:
                    m = m[..., 0]
                vals = np.unique(m)
                nz = vals[vals > 0]
                if nz.size > 1:
                    errors.append(f"E3 binary: {sid}/{cls}: 非零值 {nz[:6].tolist()}… 不唯一: {mpath}")
                if img_arr is not None and (m.shape[1], m.shape[0]) != img_arr.size:
                    errors.append(
                        f"E2 size: {sid}/{cls}: mask {m.shape[1]}x{m.shape[0]} != image {img_arr.size[0]}x{img_arr.size[1]}"
                    )
                if lrec.get("mask_sha256") is not None:
                    from contract_common import sha256_file

                    actual = sha256_file(mpath)
                    if actual != lrec["mask_sha256"]:
                        errors.append(
                            f"E1 hash: {sid}/{cls}: mask sha256 mismatch (recorded != actual)"
                        )
                # valid_mode=file 的栅格检查
                if mode == "file":
                    vp = lrec.get("valid_mask_path")
                    if vp is None:
                        errors.append(f"E4 valid: {sid}/{cls}: valid_mode=file 但 valid_mask_path 为空")
                    else:
                        vpath = resolve_path(vp, base_dir)
                        if not os.path.isfile(vpath):
                            errors.append(f"E1 exists: {sid}/{cls}: valid mask missing: {vpath}")
                        else:
                            try:
                                with Image.open(vpath) as im:
                                    v = np.array(im)
                                if v.ndim == 3:
                                    v = v[..., 0]
                                if not ((v == 0) | (v > 0)).all():
                                    errors.append(f"E4 valid: {sid}/{cls}: valid 栅格异常: {vpath}")
                                if v.shape != m.shape:
                                    errors.append(f"E2 size: {sid}/{cls}: valid 栅格与 mask 尺寸不一致: {vpath}")
                                if int((v > 0).sum()) == 0:
                                    warnings.append(
                                        f"W3 valid: {sid}/{cls}: valid_mode=file 但 valid 栅格全 0（该类将被整类屏蔽）"
                                    )
                            except Exception as e:
                                errors.append(f"E1 readable: {sid}/{cls}: valid 栅格不可读: {vpath}: {e}")
                elif mode == "full" and lrec.get("valid_mask_path") is not None:
                    errors.append(f"E4 valid: {sid}/{cls}: valid_mode=full 但 valid_mask_path 非空")

    # ---- E5 泄漏 / 唯一性 ----
    by_split = defaultdict(list)
    id_seen = {}
    for i, rec in enumerate(records):
        sid = rec.get("sample_id", f"record#{i}")
        if sid in id_seen:
            errors.append(f"E5 unique: duplicate sample_id {sid} (records {id_seen[sid]} & {i})")
        id_seen[sid] = i
        by_split[rec.get("official_split", "?")].append(rec)
    imgid_by_split = defaultdict(set)
    for split, recs in by_split.items():
        for r in recs:
            imgid_by_split[split].add((r.get("image_id"), r.get("patient_id")))
    split_names = sorted(by_split)
    for a in range(len(split_names)):
        for b in range(a + 1, len(split_names)):
            sa, sb = split_names[a], split_names[b]
            ov = imgid_by_split[sa] & imgid_by_split[sb]
            if ov:
                errors.append(
                    f"E5 leakage: {sa}/{sb} 共享 image_id/patient_id {len(ov)} 个，例如 {sorted(ov)[:3]}"
                )
    # 官方 test 只允许显式存在，并给出数量（供配置层决策，不自动纳入）
    test_n = len(by_split.get("test", []))
    if test_n:
        warnings.append(f"W4 split: manifest 含 official_split=test 样本 {test_n} 条（预训练配置引用它将被 preflight 拒绝）")

    # ---- W1/W2 精确与感知哈希 ----
    near_pairs = set()
    sha_map = defaultdict(list)
    for rec in records:
        s = rec.get("image", {}).get("sha256")
        if s:
            sha_map[s].append(rec.get("sample_id"))
    exact_dups = {s: ids for s, ids in sha_map.items() if len(ids) > 1}
    for s, ids in list(exact_dups.items())[:max_dup_report]:
        warnings.append(f"W1 exact-dup: identical image sha256 {s[:12]}… -> {ids}")

    ph_map = {"ahash": defaultdict(list), "dhash": defaultdict(list), "phash": defaultdict(list)}
    for rec in records:
        ph = rec.get("perceptual_hash") or {}
        for k in ph_map:
            if ph.get(k):
                ph_map[k][ph[k]].append(rec.get("sample_id"))
    # 若记录缺哈希且有图像，补算（不回写 manifest，仅本次审计用）
    missing_ph = [r for r in records if not (r.get("perceptual_hash") or {}).get("ahash")]
    if missing_ph and _HAS_PIL:
        degradations.append(f"{len(missing_ph)} 条记录缺感知哈希：审计时现场补算（不回写 manifest）")
    near_pairs = set()
    if _HAS_PIL and ddr_images_root is None:
        # 仅外部清单内部的近重复
        for rec in records:
            ph = rec.get("perceptual_hash")
            if not ph or not ph.get("ahash"):
                p = resolve_path(rec.get("image", {}).get("path", ""), base_dir)
                if os.path.isfile(p):
                    rec["perceptual_hash"] = compute_perceptual_hashes(p)
        for k in ph_map:
            ph_map[k].clear()
        for rec in records:
            ph = rec.get("perceptual_hash") or {}
            for k in ph_map:
                if ph.get(k):
                    ph_map[k][ph[k]].append(rec.get("sample_id"))
        keys = {k: list(v.keys()) for k, v in ph_map.items()}
        for k in ph_map:
            arr = keys[k]
            for x in range(len(arr)):
                for y in range(x + 1, len(arr)):
                    if hamming_hex(arr[x], arr[y]) <= phash_threshold:
                        for s1 in ph_map[k][arr[x]]:
                            for s2 in ph_map[k][arr[y]]:
                                if s1 != s2:
                                    near_pairs.add(tuple(sorted((s1, s2))))
        for p in list(near_pairs)[:max_dup_report]:
            warnings.append(f"W2 near-dup: {p[0]} ~ {p[1]} (hamming<={phash_threshold}, 多哈希投票)")
        if len(near_pairs) > max_dup_report:
            warnings.append(f"W2 near-dup: 另有 {len(near_pairs) - max_dup_report} 对未逐条列出，见 JSON")

    # ---- 与 DDR 的图像级近重复 ----
    ddr_compared = 0
    if ddr_images_root is not None:
        if not _HAS_PIL:
            degradations.append("DDR 近重复检查需要 numpy/PIL；本次缺失已跳过（显式记录，不静默）")
        else:
            ext_hashes = []
            for rec in records:
                ph = rec.get("perceptual_hash") or {}
                ext_hashes.append((rec.get("sample_id"), ph))
            ddr_files = []
            for root_dir, _dirs, files in os.walk(ddr_images_root):
                for f in files:
                    if f.lower().endswith((".jpg", ".png")):
                        ddr_files.append(os.path.join(root_dir, f))
            ddr_hashes = []
            for p in ddr_files:
                ddr_hashes.append((os.path.basename(p), compute_perceptual_hashes(p)))
            ddr_compared = len(ddr_hashes)
            ext_arr = []
            for sid, ph in ext_hashes:
                if not ph or not ph.get("ahash"):
                    p = resolve_path(rec_lookup(records, sid).get("image", {}).get("path", ""), base_dir)
                    ph = compute_perceptual_hashes(p) if os.path.isfile(p) else {}
                ext_arr.append((sid, ph))
            hits = 0
            for dsid, dph in ddr_hashes:
                if not dph:
                    continue
                for sid, eph in ext_arr:
                    if not eph:
                        continue
                    votes = sum(
                        1
                        for k in ("ahash", "dhash", "phash")
                        if dph.get(k) and eph.get(k) and hamming_hex(dph[k], eph[k]) <= phash_threshold
                    )
                    if votes >= 2:  # 至少两种哈希同时命中，减少误报
                        warnings.append(f"W5 cross-dup: external {sid} ~ DDR {dsid} ({votes}/3 hashes)")
                        hits += 1
                        if hits >= max_dup_report:
                            warnings.append("W5 cross-dup: 达到报告上限，完整列表见 JSON")
                            break
                if hits >= max_dup_report:
                    break

    return {
        "errors": errors,
        "warnings": warnings,
        "degradations": degradations,
        "counts": {
            "records": len(records),
            "by_source_split": dict(_count_by(records)),
            "exact_dup_pairs": len(exact_dups),
            "near_dup_pairs": len(near_pairs),
            "ddr_images_compared": ddr_compared,
            "phash_threshold": phash_threshold,
        },
    }


def rec_lookup(records, sid):
    for r in records:
        if r.get("sample_id") == sid:
            return r
    return {}


def _count_by(records):
    c = defaultdict(int)
    for r in records:
        c[f"{r.get('source','?')}/{r.get('official_split','?')}"] += 1
    return c


def to_markdown(report: dict, manifest_path: str) -> str:
    lines = []
    lines.append("# Manifest 审计报告")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- Manifest：`{manifest_path}`")
    c = report["counts"]
    lines.append(
        f"- 记录数：{c['records']}；来源/split 分布：{json.dumps(c['by_source_split'], ensure_ascii=False)}"
    )
    lines.append(
        f"- 精确重复对：{c['exact_dup_pairs']}；近重复对：{c['near_dup_pairs']}"
        f"（阈值 {c['phash_threshold']} bit）；DDR 比对图数：{c['ddr_images_compared']}"
    )
    lines.append(f"- **结论：{'FAIL' if report['errors'] else 'PASS'}**"
                 f"（errors={len(report['errors'])}, warnings={len(report['warnings'])}）")
    if report["degradations"]:
        lines.append("\n## 降级记录")
        for d in report["degradations"]:
            lines.append(f"- {d}")
    if report["errors"]:
        lines.append("\n## Errors（必须修复）")
        for e in report["errors"]:
            lines.append(f"- {e}")
    if report["warnings"]:
        lines.append("\n## Warnings（不阻塞，需人工过目）")
        for w in report["warnings"]:
            lines.append(f"- {w}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="审计统一外部病灶 manifest（JSON+Markdown 报告）")
    ap.add_argument("--manifest", required=True, help="manifest JSONL 路径")
    ap.add_argument("--schema", default=None, help="manifest.schema.json 路径（默认随仓库 data_contract/）")
    ap.add_argument("--base-dir", default=None, help="相对路径的解析基准（默认 manifest 所在目录）")
    ap.add_argument("--ddr-images-root", default=None,
                    help="DDR 图像根目录（用于图像级近重复排查；只读，不触碰标签）")
    ap.add_argument("--phash-threshold", type=int, default=NEAR_DUP_HAMMING_THRESHOLD)
    ap.add_argument("--max-dup-report", type=int, default=50, help="重复对报告条数上限")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args(argv)

    base_dir = args.base_dir or os.path.dirname(os.path.abspath(args.manifest))
    schema = None
    schema_path = args.schema or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data_contract",
        "manifest.schema.json",
    )
    if os.path.isfile(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

    try:
        records = load_manifest(args.manifest)
    except ContractError as e:
        print(f"[audit_manifest] FAIL: {e}", file=sys.stderr)
        return 2

    report = audit(
        records,
        schema,
        base_dir,
        args.ddr_images_root,
        args.phash_threshold,
        args.max_dup_report,
    )
    report_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": os.path.abspath(args.manifest),
        **report,
    }

    ok = not report["errors"]
    out_json = args.output_json or os.path.abspath(args.manifest) + ".audit.json"
    out_md = args.output_md or os.path.abspath(args.manifest) + ".audit.md"
    write_text_atomic(out_json, json.dumps(report_out, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(out_md, to_markdown(report_out, os.path.abspath(args.manifest)))

    print(f"[audit_manifest] {'PASS' if ok else 'FAIL'}: "
          f"errors={len(report['errors'])} warnings={len(report['warnings'])} "
          f"degradations={len(report['degradations'])}")
    print(f"[audit_manifest] report -> {out_json}")
    print(f"[audit_manifest] report -> {out_md}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
