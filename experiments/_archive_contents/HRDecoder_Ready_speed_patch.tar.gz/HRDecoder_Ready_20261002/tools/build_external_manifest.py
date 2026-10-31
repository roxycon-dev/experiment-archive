# -*- coding: utf-8 -*-
r"""build_external_manifest.py — 从官方目录结构构建统一外部病灶 manifest。

支持的官方布局：
1. IDRiD（与旧转换器 D:\HRDecoder\tools\convert_dataset\idrid.py 相同的目录约定）：
       <root>/images/{train,test}/*.jpg
       <root>/labels/{train,test}/{EX,HE,MA,SE}/<stem>_<CLASS>.tif
2. MAPLES-DR（标注与 MESSIDOR 原图分别取得；两种标注布局自动探测，探测不出则 fail closed）：
       布局 A（逐类掩膜）：<maples_root>/anomalies/{split}/<stem>_<ANOM>.png
                          或 <maples_root>/anomalies/{split}/<ANOM>/<stem>.png
       布局 B（单文件编码）：<maples_root>/anomalies/{split}/<stem>.png，
          必须用 --anomaly-gray-map 显式提供官方灰度->类别映射，工具绝不猜颜色 class id
       原图：优先 <maples_root>/images/{split}/，否则必须给 --images-root（MESSIDOR 原图）

fail closed 原则：
- 原图缺失/不可读、已标注类的掩膜缺失/不可读、尺寸不对齐、掩膜非二值 → 报错退出（非零）；
- 多通道彩色掩膜：仅当三通道完全一致时按灰度处理，否则拒绝（禁止猜颜色 class id）；
- 官方 test 默认不纳入；需要时必须显式 `--include-splits train test`，
  且 official_split 字段逐字保留，preflight 会在预训练 manifest 中拦截 test。

用法示例：
  python tools/build_external_manifest.py --dataset idrid --root D:/data/IDRiD \
      --output manifests/idrid_train_v1.jsonl
  python tools/build_external_manifest.py --dataset maples-dr --root D:/data/MAPLES-DR \
      --images-root D:/data/MESSIDOR --output manifests/maples_train_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contract_common import (  # noqa: E402
    ALLOWED_OFFICIAL_SPLITS,
    LESION_CHANNEL_ORDER,
    MANIFEST_SCHEMA_VERSION,
    ContractError,
    compute_perceptual_hashes,
    sha256_file,
    write_jsonl_atomic,
    write_text_atomic,
)

try:
    import numpy as np
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def list_dir_safe(path: str) -> list:
    if not os.path.isdir(path):
        return []
    return sorted(os.listdir(path))


def fail(errors: list, msg: str) -> None:
    errors.append(msg)


def load_binary_mask(path: str, allow_multivalue: bool, errors: list):
    """读取掩膜并做二值解释。

    返回 (bool ndarray, nonzero_values, shape)。任何"需要猜"的情况都 fail closed：
    - 多通道且三通道不一致 → 拒绝（禁止从颜色猜 class id）；
    - 多个非零灰度值 → 默认拒绝，除非 --allow-multivalue-nonzero（按 >0 解释并记录）。
    """
    try:
        with Image.open(path) as im:
            raw = np.array(im)
    except Exception as e:
        fail(errors, f"mask unreadable: {path}: {e}")
        return None, None, None

    if raw.ndim == 3:
        # 多通道：三通道完全一致才允许退化为单通道；否则拒绝
        if raw.shape[2] >= 3 and bool(np.array_equal(raw[..., 0], raw[..., 1])) and bool(
            np.array_equal(raw[..., 0], raw[..., 2])
        ):
            raw = raw[..., 0]
        else:
            fail(
                errors,
                f"mask is multi-channel with inconsistent channels, refusing to guess "
                f"class id from colors: {path}",
            )
            return None, None, None
    elif raw.ndim != 2:
        fail(errors, f"mask is not a 2D image: {path}")
        return None, None, None

    vals = np.unique(raw)
    nz = vals[vals > 0]
    if nz.size == 0:
        return np.zeros(raw.shape, dtype=bool), [], raw.shape
    if nz.size == 1:
        return raw == nz[0], [int(nz[0])], raw.shape
    if allow_multivalue:
        return raw > 0, [int(v) for v in nz], raw.shape
    fail(
        errors,
        f"mask has {nz.size} distinct nonzero values {nz[:8].tolist()} (…): refusing to "
        f"interpret as binary without explicit flag: {path}",
    )
    return None, None, raw.shape


def image_size(path: str, errors: list):
    try:
        with Image.open(path) as im:
            w, h = im.size
        return int(w), int(h)
    except Exception as e:
        fail(errors, f"image unreadable: {path}: {e}")
        return None, None


def record_for_sample(*, source, source_version, official_split, image_id, patient_id,
                      patient_id_source, image_path, image_sha, width, height,
                      lesions, phashes, notes):
    """组装一条 manifest 记录（字段顺序固定，便于 diff）。"""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_id": f"{source}_{official_split}_{image_id}",
        "source": source,
        "source_version": source_version,
        "official_split": official_split,
        "patient_id": patient_id,
        "patient_id_source": patient_id_source,
        "image_id": image_id,
        "original_width": int(width),
        "original_height": int(height),
        "image": {
            "path": image_path,
            "sha256": image_sha,
            "format": os.path.splitext(image_path)[1].lstrip(".").lower(),
        },
        "lesions": lesions,
        "perceptual_hash": phashes if phashes else None,
        "notes": notes if notes else None,
    }


def path_arg(p: str, relative_to=None) -> str:
    """路径标准化：默认绝对路径；给 --relative-to 时存相对路径（可移植清单）。"""
    ap = os.path.abspath(p)
    if relative_to:
        try:
            return os.path.relpath(ap, os.path.abspath(relative_to)).replace("\\", "/")
        except ValueError:
            # 跨盘符等无法相对化时退回绝对路径
            return ap.replace("\\", "/")
    return ap.replace("\\", "/")


# ---------------------------------------------------------------------------
# IDRiD 适配器
# ---------------------------------------------------------------------------

def find_idrid_images(root: str, split: str, errors: list):
    img_dir = os.path.join(root, "images", split)
    files = [f for f in list_dir_safe(img_dir) if f.lower().endswith((".jpg", ".png"))]
    if not files:
        fail(errors, f"IDRiD: no images found under {img_dir}")
    return img_dir, files


def build_idrid(args, errors: list):
    """IDRiD：官方四个类全部提供逐类 TIF，缺任何一张都 fail closed。"""
    records = []
    for split in args.include_splits:
        if split not in ("train", "test"):
            fail(errors, f"IDRiD official split must be train/test, got: {split}")
            continue
        img_dir, files = find_idrid_images(args.root, split, errors)
        for fname in files:
            stem = os.path.splitext(fname)[0]
            image_path = os.path.join(img_dir, fname)
            w, h = image_size(image_path, errors)
            if w is None:
                continue
            lesions = {}
            ok = True
            for cls in LESION_CHANNEL_ORDER:
                # 旧转换器命名约定：<stem>_<CLASS>.tif（idrid.py:58）
                mask_path = os.path.join(args.root, "labels", split, cls, f"{stem}_{cls}.tif")
                if not os.path.isfile(mask_path):
                    fail(errors, f"IDRiD: official mask missing (fail closed): {mask_path}")
                    ok = False
                    continue
                mask, nz, mshape = load_binary_mask(
                    mask_path, args.allow_multivalue_nonzero, errors
                )
                if mask is None:
                    ok = False
                    continue
                if (mshape[1], mshape[0]) != (w, h):
                    fail(
                        errors,
                        f"IDRiD: mask size {mshape[1]}x{mshape[0]} != image {w}x{h}: {mask_path}",
                    )
                    ok = False
                    continue
                lesions[cls] = {
                    "annotated": True,  # 官方逐类 TIF 存在即视为已标注
                    "valid_mode": "full",
                    "mask_path": path_arg(mask_path, args.relative_to),
                    "mask_sha256": sha256_file(mask_path),
                    "valid_mask_path": None,
                    "positive_pixels": int(mask.sum()),
                }
            if not ok:
                continue
            phashes = compute_perceptual_hashes(image_path) if args.compute_phash else {}
            notes = "built_from=idrid_official_layout; positive_policy=" + (
                "nonzero_multivalue" if args.allow_multivalue_nonzero else "single_nonzero_value"
            )
            records.append(
                record_for_sample(
                    source="IDRiD",
                    source_version=args.source_version,
                    official_split=split,
                    image_id=stem,
                    patient_id=None,  # IDRiD 公开版不提供患者 ID，禁止冒充
                    patient_id_source=None,
                    image_path=path_arg(image_path, args.relative_to),
                    image_sha=sha256_file(image_path),
                    width=w,
                    height=h,
                    lesions=lesions,
                    phashes=phashes,
                    notes=notes,
                )
            )
    return records


# ---------------------------------------------------------------------------
# MAPLES-DR 适配器
# ---------------------------------------------------------------------------

# MAPLES-DR 的病灶缩写中只有这四个映射进四通道；其余（IRMA/VB/NV/FX 等）记录不映射
MAPLES_TARGET_ANOM = {c: c for c in LESION_CHANNEL_ORDER}  # EX,HE,SE,MA 同名


def detect_maples_layout(anom_root: str, split: str):
    """探测 MAPLES-DR 标注布局。返回 ('per-anomaly-flat'|'per-anomaly-dirs'|'single-file'|None, 根目录)。"""
    sdir = os.path.join(anom_root, split)
    if not os.path.isdir(sdir):
        return None, None
    entries = list_dir_safe(sdir)
    png_files = [e for e in entries if e.lower().endswith(".png")]
    subdirs = [e for e in entries if os.path.isdir(os.path.join(sdir, e))]
    if subdirs and all(d.upper() in ("EX", "HE", "SE", "MA", "IRMA", "VB", "NV", "FX") for d in subdirs):
        return "per-anomaly-dirs", sdir
    stems = {os.path.splitext(e)[0] for e in png_files}
    suffix_hits = sum(1 for s in stems if s.rsplit("_", 1)[-1].upper() in ("EX", "HE", "SE", "MA", "IRMA", "VB", "NV", "FX"))
    if png_files and suffix_hits >= max(1, len(stems) // 2):
        return "per-anomaly-flat", sdir
    if png_files:
        return "single-file", sdir
    return None, sdir


def maples_mask_path(layout, sdir, stem, anom):
    """按布局返回候选掩膜路径（可能不存在）。"""
    if layout == "per-anomaly-dirs":
        return os.path.join(sdir, anom, f"{stem}.png")
    if layout == "per-anomaly-flat":
        return os.path.join(sdir, f"{stem}_{anom}.png")
    return None


def build_maples(args, errors: list):
    records = []
    layout_all = None
    for split in args.include_splits:
        if split not in ("train", "test"):
            fail(errors, f"MAPLES-DR official split must be train/test, got: {split}")
            continue

        # ---- 标注布局探测 ----
        anom_root = os.path.join(args.root, "anomalies") if os.path.isdir(
            os.path.join(args.root, "anomalies")
        ) else args.root
        layout, sdir = detect_maples_layout(anom_root, split)
        if layout is None:
            fail(errors, f"MAPLES-DR: cannot detect annotation layout under {anom_root}/{split}")
            continue
        layout_all = layout_all or layout

        # ---- 原图来源 ----
        img_dir = None
        cand = os.path.join(args.root, "images", split)
        if os.path.isdir(cand) and list_dir_safe(cand):
            img_dir = cand
        elif args.images_root:
            img_dir = os.path.join(args.images_root, split)
            if not os.path.isdir(img_dir):
                img_dir = args.images_root
        if img_dir is None:
            fail(
                errors,
                f"MAPLES-DR: no images under {cand} and --images-root not given; "
                f"MESSIDOR fundus images must be obtained separately (see data_contract/README.md)",
            )
            continue
        img_files = [f for f in list_dir_safe(img_dir) if f.lower().endswith((".jpg", ".png"))]
        if not img_files:
            fail(errors, f"MAPLES-DR: no images found under {img_dir}")
            continue

        # ---- 布局 B：必须显式给官方灰度映射 ----
        gray_map = {}
        if layout == "single-file":
            if not args.anomaly_gray_map:
                fail(
                    errors,
                    "MAPLES-DR: single-file encoded masks detected. This tool refuses to guess "
                    "gray-level class ids; pass --anomaly-gray-map from the official "
                    "documentation, e.g. --anomaly-gray-map MA=60,HE=120,EX=180,SE=240",
                )
                continue
            for kv in args.anomaly_gray_map.split(","):
                k, v = kv.split("=")
                if k.strip().upper() not in LESION_CHANNEL_ORDER:
                    fail(errors, f"--anomaly-gray-map: unknown anomaly key {k!r}")
                gray_map[k.strip().upper()] = int(v)

        # ---- 逐图构建 ----
        for fname in img_files:
            stem = os.path.splitext(fname)[0]
            image_path = os.path.join(img_dir, fname)
            w, h = image_size(image_path, errors)
            if w is None:
                continue
            lesions = {}
            ok = True
            for cls in LESION_CHANNEL_ORDER:
                if layout in ("per-anomaly-dirs", "per-anomaly-flat"):
                    mask_path = maples_mask_path(layout, sdir, stem, cls)
                    if not os.path.isfile(mask_path):
                        if args.treat_missing_as_unannotated:
                            # 显式降级：缺失掩膜记录为"未标注"，绝不当阴性
                            lesions[cls] = {
                                "annotated": False,
                                "valid_mode": "none",
                                "mask_path": None,
                                "mask_sha256": None,
                                "valid_mask_path": None,
                                "positive_pixels": None,
                            }
                            continue
                        fail(
                            errors,
                            f"MAPLES-DR: anomaly mask missing (fail closed; use "
                            f"--treat-missing-as-unannotated to record as unannotated): {mask_path}",
                        )
                        ok = False
                        continue
                    mask, nz, mshape = load_binary_mask(
                        mask_path, args.allow_multivalue_nonzero, errors
                    )
                    if mask is None:
                        ok = False
                        continue
                    if (mshape[1], mshape[0]) != (w, h):
                        fail(errors, f"MAPLES-DR: mask size mismatch vs image: {mask_path}")
                        ok = False
                        continue
                    lesions[cls] = {
                        "annotated": True,
                        "valid_mode": "full",
                        "mask_path": path_arg(mask_path, args.relative_to),
                        "mask_sha256": sha256_file(mask_path),
                        "valid_mask_path": None,
                        "positive_pixels": int(mask.sum()),
                    }
                else:  # single-file
                    pass  # 在下方统一解析
            if layout == "single-file" and ok:
                fp = os.path.join(sdir, f"{stem}.png")
                if not os.path.isfile(fp):
                    fail(errors, f"MAPLES-DR: encoded mask missing: {fp}")
                    ok = False
                else:
                    try:
                        with Image.open(fp) as im:
                            raw = np.array(im)
                    except Exception as e:
                        fail(errors, f"MAPLES-DR: mask unreadable: {fp}: {e}")
                        ok = False
                        raw = None
                    if raw is not None:
                        if raw.ndim == 3:
                            if raw.shape[2] >= 3 and bool(
                                np.array_equal(raw[..., 0], raw[..., 1])
                            ) and bool(np.array_equal(raw[..., 0], raw[..., 2])):
                                raw = raw[..., 0]
                            else:
                                fail(errors, f"MAPLES-DR: multi-channel encoded mask, refusing: {fp}")
                                ok = False
                        if ok:
                            if (raw.shape[1], raw.shape[0]) != (w, h):
                                fail(errors, f"MAPLES-DR: mask size mismatch vs image: {fp}")
                                ok = False
                            else:
                                for cls in LESION_CHANNEL_ORDER:
                                    gval = gray_map[cls]
                                    mask = raw == gval
                                    lesions[cls] = {
                                        "annotated": True,
                                        "valid_mode": "full",
                                        "mask_path": path_arg(fp, args.relative_to),
                                        "mask_sha256": sha256_file(fp),
                                        "valid_mask_path": None,
                                        "positive_pixels": int(mask.sum()),
                                    }
            if not ok:
                continue
            phashes = compute_perceptual_hashes(image_path) if args.compute_phash else {}
            note_bits = [
                f"built_from=maples_dr_layout_{layout}",
                "patient_id_is_image_basename_fallback",
            ]
            if layout == "single-file":
                note_bits.append("encoded_mask_gray_map=" + json.dumps(gray_map, sort_keys=True))
            records.append(
                record_for_sample(
                    source="MAPLES-DR",
                    source_version=args.source_version,
                    official_split=split,
                    image_id=stem,
                    # MESSIDOR 命名不含可靠患者字段：用 basename 兜底并在来源字段明示
                    patient_id=stem,
                    patient_id_source="image_basename_fallback",
                    image_path=path_arg(image_path, args.relative_to),
                    image_sha=sha256_file(image_path),
                    width=w,
                    height=h,
                    lesions=lesions,
                    phashes=phashes,
                    notes="; ".join(note_bits),
                )
            )
    return records


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="构建统一外部病灶 manifest（fail closed）")
    ap.add_argument("--dataset", required=True, choices=["idrid", "maples-dr"])
    ap.add_argument("--root", required=True, help="数据集根目录（官方布局）")
    ap.add_argument("--images-root", default=None, help="MAPLES-DR 的 MESSIDOR 原图根目录（单独取得）")
    ap.add_argument("--output", required=True, help="输出 manifest JSONL 路径")
    ap.add_argument("--summary-output", default=None, help="构建摘要 JSON（默认 <output>.summary.json）")
    ap.add_argument("--include-splits", nargs="+", default=["train"],
                    help="官方 split 白名单，默认仅 train；纳入 test 必须显式给出")
    ap.add_argument("--source-version", default=None, help="数据集版本标识；未知则留空（存 null）")
    ap.add_argument("--allow-multivalue-nonzero", action="store_true",
                    help="掩膜出现多个非零灰度值时按 >0 解释（默认 fail closed）")
    ap.add_argument("--treat-missing-as-unannotated", action="store_true",
                    help="MAPLES-DR 缺失异常掩膜记录为未标注 valid=none（默认 fail closed）")
    ap.add_argument("--anomaly-gray-map", default=None,
                    help="布局 B 的官方灰度->类别映射，如 MA=60,HE=120,EX=180,SE=240")
    ap.add_argument("--compute-phash", action="store_true", default=True,
                    help="构建时计算感知哈希（PIL 缺失时自动降级）")
    ap.add_argument("--no-compute-phash", dest="compute_phash", action="store_false")
    ap.add_argument("--relative-to", default=None, help="把清单内路径写成相对该目录的路径（可移植清单）")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    errors: list = []

    if not _HAS_PIL:
        errors.append("PIL/numpy unavailable on this interpreter; cannot build manifest")

    # 官方 test 防呆：默认只有 train；显式包含 test 允许但必须自知（official_split 保留）
    if args.include_splits == ["train"]:
        pass
    else:
        unknown = [s for s in args.include_splits if s not in ALLOWED_OFFICIAL_SPLITS]
        if unknown:
            errors.append(f"unknown official splits: {unknown}; allowed {ALLOWED_OFFICIAL_SPLITS}")

    if not os.path.isdir(args.root):
        errors.append(f"dataset root not found: {args.root}")

    records = []
    if not errors:
        if args.dataset == "idrid":
            records = build_idrid(args, errors)
        else:
            records = build_maples(args, errors)

    # sample_id 唯一性
    seen = set()
    for r in records:
        if r["sample_id"] in seen:
            errors.append(f"duplicate sample_id built: {r['sample_id']}")
        seen.add(r["sample_id"])

    if errors:
        # fail closed：输出明确错误清单后以非零码退出，不产出半成品 manifest
        print("[build_external_manifest] FAILED with %d error(s):" % len(errors), file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        print(
            "未写出 manifest（fail closed）。请补齐缺失资产或显式使用对应降级旗标后重试。",
            file=sys.stderr,
        )
        return 2

    write_jsonl_atomic(args.output, records)
    summary_path = args.summary_output or (args.output + ".summary.json")
    counts = {}
    for r in records:
        key = (r["source"], r["official_split"])
        counts[f"{key[0]}/{key[1]}"] = counts.get(f"{key[0]}/{key[1]}", 0) + 1
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "include_splits": list(args.include_splits),
        "records": len(records),
        "counts": counts,
        "output": os.path.abspath(args.output),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "notes": "official test 仅在显式 --include-splits 中出现；official_split 字段逐字保留",
    }
    write_text_atomic(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"[build_external_manifest] OK: {len(records)} records -> {args.output}")
    print(f"[build_external_manifest] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
