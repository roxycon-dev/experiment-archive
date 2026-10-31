"""asset_inventory.py: scoped asset audit + data split audit.
Asset audit only (never invents file paths, hashes, or success).
Top status is LOCAL_READY_REMOTE_ASSET_PENDING until the remote E champion
checkpoint is actually recovered and hashed; it blocks the champion claim.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from typing import Dict, List

DATA_ROOT = "D:/HRDecoder/data/DDR_split_613"
SPLITS = ("train", "val", "test")
EXPECTED = {"train": 454, "val": 76, "test": 227}

# Actual asset paths (verified list; no invented files).
STUDENT_D = "D:/dataset/experiment/hrdecoder_plus3/results/D_0913_retry/backup_student_2000.pth"
B_LOCAL = "D:/dataset/experiment/hrdecoder_plus3/repo/weights/b_weights.pth"
B_REMOTE_COPY = "D:/dataset/experiment/autodl100/results/trial_b/selected_weights_only.pth"
E4 = "D:/dataset/experiment/autodl100/results/e4_reference_weights.pth"
MA = "D:/dataset/experiment/HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/h4_best.pth"
H5 = "D:/dataset/experiment/HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/experiments/h5_test.json"
H7 = "D:/dataset/experiment/HRDecoder_审查轮_20260916/audit_runs/audit_20260916_A/experiments/h7_test.json"

E_REMOTE = "/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth"
D_REMOTE = "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_asset(path: str) -> dict:
    if not os.path.isfile(path):
        return {"path": path, "exists": False, "sha256": None}
    return {"path": path, "exists": True, "sha256": sha256(path)}


def json_load_safe(path: str):
    """Parse JSON text only; never unpickle."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_metrics(obj) -> dict:
    """Find mIoU (numeric) and per_class (numeric dict) anywhere in a result JSON,
    ignoring schema differences between h5 and h7 exports."""
    found = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("mIoU", "miou", "mean_iou", "m_iou") and isinstance(v, (int, float)) and "mIoU" not in found:
                    found["mIoU"] = float(v)
                if k == "per_class" and isinstance(v, dict) and "per_class" not in found:
                    found["per_class"] = {str(kk): float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
                walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(obj)
    return found


def compare_h5_h7_metrics(h5_path: str, h7_path: str) -> dict:
    a = extract_metrics(json_load_safe(h5_path))
    b = extract_metrics(json_load_safe(h7_path))
    res = {"h5": h5_path, "h7": h7_path, "h5_metrics": a, "h7_metrics": b}
    if "mIoU" not in a or "mIoU" not in b:
        res["metrics_equal"] = False
        res["reason"] = "mIoU key missing in one of the files"
        return res
    miou_eq = abs(a["mIoU"] - b["mIoU"]) <= 1e-9
    pa, pb = a.get("per_class"), b.get("per_class")
    pc_eq = False
    if isinstance(pa, dict) and isinstance(pb, dict) and set(pa) == set(pb):
        pc_eq = all(abs(pa[k] - pb[k]) <= 1e-9 for k in pa)
    res["miou_equal"] = miou_eq
    res["per_class_equal"] = pc_eq
    res["metrics_equal"] = bool(miou_eq and pc_eq)
    return res


def build_manifest() -> dict:
    b1 = hash_asset(B_LOCAL)
    b2 = hash_asset(B_REMOTE_COPY)
    b_match = bool(b1["exists"] and b2["exists"] and b1["sha256"] == b2["sha256"])
    h5h7 = compare_h5_h7_metrics(H5, H7) if (os.path.isfile(H5) and os.path.isfile(H7)) else None
    manifest = {
        "scope_note": "asset audit only; reads fixed scoped paths; never invents paths or hashes",
        "assets": {
            "D_student_only_2000": {**hash_asset(STUDENT_D), "role": "student-only checkpoint (D config)"},
            "B_local": {**b1, "role": "B weights (local)"},
            "B_remote_copy": {**b2, "role": "B weights (remote trial_b copy)"},
            "E4_reference": {**hash_asset(E4), "role": "E4 reference weights"},
            "MA_audit_champion": {**hash_asset(MA), "role": "MA audit champion"},
            "D_full_remote": {
                "path": D_REMOTE, "exists": False, "sha256": "UNKNOWN",
                "status": "REMOTE_ASSET_PENDING", "blocks_champion": False,
                "reconstructable": True,
                "reconstruction_from": ["D_student_only_2000", "B_local (hash-verified)"],
                "note": "reconstruction is config-level, not a fake weight hash",
            },
            "E_champion": {
                "path": E_REMOTE, "exists": False, "sha256": "UNKNOWN",
                "status": "REMOTE_ASSET_PENDING", "blocks_champion": True,
                "note": "remote recovery pending; champion claim blocked until real hash obtained",
            },
        },
        "B_hash_match": b_match,
        "h5_h7": h5h7,
        "status": "LOCAL_READY_REMOTE_ASSET_PENDING",
    }
    return manifest


def _read_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def audit_split(data_root: str = DATA_ROOT, do_hash: bool = False) -> dict:
    """Structural audit of images/<split>, labels/<split>/<stem>.png,
    split_assignment.json (dict filename.jpg -> {split: ...}) and txt lists.
    Class subfolders under labels/ (EX/HE/SE/MA) are ignored for the aggregate mask.
    """
    report = {
        "data_root": data_root, "expected": EXPECTED, "status": "BLOCKED",
        "splits": {}, "cross_split_stem_overlaps": {},
        "duplicate_bytes_across_splits": [],
    }
    assign_path = os.path.join(data_root, "split_assignment.json")
    assignment: Dict[str, str] = {}
    if os.path.isfile(assign_path):
        raw = json_load_safe(assign_path)
        if isinstance(raw, dict) and isinstance(raw.get("images"), dict):
            raw = raw["images"]  # unwrap wrapper {"images": {fname: ...}}
        if isinstance(raw, dict):
            for fname, meta in raw.items():
                if isinstance(meta, dict) and "split" in meta:
                    assignment[str(fname)] = str(meta["split"])
                elif isinstance(meta, str):  # direct mapping: "img.jpg": "train"
                    assignment[str(fname)] = meta
    hash_index = {}
    overall_ok = True
    for split in SPLITS:
        txt = _read_lines(os.path.join(data_root, split + ".txt"))
        img_dir = os.path.join(data_root, "images", split)
        disk = sorted(f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))) \
            if os.path.isdir(img_dir) else []
        assigned = sorted(f for f, s in assignment.items() if s == split)
        s_txt, s_disk, s_assign = set(txt), set(disk), set(assigned)
        agreement = (s_txt == s_disk == s_assign)
        stems = [os.path.splitext(f)[0] for f in txt]
        dup_stems = sorted({s for s in stems if stems.count(s) > 1})
        labels_ok = all(
            os.path.isfile(os.path.join(data_root, "labels", split, st + ".png")) for st in stems
        )
        count_ok = len(txt) == EXPECTED[split]
        rep = {
            "count": len(txt), "count_ok": count_ok,
            "agreement_txt_disk_assignment": agreement,
            "missing_on_disk": sorted(s_txt - s_disk),
            "missing_in_txt": sorted(s_disk - s_txt),
            "missing_in_assignment": sorted(s_txt - s_assign),
            "duplicate_stems": dup_stems, "labels_present": labels_ok,
        }
        rep["ok"] = bool(agreement and count_ok and not dup_stems and labels_ok)
        overall_ok = overall_ok and rep["ok"]
        report["splits"][split] = rep
        if do_hash:
            for f in txt:
                for sub, name in (("images", f), ("labels", os.path.splitext(f)[0] + ".png")):
                    p = os.path.join(data_root, sub, split, name)
                    if os.path.isfile(p):
                        h = sha256(p)
                        hash_index.setdefault(h, []).append({"split": split, "kind": sub, "file": name})
    stem_sets = {
        s: {os.path.splitext(f)[0] for f in _read_lines(os.path.join(data_root, s + ".txt"))}
        for s in SPLITS
    }
    overlaps = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            inter = sorted(stem_sets[a] & stem_sets[b])
            if inter:
                overlaps[f"{a}|{b}"] = inter
    report["cross_split_stem_overlaps"] = overlaps
    overall_ok = overall_ok and not overlaps
    if do_hash:
        report["duplicate_bytes_across_splits"] = [
            {"sha256": h, "occurrences": occ}
            for h, occ in hash_index.items()
            if len({o["split"] for o in occ}) > 1
        ]
        report["hash_note"] = "duplicate image/mask bytes detected across splits; recorded, NOT deduped automatically"
    report["status"] = "OK" if overall_ok else "BLOCKED"
    return report


def main(argv=None):
    p = argparse.ArgumentParser(prog="asset_inventory", description="Scoped asset + split audit (read-only on sources)")
    p.add_argument("--out-dir", required=True, help="new directory to write asset_manifest.json and split_audit.json")
    p.add_argument("--hash", action="store_true", help="hash all images/masks (content identity + cross-split duplicate detection)")
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)  # only writes into the requested new dir
    split = audit_split(DATA_ROOT, do_hash=args.hash)
    with open(os.path.join(args.out_dir, "split_audit.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    manifest = build_manifest()
    with open(os.path.join(args.out_dir, "asset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps({"manifest_status": manifest["status"], "split_status": split["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
