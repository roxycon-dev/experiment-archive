# storage_preflight.py
"""Prep-only storage gate: estimate cache (image byte spans) + checkpoints vs free disk. No writes except --out-json."""
import argparse, json, os, shutil, sys
from pathlib import Path

def image_bytes(data_root: Path, split: str = "train") -> tuple:
    """Estimate cache bytes from images/<split> only (no recursive scan of data_root)."""
    from PIL import Image
    total = 0
    n = 0
    img_dir = Path(data_root) / "images" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"missing image directory: {img_dir}")
    for p in sorted(img_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            continue
        with Image.open(p) as im:
            w, h = im.size  # header only; PIL lazy-loads
        total += w * h * 16  # reference HWC float32 x4 channels
        n += 1
    return total, n


def check_storage(data_root, cache_dir, run_dir, student_ckpt, reference_ckpt,
                  num_checkpoints=24, headroom_frac=0.20, headroom_gib=2.0,
                  expected_count=454):
    cache_dir = Path(cache_dir); run_dir = Path(run_dir)
    missing = []
    for p, label in ((data_root, "data-root"), (student_ckpt, "student-checkpoint"),
                     (reference_ckpt, "reference-checkpoint")):
        if not Path(p).exists():
            missing.append(label)
    if missing:
        return {"status": "block", "reason": f"missing inputs: {missing}"}

    img_bytes, n_imgs = image_bytes(data_root)
    student_sz = Path(student_ckpt).stat().st_size
    ref_sz = Path(reference_ckpt).stat().st_size
    per_ckpt = 2 * (student_sz + ref_sz)  # student+reference, conservative
    ckpt_bytes = per_ckpt * num_checkpoints

    required = img_bytes + ckpt_bytes
    required *= 1.0 + headroom_frac
    required += int(headroom_gib * (1 << 30))

    def free_for(d: Path) -> int:
        d = Path(d).resolve()
        while not d.exists():
            d = d.parent  # nearest existing parent
        return shutil.disk_usage(str(d)).free, str(d)

    cache_free, cache_fs = free_for(cache_dir)
    run_free, run_fs = free_for(run_dir)
    same_fs = os.stat(cache_fs).st_dev == os.stat(run_fs).st_dev

    if same_fs:
        free = min(cache_free, run_free)
        ok = required <= free
    else:
        # cache holds image spans; run holds checkpoints
        ok = (img_bytes <= cache_free) and (ckpt_bytes <= run_free)
        free = cache_free + run_free

    ok = ok and n_imgs >= expected_count
    return {
        "status": "ok" if ok else "block",
        "images": {"count": n_imgs, "bytes": img_bytes,
                   "expected_count": expected_count,
                   "expected_count_met": n_imgs >= expected_count},
        "checkpoints": {"per_ckpt_bytes": per_ckpt, "count": num_checkpoints, "bytes": ckpt_bytes},
        "required_bytes": int(required),
        "free_bytes": int(free),
        "same_filesystem": same_fs,
        "cache_free_bytes": int(cache_free), "run_free_bytes": int(run_free),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--student-checkpoint", required=True)
    ap.add_argument("--reference-checkpoint", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--expected-count", type=int, default=454,
                    help="minimum image count in images/train for status ok (production: 454)")
    a = ap.parse_args()

    report = check_storage(a.data_root, a.cache_dir, a.run_dir,
                           a.student_checkpoint, a.reference_checkpoint,
                           expected_count=a.expected_count)
    out = Path(a.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    sys.exit(0 if report["status"] == "ok" else 2)

if __name__ == "__main__":
    main()
