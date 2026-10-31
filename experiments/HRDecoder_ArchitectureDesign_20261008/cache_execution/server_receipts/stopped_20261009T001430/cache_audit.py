"""CPU-only integrity, coverage, signed contrast energy and covariance audit."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from cache_common import ROOT, DATA, CLASSES, samples, sha, array_sha, read_json, write_json

def inspect(row, indexes):
    sid = row["sid"]
    probabilities = []
    file_bytes = 0
    for member in ("A", "U", "S"):
        entry = indexes[member]["entries"][sid]
        path = ROOT / "cache/train" / member / entry["file"]
        assert sha(path) == entry["file_sha256"], sid + ": file SHA mismatch"
        with np.load(path, allow_pickle=False) as stored: value = stored["prob"]
        assert value.dtype == np.float32 and list(value.shape) == [4] + row["hw"]
        assert array_sha(value) == entry["probability_sha256"], sid + ": tensor SHA mismatch"
        assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
        assert entry["image_sha256"] == row["image_sha256"] and entry["label_sha256"] == row["label_sha256"]
        probabilities.append(value)
        file_bytes += entry["bytes"]
    with Image.open(DATA / "labels/train" / (sid + ".png")) as opened: label = np.asarray(opened)
    valid = label != 255
    a, u, s = probabilities
    class_rows = []
    max_inverse = 0.
    for c, name in enumerate(CLASSES):
        av, uv, sv = a[c][valid], u[c][valid], s[c][valid]
        du = uv - av
        p0 = .75 * av + .25 * uv
        ds = sv - p0
        target = .75 * p0 + .25 * sv
        ar = target - .25 * du - .25 * ds
        ur = ar + du
        sr = ar + .25 * du + ds
        if len(av):
            max_inverse = max(max_inverse, float(np.max(np.abs(ar-av))),
                              float(np.max(np.abs(ur-uv))), float(np.max(np.abs(sr-sv))))
        x, y = du.astype(np.float64), ds.astype(np.float64)
        class_rows.append({"class": name, "n": len(x), "gt_positive": int(((label == c+1) & valid).sum()),
                           "sum_du": float(x.sum()), "sum_ds": float(y.sum()),
                           "sum_du2": float(np.dot(x, x)), "sum_ds2": float(np.dot(y, y)),
                           "sum_du_ds": float(np.dot(x, y)),
                           "sum_abs_du": float(np.abs(x).sum()), "sum_abs_ds": float(np.abs(y).sum()),
                           "positive_du": int((du > 1e-7).sum()), "negative_du": int((du < -1e-7).sum()),
                           "positive_ds": int((ds > 1e-7).sum()), "negative_ds": int((ds < -1e-7).sum()),
                           "opposite_sign": int(((du > 1e-7) & (ds < -1e-7) | (du < -1e-7) & (ds > 1e-7)).sum()),
                           "u_sum": float(uv.sum(dtype=np.float64)), "u_square_sum": float(np.square(uv.astype(np.float64)).sum()),
                           "s_sum": float(sv.sum(dtype=np.float64)), "s_square_sum": float(np.square(sv.astype(np.float64)).sum())})
    assert max_inverse < 1e-6, sid + ": teacher coordinate reconstruction mismatch"
    return {"sid": sid, "hw": row["hw"], "valid_pixels": int(valid.sum()), "class_statistics": class_rows,
            "max_algebra_inverse_abs_difference": max_inverse, "compressed_bytes": file_bytes}

def main():
    started = time.monotonic()
    rows = samples("train")
    assert len(rows) == 454
    indexes = {}
    for member in ("A", "U", "S"):
        index = read_json(ROOT / "cache/train" / member / "index.json")
        assert index["status"] == "COMPLETE" and index["completed_images"] == index["expected_images"] == 454
        assert index["identity"]["split"] == "train" and index["identity"]["dtype"] == "float32"
        assert index["identity"]["channels"] == CLASSES
        assert set(index["entries"]) == {row["sid"] for row in rows}
        indexes[member] = index
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(lambda row: inspect(row, indexes), rows):
            results.append(record)
            if len(results) % 10 == 0 or len(results) == len(rows):
                progress = {"stage": "CPU_CACHE_AUDIT", "completed": len(results), "expected": len(rows),
                            "seconds": time.monotonic() - started}
                write_json(ROOT / "reports/cache_audit_progress.json", progress)
                print(json.dumps(progress), flush=True)
    summary = []
    for c, name in enumerate(CLASSES):
        records = [record["class_statistics"][c] for record in results]
        total = {key: sum(record[key] for record in records) for key in records[0] if key != "class"}
        n = max(total["n"], 1)
        mx, my = total["sum_du"] / n, total["sum_ds"] / n
        covariance = np.array([[total["sum_du2"]/n - mx*mx, total["sum_du_ds"]/n - mx*my],
                               [total["sum_du_ds"]/n - mx*my, total["sum_ds2"]/n - my*my]])
        eigenvalues = np.linalg.eigvalsh(covariance)
        threshold = max(float(eigenvalues.max()), 1e-12) * 1e-6
        summary.append({"class": name, "valid_pixels": total["n"], "gt_positive_pixels": total["gt_positive"],
                        "du_mean": mx, "ds_mean": my,
                        "du_mse_energy": total["sum_du2"]/n, "ds_mse_energy": total["sum_ds2"]/n,
                        "du_mean_absolute": total["sum_abs_du"]/n, "ds_mean_absolute": total["sum_abs_ds"]/n,
                        "du_std": float(np.sqrt(max(covariance[0, 0], 0))), "ds_std": float(np.sqrt(max(covariance[1, 1], 0))),
                        "U_probability_std": float(np.sqrt(max(total["u_square_sum"]/n - (total["u_sum"]/n)**2, 0))),
                        "S_probability_std": float(np.sqrt(max(total["s_square_sum"]/n - (total["s_sum"]/n)**2, 0))),
                        "du_positive_fraction": total["positive_du"]/n, "du_negative_fraction": total["negative_du"]/n,
                        "ds_positive_fraction": total["positive_ds"]/n, "ds_negative_fraction": total["negative_ds"]/n,
                        "opposite_sign_fraction": total["opposite_sign"]/n,
                        "covariance": covariance.tolist(), "covariance_eigenvalues": eigenvalues.tolist(),
                        "covariance_rank_relative_tolerance_1e_minus6": int((eigenvalues > threshold).sum()),
                        "images_with_gt_positive": sum(record["gt_positive"] > 0 for record in records)})
    parity = read_json(ROOT / "reports/S20_canary_parity.json")
    assert parity["max_abs_difference"] <= 1e-5
    compressed = sum(record["compressed_bytes"] for record in results)
    uncompressed = sum(row["hw"][0]*row["hw"][1] for row in rows)*48
    result = {"status": "PASS", "stage": "TRAIN_CACHE_READY", "images_per_member": 454,
              "members": ["A", "U", "S"], "dtype": "float32", "channel_order": CLASSES,
              "coordinates": "original_image", "teacher_predictions_are_in_sample": True, "OOF": False,
              "all_file_and_tensor_SHA_verified": True, "all_lossless_roundtrips_verified": True,
              "member_identity_sha256": {member: index["identity_sha256"] for member, index in indexes.items()},
              "compressed_bytes": compressed, "uncompressed_bytes": uncompressed,
              "compression_ratio": compressed/uncompressed, "data_disk_free_bytes": shutil.disk_usage(ROOT).free,
              "S20_locked_val_canary_parity": parity, "class_statistics": summary,
              "per_image": results, "seconds": time.monotonic()-started,
              "no_training_performed": True,
              "interpretation": "Contrast energy/rank verifies targets exist; it does not prove useful knowledge, generalization or improvement."}
    write_json(ROOT / "reports/train_cache_audit.json", result)
    write_json(ROOT / "CACHE_READY.json", {key: value for key, value in result.items() if key != "per_image"})
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}), flush=True)

if __name__ == "__main__": main()
