"""Inventory reusable local assets and estimate incremental v3 preparation cost."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil
from PIL import Image


HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent
EXPERIMENT = DESIGN.parent
IDENTITY = EXPERIMENT / "HRDecoder_FinalDecision_20261005/execution/server_results/reports/data_identity.json"
DATA = Path("D:/HRDecoder/data/DDR_split_613")
CODE = EXPERIMENT / "autodl100/work/HRDecoder"
WEIGHT_MANIFEST = EXPERIMENT / "HRDecoder_Ready_20261002/best_pipeline/local_weight_manifest.json"
S20 = EXPERIMENT / "HRDecoder_FinalDecision_20261005/execution/server_results/runs/S_seed42/checkpoint_u20000.pth"
W48 = Path("C:/Users/GuanXingRen/.cache/torch/hub/checkpoints/hrnetv2_w48-d2186c55.pth")
GIB = 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
    split_inventory = {}
    for split, section in identity["splits"].items():
        total_bytes = 0
        pixels = 0
        max_pixels = 0
        missing = []
        mismatches = []
        for sample in section["samples"]:
            for directory, suffix, size_key in (
                ("images", "jpg", "image_bytes"),
                ("labels", "png", "label_bytes"),
            ):
                path = DATA / directory / split / f"{sample['sid']}.{suffix}"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                size = path.stat().st_size
                total_bytes += size
                if size != sample[size_key]:
                    mismatches.append(str(path) + ": byte size")
            image = DATA / "images" / split / f"{sample['sid']}.jpg"
            if image.is_file():
                with Image.open(image) as opened:
                    width, height = opened.size
                pixels += width * height
                max_pixels = max(max_pixels, width * height)
                if [height, width] != sample["hw"]:
                    mismatches.append(str(image) + ": shape")
        split_inventory[split] = {
            "images": len(section["samples"]),
            "actual_bytes": total_bytes,
            "pixels_from_actual_jpg_headers": pixels,
            "largest_image_pixels": max_pixels,
            "missing": missing,
            "metadata_mismatches": mismatches,
            "content_sha_verified": False,
        }
    weights = json.loads(WEIGHT_MANIFEST.read_text(encoding="utf-8"))["weights"]
    asset_paths = [Path(item["path"]) for item in weights.values()] + [S20]
    teacher_assets = [
        {"path": str(path), "exists": path.is_file(),
         "bytes": path.stat().st_size if path.is_file() else None,
         "content_sha_verified_this_audit": False}
        for path in asset_paths
    ]
    source_files = [path for path in CODE.rglob("*") if path.is_file()
                    and path.suffix in {".py", ".md", ".txt", ".json", ".yaml", ".yml"}]
    measurement = json.loads((HERE / "cpu_resource_measurement.json").read_text(encoding="utf-8"))
    disk = shutil.disk_usage("D:/")
    memory = psutil.virtual_memory()
    train_pixels = split_inventory["train"]["pixels_from_actual_jpg_headers"]
    payload = train_pixels * 3 * 4 * 4
    result = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Incremental local preparation before server access; reuse data, weights and Python; large cache and training stay on server.",
        "design_source": str(DESIGN / "05_v3_机制重构与反驳审查.md"),
        "identity_source": str(IDENTITY),
        "disk": {"drive": "D:", "total_bytes": disk.total, "free_bytes": disk.free},
        "ram": {"total_bytes": memory.total, "available_bytes": memory.available},
        "runtime": {"python": sys.executable,
                    "installed_modules": {key: importlib.util.find_spec(key) is not None
                                          for key in ["torch", "numpy", "PIL", "psutil", "mmcv", "mmseg"]},
                    "new_environment_required_for_current_small_probe": False},
        "data": {"root": str(DATA), "splits": split_inventory,
                 "all_splits_total_bytes": sum(item["actual_bytes"] for item in split_inventory.values()),
                 "incremental_copy_bytes": 0},
        "baseline_source": {"root": str(CODE), "source_file_count": len(source_files),
                            "source_bytes": sum(path.stat().st_size for path in source_files)},
        "teacher_assets": teacher_assets,
        "teacher_existing_total_bytes": sum(item["bytes"] or 0 for item in teacher_assets),
        "student_imagenet_initialization": {"path": str(W48), "bytes": W48.stat().st_size,
                                            "sha256": sha256(W48),
                                            "historical_B0_tensor_equality_verified": False},
        "teacher_cache": {
            "split": "train_454_only", "pixels": train_pixels, "members": 3,
            "channels": 4, "dtype": "float32", "bytes_per_pixel": 48,
            "uncompressed_array_payload_bytes": payload,
            "payload_GiB": payload / GIB,
            "largest_one_image_three_member_payload_bytes": split_inventory["train"]["largest_image_pixels"] * 48,
            "local_generation_or_download": False,
            "derived_T_dU_dS_extra_disk_bytes_if_computed_in_memory": 0,
            "extra_bytes_if_three_full_derived_maps_also_saved": payload,
            "compression_ratio_measured": False,
        },
        "small_cpu_probe": measurement,
        "planning_budgets_not_measured_usage": {
            "code_configs_reports_upload_bundle_disk_range_MiB": [20, 100],
            "local_incremental_disk_reserve_GiB": 2,
            "local_small_checks_RAM_reserve_GiB": [2, 4],
            "optional_all_selected_weights_staging_bytes": sum(item["bytes"] or 0 for item in teacher_assets) + W48.stat().st_size,
            "optional_fresh_environment_GiB": [3, 8],
            "optional_fresh_environment_note": "Rough allowance only; depends on CPU/CUDA packages and installation cache. Not required for current preparation.",
            "server_free_disk_target_GiB": [160, 200],
            "server_disk_assumptions": "One uncompressed A/U/S train cache, reused source data and weights, computed derived targets, bounded checkpoint retention; upload archive plus extracted cache not both retained.",
            "full_2048_training_RAM_or_VRAM": "unmeasured; requires server smoke test",
        },
        "preparation_status": {
            "inventory_and_small_private_module_check": "done",
            "full_HRNet_trainer_integration": "not_done",
            "AMP_DDP_gradient_merge": "not_verified",
            "geometric_pipeline_integration": "not_done",
            "large_teacher_cache": "not_generated",
        },
    }
    (HERE / "resource_estimate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "disk_free_GiB": disk.free / GIB,
        "RAM_total_GiB": memory.total / GIB,
        "RAM_available_GiB": memory.available / GIB,
        "dataset_existing_GiB": result["data"]["all_splits_total_bytes"] / GIB,
        "teacher_weights_existing_GiB": result["teacher_existing_total_bytes"] / GIB,
        "imagenet_existing_GiB": W48.stat().st_size / GIB,
        "CPU_peak_working_set_GiB": measurement["peak_working_set_bytes"] / GIB,
        "CPU_peak_commit_GiB": measurement["memory_bytes"]["peak_pagefile"] / GIB,
        "cache_uncompressed_GiB": payload / GIB,
        "max_image_three_teacher_payload_GiB": result["teacher_cache"]["largest_one_image_three_member_payload_bytes"] / GIB,
        "metadata_mismatches": sum(len(item["metadata_mismatches"]) for item in split_inventory.values()),
        "missing_files": sum(len(item["missing"]) for item in split_inventory.values()),
        "report": str(HERE / "resource_estimate.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
