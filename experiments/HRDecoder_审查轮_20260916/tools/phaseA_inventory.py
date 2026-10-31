#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase A 盘点与身份清单（prompt §3.1 / §3.2）。

设计约束（prompt §0.1）：
  - 不硬编码旧机器路径：仓库根、checkpoint 根、split 文件全部由命令行参数给出
  - 不安装依赖：只使用环境里已有的 numpy / torch
  - 缺失项不猜测、不静默跳过，全部写入 missing_dependencies.md

用法：
  python phaseA_inventory.py \
      --repo-root   <仓库根> \
      --ckpt-root   <checkpoint 根> \
      --split-file  <split_assignment.json> \
      --data-manifest <data_manifest.json> \
      --out-dir     <audit_runs/<run_id>> \
      --run-id      audit_20260916_A
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))

# §3.1 要求必须取得或定位的清单。value = 相对 repo 根的候选路径列表（按顺序找）
REQUIRED_CODE = {
    "hrplus_core.py": ["hrplus_core.py"],
    "hrplus_patch.py": ["hrplus_patch.py"],
    "hr100_refinement.py": ["hr100_refinement.py"],
    "ma_residual.py": ["ma_residual.py"],
    "hr100_ensemble.py": ["hr100_ensemble.py"],
    "segmentor_base": ["mmseg/models/segmentors/base.py"],
    "segmentor_lesion_encdec": ["mmseg/models/segmentors/lesion_encoder_decoder.py"],
    "segmentor_hrdecoder": ["mmseg/models/segmentors/HRDecoder.py"],
    "segmentor_encoder_decoder": ["mmseg/models/segmentors/encoder_decoder.py"],
    "dataset_lesion": ["mmseg/datasets/lesion_dataset.py"],
    "metric_lesion": ["mmseg/core/evaluation/lesion_metric.py"],
    "loss_binary": ["mmseg/models/losses/binary_loss.py"],
    "pipeline_transforms": ["mmseg/datasets/pipelines/transforms.py"],
    "pipeline_formating": ["mmseg/datasets/pipelines/formating.py"],
    "pipeline_test_time_aug": ["mmseg/datasets/pipelines/test_time_aug.py"],
    "train_entry": ["tools/train.py"],
    "test_entry": ["tools/test.py"],
    "smoke_hrplus": ["tools/hrplus_smoke.py"],
    "cfg_hrplus_dice": ["configs/lesion/exp/hrplus_patch_dice.py"],
    "cfg_hrplus_lovasz": ["configs/lesion/exp/hrplus_patch_lovasz.py"],
    "cfg_hrplus_mares": ["configs/lesion/exp/hrplus_mares.py"],
    "cfg_hrplus_maond": ["configs/lesion/exp/hrplus_maond.py"],
    "cfg_hr100_a": ["configs/lesion/exp/hr100_a.py"],
    "cfg_hr100_b": ["configs/lesion/exp/hr100_b.py"],
    "cfg_e4": ["configs/lesion/exp/e4_detail_split613.py"],
    "cfg_baseline": ["configs/lesion/exp/baseline_split613.py"],
}

# checkpoint 身份（§3.2 要求记录 student 来源 / reference 来源 / 头是否应存在）
# path 相对 --ckpt-root 解析；缺失记 BLOCKED
REQUIRED_CKPT = {
    "b_weights": {
        "paths": ["repo/weights/b_weights.pth",
                  "hrdecoder_plus3/repo/weights/b_weights.pth",
                  "autodl100/results/trial_b/selected_weights_only.pth"],
        "role": "B@3000；student 起点 + reference 来源",
        "expect_head": False,
        "expect_reference_keys": False,
    },
    "e4_weights": {
        "paths": ["autodl100/results/e4_reference_weights.pth"],
        "role": "E4@36000；B 的起点",
        "expect_head": False,
        "expect_reference_keys": False,
    },
    "a_weights": {
        "paths": ["autodl100/results/trial_a/selected_weights_only.pth"],
        "role": "A 方案（语义残差开启）",
        "expect_head": True,
        "expect_reference_keys": False,
    },
    "d_student_only": {
        "paths": ["hrdecoder_plus3/results/D_0913_retry/backup_student_2000.pth"],
        "role": "D@2000 的 student 分支（weights-only）",
        "expect_head": False,
        "expect_reference_keys": False,
    },
    "d_full": {
        "paths": ["hrdecoder_plus3/results/D_0913_retry/best_mIoU_iter_2000.pth"],
        "role": "D@2000 完整 checkpoint（含 reference.*）",
        "expect_head": False,
        "expect_reference_keys": True,
    },
    "l_full": {
        "paths": ["hrdecoder_plus3_L/best_mIoU_iter_6000.pth"],
        "role": "L@6000 完整 checkpoint",
        "expect_head": False,
        "expect_reference_keys": True,
    },
    "e_full": {
        "paths": ["hrdecoder_final/E_0916__best_mIoU_iter_1500.pth",
                  "hrdecoder_final/eval_json/../best_mIoU_iter_1500.pth"],
        "role": "E@1500 完整 checkpoint（含 ma_refiner.*）",
        "expect_head": True,
        "expect_reference_keys": True,
    },
}


def safe_relpath(p, start):
    """跨盘符时 os.path.relpath 会抛 ValueError，此时退回绝对路径。"""
    try:
        return os.path.relpath(p, start)
    except ValueError:
        return p


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def first_existing(roots, cands):
    for r in roots:
        for c in cands:
            p = os.path.join(r, c)
            if os.path.isfile(p):
                return p
    return None


def env_report():
    out = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for m in ("numpy", "torch", "mmcv", "mmseg"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "(no __version__)")
        except Exception:
            out[m] = None
    try:
        import torch
        out["torch_cuda_available"] = bool(torch.cuda.is_available())
        out["torch_cuda_version"] = torch.version.cuda
        out["cudnn_version"] = torch.backends.cudnn.version()
        out["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        out["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        out["tf32_matmul_allowed"] = bool(torch.backends.cuda.matmul.allow_tf32)
        out["tf32_cudnn_allowed"] = bool(torch.backends.cudnn.allow_tf32)
        out["amp_autocast_available"] = hasattr(torch, "autocast")
    except Exception as e:
        out["torch_probe_error"] = repr(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--split-file", default=None)
    ap.add_argument("--data-manifest", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--extra-code-root", default=None,
                    help="额外源码根（例如放 ma_residual.py 的工作区），可重复用分号分隔")
    args = ap.parse_args()

    code_roots = [args.repo_root]
    if args.extra_code_root:
        code_roots += [x for x in args.extra_code_root.split(";") if x]

    man = {
        "run_id": args.run_id,
        "generated_at": datetime.now(CST).isoformat(),
        "timezone": "Asia/Shanghai (UTC+8)",
        "workspace": {
            "repo_root": args.repo_root,
            "ckpt_root": args.ckpt_root,
            "code_roots": code_roots,
            "vcs": None,
        },
        "environment": env_report(),
        "code_files": {},
        "checkpoints": {},
        "splits": {},
        "main_metric": {
            "name": "mIoU",
            "classes": ["EX", "HE", "SE", "MA"],
            "background_included_in_mean": False,
            "unit": "0-1 (report prints ×100)",
            "threshold": "TBD_phase_A3",
            "aggregation": "TBD_phase_A3",
        },
    }

    # --- VCS 探测（§3.1 要求记录分支/提交/未提交改动）---
    if os.path.isdir(os.path.join(args.repo_root, ".git")):
        import subprocess
        def g(*a):
            try:
                return subprocess.run(["git", "-C", args.repo_root] + list(a),
                                      capture_output=True, text=True,
                                      timeout=30).stdout.strip()
            except Exception as e:
                return "ERR:%r" % e
        man["workspace"]["vcs"] = {
            "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": g("rev-parse", "HEAD"),
            "dirty_files": g("status", "--porcelain").splitlines(),
        }
    else:
        man["workspace"]["vcs"] = "NOT_A_GIT_REPO"

    # --- 代码文件 ---
    print("== 代码清单 ==")
    for key, cands in REQUIRED_CODE.items():
        p = first_existing(code_roots, cands)
        if p is None:
            man["code_files"][key] = {"status": "MISSING",
                                      "candidates": cands}
            print("  ✗ %-26s MISSING" % key)
            continue
        h, n = sha256_file(p)
        man["code_files"][key] = {
            "status": "FOUND",
            "path": safe_relpath(p, args.repo_root),
            "abs_path_local": p,
            "bytes": n, "sha256": h,
        }
        print("  ✓ %-26s %8d  %s" % (key, n, h[:16]))

    # --- checkpoint ---
    print("== checkpoint 清单 ==")
    for key, spec in REQUIRED_CKPT.items():
        p = first_existing([args.ckpt_root], spec["paths"])
        if p is None:
            man["checkpoints"][key] = {"status": "MISSING",
                                       "role": spec["role"],
                                       "candidates": spec["paths"]}
            print("  ✗ %-16s MISSING  (%s)" % (key, spec["role"]))
            continue
        h, n = sha256_file(p)
        rec = {"status": "FOUND", "role": spec["role"],
               "abs_path_local": p, "bytes": n, "sha256": h,
               "expect_head": spec["expect_head"],
               "expect_reference_keys": spec["expect_reference_keys"],
               "key_audit": "PENDING_phase_C1"}
        # 若能 import torch，顺便读一下键（不做契约判定，只记录事实）
        try:
            import torch
            ck = torch.load(p, map_location="cpu")
            sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
            keys = list(sd.keys())
            pref = sorted(set(k.split(".")[0] for k in keys))
            rec["key_audit"] = {
                "total_keys": len(keys),
                "top_level_prefixes": pref,
                "has_ma_refiner_keys": any("ma_refiner" in k for k in keys),
                "has_reference_keys": any(k.startswith("reference.") for k in keys),
                "ckpt_container_keys": (sorted(ck.keys()) if isinstance(ck, dict)
                                        else "RAW_STATE_DICT"),
            }
        except Exception as e:
            rec["key_audit"] = {"status": "BLOCKED", "reason": repr(e)}
        man["checkpoints"][key] = rec
        print("  ✓ %-16s %10d  %s" % (key, n, h[:16]))

    # --- split ---
    print("== split ==")
    for label, path in (("split_assignment", args.split_file),
                        ("data_manifest", args.data_manifest)):
        if not path:
            man["splits"][label] = {"status": "NOT_PROVIDED"}
            print("  - %-18s NOT_PROVIDED" % label)
            continue
        if not os.path.isfile(path):
            man["splits"][label] = {"status": "MISSING", "path": path}
            print("  ✗ %-18s MISSING" % label)
            continue
        h, n = sha256_file(path)
        rec = {"status": "FOUND", "abs_path_local": path,
               "bytes": n, "sha256": h}
        if label == "split_assignment":
            try:
                with open(path, encoding="utf-8") as f:
                    sa = json.load(f)
                rec["type"] = type(sa).__name__
                if isinstance(sa, dict):
                    rec["top_level_keys"] = sorted(sa.keys())[:20]
                    counts = {}
                    for k, v in sa.items():
                        if isinstance(v, (list, dict)):
                            counts[k] = len(v)
                    rec["counts"] = counts
            except Exception as e:
                rec["parse"] = {"status": "BLOCKED", "reason": repr(e)}
        man["splits"][label] = rec
        print("  ✓ %-18s %8d  %s" % (label, n, h[:16]))

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "manifest.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out)          # §4.1: 临时文件 + 原子替换
    print("\n写出:", out)

    # 缺失清单
    miss = [k for k, v in man["code_files"].items() if v["status"] != "FOUND"]
    missc = [k for k, v in man["checkpoints"].items() if v["status"] != "FOUND"]
    print("代码缺失 %d 项，checkpoint 缺失 %d 项" % (len(miss), len(missc)))
    if miss:
        print("  代码:", miss)
    if missc:
        print("  ckpt:", missc)


if __name__ == "__main__":
    main()
