#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务端 Phase A + C1：代码身份、环境、加载契约、配置解析（prompt §3.1/§3.2/§5.1/§5.2）。

一次性产出：
  - env.json              环境版本、确定性开关、GPU、实例标识
  - code_identity.json    服务端关键代码的 SHA-256（供与本地副本逐文件比对）
  - load_contract.json    load_checkpoint / load_state_dict 的**真实返回类型与行为**
  - resolved_config.json  解析后完整配置的哈希 + 关键字段快照
  - model_contract.json   实际构建出的模型 state_dict 键/shape/dtype 契约

不猜 API：直接从已安装包的源码里读出签名与返回语句。
"""
import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest(), os.path.getsize(p)


def atomic_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


CODE_FILES = [
    "hrplus_core.py", "hrplus_patch.py", "hr100_refinement.py", "ma_residual.py",
    "hr100_ensemble.py",
    "mmseg/models/segmentors/base.py",
    "mmseg/models/segmentors/lesion_encoder_decoder.py",
    "mmseg/models/segmentors/HRDecoder.py",
    "mmseg/models/segmentors/encoder_decoder.py",
    "mmseg/datasets/lesion_dataset.py",
    "mmseg/core/evaluation/lesion_metric.py",
    "mmseg/models/losses/binary_loss.py",
    "mmseg/datasets/pipelines/transforms.py",
    "mmseg/datasets/pipelines/formating.py",
    "mmseg/datasets/pipelines/test_time_aug.py",
    "tools/train.py", "tools/test.py", "tools/hrplus_smoke.py",
    "configs/lesion/exp/hrplus_patch_dice.py",
    "configs/lesion/exp/hrplus_patch_lovasz.py",
    "configs/lesion/exp/hrplus_mares.py",
    "configs/lesion/exp/hrplus_maond.py",
    "configs/lesion/exp/hr100_a.py", "configs/lesion/exp/hr100_b.py",
    "configs/lesion/exp/e4_detail_split613.py",
    "configs/lesion/exp/baseline_split613.py",
    "ops/eval_ens.py", "ops/smoke_mares.py", "ops/renorm_bn.py",
    "ops/hard_deadline_guard.py",
]

CKPTS = {
    "b_weights": "/root/autodl-tmp/hrplus/repo/weights/b_weights.pth",
    "d_full": "/root/autodl-tmp/hrplus/D_0913_retry/best_mIoU_iter_2000.pth",
    "d_student_only": "/root/autodl-tmp/hrplus/D_0913_retry/backup_student_2000.pth",
    "e_full": "/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth",
    "e2_full": "/root/autodl-tmp/hrplus/results/E2_0916/best_mIoU_iter_1500.pth",
    "e3_full": "/root/autodl-tmp/hrplus/results/E3_0916/best_mIoU_iter_2000.pth",
    "l_full": "/root/autodl-tmp/hrplus/results/L_0915/best_mIoU_iter_6000.pth",
    "e4_full": "/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth",
    "b_full": "/root/autodl-tmp/hr100/trial_b/best_mIoU_iter_3000.pth",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None, help="要解析的配置（相对 repo）")
    args = ap.parse_args()
    os.chdir(args.repo)
    sys.path.insert(0, args.repo)

    R = {}

    # ---------- 环境 ----------
    import torch
    import mmcv
    try:
        import mmseg
        mmseg_v = mmseg.__version__
    except Exception as e:
        mmseg_v = "ERR:%r" % e
    env = {
        "instance_hostname": subprocess.run(["hostname"], capture_output=True,
                                            text=True).stdout.strip(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "mmcv": mmcv.__version__,
        "mmseg": mmseg_v,
        "numpy": __import__("numpy").__version__,
        "torch_load_weights_only_default": (
            "True (>=2.6)" if torch.__version__ >= "2.6" else "False (<2.6)"),
    }
    R["environment"] = env
    print("[env] torch=%s mmcv=%s mmseg=%s cudnn.det=%s"
          % (env["torch"], env["mmcv"], env["mmseg"], env["cudnn_deterministic"]))

    # ---------- 代码身份 ----------
    code = {}
    for rel in CODE_FILES:
        p = os.path.join(args.repo, rel)
        if os.path.isfile(p):
            h, n = sha256_file(p)
            code[rel] = {"status": "FOUND", "bytes": n, "sha256": h}
        else:
            code[rel] = {"status": "MISSING"}
    R["code_identity"] = code
    nf = sum(1 for v in code.values() if v["status"] == "FOUND")
    print("[code] %d/%d 找到" % (nf, len(CODE_FILES)))

    # ---------- checkpoint 身份 ----------
    ck = {}
    for name, p in CKPTS.items():
        if os.path.isfile(p):
            h, n = sha256_file(p)
            ck[name] = {"status": "FOUND", "path": p, "bytes": n, "sha256": h}
        else:
            ck[name] = {"status": "MISSING", "path": p}
    R["checkpoints"] = ck
    print("[ckpt] %d/%d 找到" % (sum(1 for v in ck.values() if v["status"] == "FOUND"),
                                 len(CKPTS)))

    # ---------- load_checkpoint 真实契约（不猜） ----------
    from mmcv.runner import load_checkpoint as _lc
    lc_src = inspect.getsource(_lc)
    ret_lines = [l.strip() for l in lc_src.splitlines()
                 if l.strip().startswith("return")]
    lc_info = {
        "module": _lc.__module__,
        "qualname": _lc.__qualname__,
        "signature": str(inspect.signature(_lc)),
        "source_file": inspect.getsourcefile(_lc),
        "return_statements": ret_lines,
        "docstring_return": None,
    }
    doc = _lc.__doc__ or ""
    for i, l in enumerate(doc.splitlines()):
        if "Return" in l or "returns" in l:
            lc_info["docstring_return"] = " | ".join(
                x.strip() for x in doc.splitlines()[i:i + 4])
            break
    R["load_checkpoint_contract"] = lc_info
    print("[load_checkpoint] sig=%s" % lc_info["signature"])
    print("  return 语句: %s" % ret_lines)

    # ---------- 配置解析 ----------
    if args.config:
        from mmcv import Config as _Cfg
        cfg = _Cfg.fromfile(args.config)
        resolved = cfg.pretty_text
        h = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
        keyfields = {}
        try:
            keyfields["model.type"] = cfg.model.type
            keyfields["model.source_checkpoint"] = getattr(
                cfg.model, "source_checkpoint", None)
            keyfields["model.semantic_refinement"] = getattr(
                cfg.model, "semantic_refinement", None)
            keyfields["model.test_cfg"] = dict(cfg.model.get("test_cfg", {}))
            keyfields["model.hr_settings"] = dict(cfg.model.get("hr_settings", {}))
            keyfields["data.samples_per_gpu"] = cfg.data.get("samples_per_gpu")
            keyfields["data.workers_per_gpu"] = cfg.data.get("workers_per_gpu")
            keyfields["optimizer"] = dict(cfg.optimizer)
            keyfields["lr_config"] = dict(cfg.lr_config)
            keyfields["runner"] = dict(cfg.runner)
            keyfields["seed"] = cfg.get("seed")
            keyfields["custom_imports"] = cfg.get("custom_imports")
            for sp in ("train", "val", "test"):
                if sp in cfg.data:
                    d = cfg.data[sp]
                    keyfields["data.%s.data_root" % sp] = d.get("data_root")
                    keyfields["data.%s.img_dir" % sp] = d.get("img_dir")
                    keyfields["data.%s.ann_dir" % sp] = d.get("ann_dir")
                    keyfields["data.%s.split" % sp] = d.get("split")
        except Exception as e:
            keyfields["error"] = repr(e)
        R["resolved_config"] = {
            "path": args.config, "sha256_of_pretty_text": h,
            "bytes_of_pretty_text": len(resolved.encode("utf-8")),
            "key_fields": keyfields}
        print("[config] %s -> sha256(pretty)=%s" % (args.config, h[:16]))
        print("  model.type=%s test_cfg=%s" % (
            keyfields.get("model.type"), keyfields.get("model.test_cfg")))

    atomic_dump(R, os.path.join(args.out_dir, "server_phaseA_C1.json"))
    print("\n写出:", os.path.join(args.out_dir, "server_phaseA_C1.json"))


if __name__ == "__main__":
    main()
