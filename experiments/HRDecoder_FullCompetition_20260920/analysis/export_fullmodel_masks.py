#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""export_fullmodel_masks — 导出本工程 U0/H0 整模的 val 二值掩膜（同一 packbits 契约）。

与 `export_hrplus_expert_masks.py` 的关键差别：这里**不复用** FusionAware 的
mmseg 评测栈，而是复用本工程自己的 canonical 评测路径：

- 模型构建：`trainer.models.build_model(cfg, stage=...)`（U0=builtin ConvNeXt-S+U-Net，
  H0=HRNet-W48 adapter）；
- 权重：`trainer.checkpoint.load_checkpoint`（`weights_only=True` 安全加载）
  + `load_state_dict(strict=True)` + 逐张量 bitwise 回验；
- 数据：`trainer.data.ddr_dataset.DDRSegDataset(split='val')`（只读 fused PNG）；
- 整图/滑窗：直接复用 `trainer.eval.pad_normalized_image` /
  `trainer.eval.sliding_window_logits` / `trainer.data.common.crop_logits_to_orig`
  / `normalize_image`，即 `WholeImageEvaluator.run()` 的同一批函数（不复制逻辑）；
- 指标交叉验证：同一批概率同时喂给 `trainer.metrics.SegMetricsAccumulator`
  （canonical mIoU），并要求它与"packbits 重算的 TP/FP/FN"逐类完全一致。

概率全程 float32（sigmoid 前显式 .float()，且在 .float() 之前拦住 fp16），
严格 p>0.5 二值化后只落 1 bit/像素。

只处理 val（76 张）。split=test、路径含 test 分量、图像数 != 76 全部拒绝。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_PROJECT_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import expert_mask_contract as C  # noqa: E402

PIPELINE_FAMILY = "fullmodel"
MODELS = ("U0", "H0")
REQUIRED_KEYS = ("ddr_root", "export",
                 "fullmodel")


def load_raw_settings(path: str) -> dict:
    p = C.safe_path(path, "--settings")
    if not os.path.isfile(p):
        raise C.PreflightBlocked("settings 文件不存在: %s" % p)
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise C.ContractError("settings 缺少必填键: %s" % missing)
    if not isinstance(raw["fullmodel"], dict) or not raw["fullmodel"].get("models"):
        raise C.ContractError("settings.fullmodel.models 必须是非空列表")
    return raw


def resolve_model_specs(raw: dict, selected) -> list:
    specs, seen = [], set()
    for spec in raw["fullmodel"]["models"]:
        name = str(spec.get("name", ""))
        if name not in MODELS:
            raise C.ContractError("fullmodel.models[].name 必须是 %s，得到 %r"
                                  % (list(MODELS), name))
        if name in seen:
            raise C.ContractError("fullmodel.models 名称重复: %r" % name)
        seen.add(name)
        if name not in selected:
            continue
        for key in ("config", "checkpoint"):
            if not spec.get(key):
                raise C.ContractError("fullmodel.models[%s] 缺少 %s" % (name, key))
        cfg_path = spec["config"]
        if not os.path.isabs(cfg_path):
            base = spec.get("config_base") or _PROJECT_ROOT
            cfg_path = os.path.join(base, cfg_path)
        cfg_path = C.safe_path(cfg_path, "models[%s].config" % name)
        ckpt = C.safe_path(spec["checkpoint"], "models[%s].checkpoint" % name)
        if not os.path.isfile(cfg_path):
            raise C.PreflightBlocked("YAML 配置缺失: %s" % cfg_path)
        if not os.path.isfile(ckpt):
            raise C.PreflightBlocked(
                "checkpoint 缺失: %s（H0 权重若无本地/服务器副本，见 README 的阻塞清单）"
                % ckpt)
        sha = C.sha256_file(ckpt)
        want = spec.get("sha256")
        if want and want != sha:
            raise C.ContractError("models[%s]: checkpoint sha256 不符（声明 %s… 实际 %s…）"
                                  % (name, str(want)[:12], sha[:12]))
        specs.append({"name": name, "config": cfg_path,
                      "config_sha256": C.sha256_file(cfg_path),
                      "checkpoint": ckpt, "sha256": sha,
                      "size": os.path.getsize(ckpt),
                      "reference_root": spec.get("reference_root"),
                      "arm_id_expected": name,
                      "expected_miou": spec.get("expected_miou")})
    missing = set(selected) - {s["name"] for s in specs}
    if missing:
        raise C.ContractError("settings.fullmodel.models 未声明: %s" % sorted(missing))
    return specs


def apply_settings_overrides(cfg: dict, spec: dict, raw: dict, ddr_root: str) -> list:
    """把 settings 的真实部署路径写进 YAML 的内存副本（绝不改磁盘文件）。

    必须在 `validate_config` / `ddr_data_config` / `build_model` **之前**调用：
    训练前 YAML 保留的占位符（`data.ddr_root` 的旧 Windows 路径、H0 的
    `structure.reference_root=REQUIRED`）在真实服务器上会直接阻塞。这里只覆盖
    settings 显式提供的 ddr_root 与 reference_root；arm_id、数据契约、结构仍由
    原 YAML 与后续闸门逐项把关，不做任何放宽。
    """
    overrides = []
    data = cfg.get("data")
    cur_root = data.get("ddr_root") if isinstance(data, dict) else None
    if isinstance(cur_root, str) and cur_root.strip():
        if C.safe_path(cur_root, "cfg.ddr_root") != C.safe_path(ddr_root, "ddr_root"):
            data["ddr_root"] = ddr_root
            overrides.append("data.ddr_root=%s" % ddr_root)
    st = cfg.get("structure")
    if spec.get("reference_root"):
        if not isinstance(st, dict) or str(st.get("family")) != "hrdecoder":
            raise C.ContractError("models[%s]: reference_root given but family != hrdecoder"
                                  % spec["name"])
        ref = C.safe_path(spec["reference_root"],
                          "models[%s].reference_root" % spec["name"])
        st["reference_root"] = ref
        overrides.append("structure.reference_root=%s" % ref)
    return overrides


def disable_pretrained_init(cfg: dict) -> list:
    """显式关闭一切会触发预训练加载的初始化字段（只改内存副本）。

    本脚本的权重**只**来自 checkpoint（`load_checkpoint` + `strict=True` 全量加载
    + 逐张量 bitwise 回验），模型构建期的 ImageNet/编码器初始化既多余，又可能把
    旧 Windows 路径或错误文件加载进模型。两个 family 的加载触发点——U 臂的
    `structure.encoder_weights_file`、H 臂的 `initialization.imagenet_weights.file`
    ——一并置空，使 `build_model` 的 weight_report 恒为 None；
    `settings.fullmodel.imagenet_weights` 因此被忽略，绝不注入 YAML。
    """
    overrides = []
    st = cfg.get("structure")
    if isinstance(st, dict) and st.get("encoder_weights_file") not in (None, ""):
        st["encoder_weights_file"] = None
        st["encoder_weights_sha256"] = None
        overrides.append(
            "structure.encoder_weights_file=<disabled; checkpoint strict load>")
    init = cfg.get("initialization")
    if isinstance(init, dict):
        iw = init.get("imagenet_weights")
        if isinstance(iw, dict) and iw.get("file") not in (None, ""):
            iw["file"] = None
            iw["sha256"] = None
            overrides.append(
                "initialization.imagenet_weights.file=<disabled; checkpoint strict load>")
    return overrides


def build_one(spec, *, ddr_root: str, raw: dict, device: str, torch):
    """按本工程 canonical 路径构建并严格加载一个整模；返回 (model, meta, ctx)。"""
    from trainer.config import (assert_no_test_split, checkpoint_selection_splits,
                                ddr_data_config, load_yaml, metrics_config,
                                require_reference_root, validate_config)
    from trainer.data.ddr_dataset import DDRSegDataset
    from trainer.models import build_model
    from trainer.checkpoint import load_checkpoint

    cfg = load_yaml(spec["config"])
    overrides = apply_settings_overrides(cfg, spec, raw, ddr_root)
    overrides += disable_pretrained_init(cfg)
    validate_config(cfg, spec["config"])
    if cfg.get("arm_id") != spec["name"]:
        raise C.ContractError("配置 arm_id=%r != settings 声明 %r（拒绝：可能拿错权重）"
                              % (cfg.get("arm_id"), spec["name"]))
    iw = ((raw.get("fullmodel") or {}).get("imagenet_weights") or {})
    if iw.get("file"):
        overrides.append("settings.fullmodel.imagenet_weights 已忽略"
                         "（strict checkpoint 全量覆盖，不加载外部初始化）")

    data_cfg = ddr_data_config(cfg)            # 校验 mean/std/channel_order/非 test
    assert_no_test_split(cfg, split=data_cfg["val_split"], role="val split")
    checkpoint_selection_splits(cfg)
    mcfg = metrics_config(cfg)

    eval_mode = str(((raw.get("fullmodel") or {}).get("eval_mode"))
                    or mcfg["eval_mode"])
    eval_src = "settings_override" if (raw.get("fullmodel") or {}).get("eval_mode") else "config_yaml"
    window = (raw.get("fullmodel") or {}).get("window_size", mcfg["window_size"])
    stride = (raw.get("fullmodel") or {}).get("window_stride", mcfg["window_stride"])
    if eval_mode not in ("whole_image", "sliding_window"):
        raise C.ContractError("eval_mode=%r 非法" % eval_mode)
    if eval_mode == "sliding_window":
        window = int(window) if window else None
        stride = int(stride) if stride else None
        if not window or not stride:
            raise C.ContractError("sliding_window 必须显式给出 window_size/window_stride（不猜）")

    stage = "ddr_finetune"
    model, meta = build_model(cfg, stage=stage, gradient_checkpointing=None)
    if meta.get("weight_report") is not None:
        raise C.ContractError(
            "models[%s]: build_model 意外加载了预训练权重（weight_report 非空）——"
            "本脚本要求模型权重完全来自 checkpoint，拒绝加载任何外部初始化"
            % spec["name"])
    ck = load_checkpoint(spec["checkpoint"])   # weights_only=True 安全加载 + 格式校验
    if ck.get("arm_id") != spec["arm_id_expected"]:
        raise C.ContractError("checkpoint arm_id=%r != %r"
                              % (ck.get("arm_id"), spec["arm_id_expected"]))
    model.load_state_dict(ck["model_state"], strict=True)
    loaded = model.state_dict()
    for k, v in ck["model_state"].items():
        if k not in loaded or not torch.equal(loaded[k].detach().cpu(),
                                             v.detach().cpu()):
            raise C.ContractError("checkpoint bitwise 回验失败: %s" % k)
    del ck, loaded
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ds = DDRSegDataset(ddr_root, data_cfg["val_split"],
                      mean=data_cfg["mean"], std=data_cfg["std"], augment=None)
    C.assert_val_only(C.SPLIT, n_images=len(ds), what="DDRSegDataset val")
    ctx = {
        "eval_mode": eval_mode, "eval_mode_source": eval_src,
        "window_size": window, "window_stride": stride,
        "mean": data_cfg["mean"], "std": data_cfg["std"],
        "stage": stage, "val_split": data_cfg["val_split"],
        "cfg_overrides": overrides,
        "threshold": mcfg["threshold"],
        "config_text_sha256": C.sha256_file(spec["config"]),
        "config_arm_id": cfg.get("arm_id"),
        "checkpoint_format": "hrdecoder-training/v1",
        "checkpoint_arm_id": spec["arm_id_expected"],
        "family": meta["family"],
        "imagenet_init_applied": bool((meta.get("weight_report") is not None)),
        "reference_root": meta.get("reference_root"),
    }
    return model, ds, ctx


def main():
    ap = argparse.ArgumentParser(
        description="导出 U0/H0 整模 val 二值掩膜（复用本工程整图/滑窗评测逻辑）")
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke：只跑前 N 张且不写 contract")
    ap.add_argument("--device", default=None, help="cuda（默认）/ cpu")
    args = ap.parse_args()
    log = C.log

    raw = load_raw_settings(args.settings)
    settings_path = C.safe_path(args.settings, "--settings")
    specs = resolve_model_specs(raw, args.models)
    ddr_root = C.safe_path(raw["ddr_root"], "ddr_root")
    out_root = C.safe_path(args.out_root or raw["export"]["out_root"], "out_root")
    C.assert_val_only(C.SPLIT, paths=(ddr_root, out_root), what="export_fullmodel")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    from trainer.data.common import crop_logits_to_orig, normalize_image
    from trainer.eval import pad_normalized_image, sliding_window_logits
    from trainer.metrics import SegMetricsAccumulator
    import cv2

    out_dir = C.ensure_dir(os.path.join(out_root, "fullmodel"))
    gt_dir = os.path.join(out_root, "_shared", "gt")
    all_per_image = {}

    for spec in specs:
        log("=== %s 构建/加载 ===" % spec["name"])
        # 每个模型必须写进**自己的**子目录：历史上掩膜曾写到 out_dir 根目录，
        # 于是 U0 的 76 张会被 H0 覆盖（与 HRPlus 六管线的覆盖缺陷同类）。
        model_dir = C.ensure_dir(os.path.join(out_dir, spec["name"]))
        if os.path.abspath(model_dir) == os.path.abspath(out_dir):
            raise C.ContractError("模型目录不得等于 fullmodel 根目录（会互相覆盖）")
        model, ds, ctx = build_one(spec, ddr_root=ddr_root, raw=raw,
                                   device=device, torch=torch)
        stems = sorted(ds.stems)
        image_hw = {}
        for i in range(len(ds)):
            img_u8, _t, _v, stem = ds.load_raw(i)
            image_hw[stem] = tuple(int(x) for x in img_u8.shape[:2])
        gt_man = C.write_or_verify_gt(
            gt_dir, data_root=ddr_root, split=C.SPLIT, stems=stems,
            label_reader=lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED),
            image_hw=image_hw)

        per_image, counts = {}, [[0, 0, 0] for _ in range(C.NUM_CLASSES)]
        acc = SegMetricsAccumulator(threshold=C.STRICT_THRESHOLD)
        prev_acc = [[0, 0, 0] for _ in range(C.NUM_CLASSES)]   # acc 的上一次累计快照
        checks = {"mask_bits_match_canonical_metrics": True,
                  "dataset_label_matches_shared_gt": True,
                  "images_processed": 0,
                  "nonfinite_or_out_of_range": 0}
        limit = len(ds) if args.limit is None else min(int(args.limit), len(ds))
        t0 = time.time()
        for i in range(limit):
            img_u8, target, valid, stem = ds.load_raw(i)
            h, w = int(img_u8.shape[0]), int(img_u8.shape[1])
            if tuple(target.shape[1:]) != (h, w) or tuple(valid.shape[1:]) != (h, w):
                raise C.ContractError(
                    "%s[%s]: label/valid 空间尺寸 %r/%r != 图像尺寸 (%d,%d)；"
                    "本工程整图口径要求二者一致" % (spec["name"], stem,
                                             tuple(target.shape[1:]),
                                             tuple(valid.shape[1:]), h, w))
            gt_np = C.load_gt_npz(os.path.join(gt_dir, C.MASK_FILENAME % stem),
                                  where="gt[%s]" % stem)
            gt_bits = gt_np["bits"].reshape(C.NUM_CLASSES, -1)
            ds_bits = (target > 0.5).reshape(C.NUM_CLASSES, -1)
            if not np.array_equal(ds_bits, gt_bits):
                checks["dataset_label_matches_shared_gt"] = False
                raise C.ContractError(
                    "%s[%s]: DDRSegDataset target 与共享 GT 不一致" % (spec["name"], stem))
            ig_bits = (gt_np["ignore"] | (valid[0] <= 0.5)).reshape(-1)

            chw = normalize_image(img_u8, ctx["mean"], ctx["std"]).transpose(2, 0, 1)
            padded, (oh, ow) = pad_normalized_image(np.ascontiguousarray(chw))
            x = torch.from_numpy(padded[None]).to(device).float()
            with torch.no_grad():
                if ctx["eval_mode"] == "whole_image":
                    logits = model.eval_forward(x)
                else:
                    logits = sliding_window_logits(
                        model, x, ctx["window_size"], ctx["window_stride"])
                C.check_float32_like(logits, "%s[%s].logits" % (spec["name"], stem))
                logits = crop_logits_to_orig(logits, (oh, ow))   # 还原原始尺寸
                probs_b = torch.sigmoid(logits.float()).float().cpu().numpy()
            # canonical 口径（trainer/eval.py:129-134）：eval_forward 返回
            # (1,4,oh,ow)，`SegMetricsAccumulator.update` 需要
            # probs=(1,4,oh,ow) 与 target[None]/valid[None]=(1,4,oh,ow) 配对。
            # 注意 update 只会给 **3-D** 的 probs 补 batch 维，却总是再包一层
            # target/valid，所以"把 probs 压成 3-D"会让两侧错位 —— 历史实现
            # 既没补 batch 也没压 probs，两种写法都会在首个样本上 exit 2。
            # 这里统一成 probs_b=(1,4,oh,ow)（喂指标），probs=(4,oh,ow)（喂契约）。
            if probs_b.ndim == 3:
                probs_b = probs_b[None]
            if (probs_b.ndim != 4 or probs_b.shape[0] != 1
                    or probs_b.shape[1] != C.NUM_CLASSES
                    or tuple(probs_b.shape[2:]) != (oh, ow)):
                raise C.ContractError(
                    "%s[%s]: 概率形状 %r != (1,%d,%d,%d)"
                    % (spec["name"], stem, probs_b.shape, C.NUM_CLASSES, oh, ow))
            probs = np.ascontiguousarray(probs_b[0])
            if not np.isfinite(probs).all() or probs.min() < 0.0 or probs.max() > 1.0:
                checks["nonfinite_or_out_of_range"] += 1
                raise C.ContractError("%s[%s]: 概率非有限或越界" % (spec["name"], stem))

            bits = C.binarize_strict(probs, "%s[%s]" % (spec["name"], stem))
            rec = C.save_mask_npz(
                os.path.join(model_dir, C.MASK_FILENAME % stem), bits, (oh, ow))
            cnt_bits = C.counts_from_bits(bits.reshape(C.NUM_CLASSES, -1), gt_bits, ig_bits)
            acc.update(probs_b, target[None], valid[None])
            # SegMetricsAccumulator.tp/fp/fn 是**跨样本累计**值；逐图证明必须
            # 用"本图增量"（cnt_acc - prev_acc）与单图 bits 计数比，而不是拿
            # 累计值直接比（历史实现把累计当单图，第 2 张图就必然 fail closed）。
            cnt_acc = [[int(acc.tp[c]), int(acc.fp[c]), int(acc.fn[c])]
                       for c in range(C.NUM_CLASSES)]
            delta_acc = [[cnt_acc[c][j] - prev_acc[c][j] for j in range(3)]
                         for c in range(C.NUM_CLASSES)]
            if cnt_bits != delta_acc:
                checks["mask_bits_match_canonical_metrics"] = False
                raise C.ContractError(
                    "%s[%s]: packbits 单图计数 %r != SegMetricsAccumulator 本图增量 %r"
                    % (spec["name"], stem, cnt_bits, delta_acc))
            prev_acc = cnt_acc
            rec["counts"] = {C.CLASSES[c]: list(cnt_bits[c]) for c in range(C.NUM_CLASSES)}
            per_image[stem] = rec
            for c in range(C.NUM_CLASSES):
                for j in range(3):
                    counts[c][j] += int(cnt_bits[c][j])
            checks["images_processed"] += 1
            if (i + 1) % 10 == 0 or (i + 1) == limit:
                log("%s export %d/%d (%.1fs)" % (spec["name"], i + 1, limit,
                                                 time.time() - t0))
            del x, logits, probs, probs_b, bits

        report = acc.compute()
        for c, cname in enumerate(C.CLASSES):
            ent = report["per_class"][cname]
            got = [int(ent["tp"]), int(ent["fp"]), int(ent["fn"])]
            want = [int(x) for x in counts[c]]
            if got != want:
                raise C.ContractError(
                    "%s: canonical 累计计数 %s=%r != 契约重算 %r"
                    % (spec["name"], cname, got, want))
        # union==0 的 fail-closed 策略是**整轮 val** 的口径（四类在 76 张里都有正样本）。
        # smoke（--limit < 76）只看前 N 张，某类可能恰好无正样本也无预测，
        # 此时按整轮策略会直接 exit 2，掩盖真正的 smoke 目的（打通链路）。
        # 因此只在非整轮时放行 union==0；其余（计数一致性/canonical 对齐）全部照跑，
        # 并且不写任何 contract（见下方 limit 分支）。
        if limit != C.EXPECTED_VAL_IMAGES:
            log("SMOKE: %s 只跑了 %d/%d 张 —— 允许 union==0 的类（不写 contract）"
                % (spec["name"], limit, C.EXPECTED_VAL_IMAGES))
        miou, per_class, detail = C.iou_from_counts(
            counts, fail_on_zero_union=(limit == C.EXPECTED_VAL_IMAGES))
        if (report["mIoU_lesion4"] is not None
                and not report["excluded_classes"]["iou"]
                and abs(report["mIoU_lesion4"] - miou) > 1e-12):
            raise C.ContractError(
                "%s: canonical mIoU %.12f != 契约重算 %.12f"
                % (spec["name"], report["mIoU_lesion4"], miou))
        if spec.get("expected_miou") is not None and limit == C.EXPECTED_VAL_IMAGES:
            # 只在**整轮 val（76 张）**口径下比对历史期望：smoke（--limit < 76）
            # 的 mIoU 只覆盖前 N 张，与历史值不可比。历史实现无条件比对，
            # 导致任何 smoke 都必然 fail closed（exit 2）。
            d = abs(float(spec["expected_miou"]) - miou)
            log("%s 历史期望 val mIoU %.12f，本次 %.12f，Δ=%.3e"
                % (spec["name"], float(spec["expected_miou"]), miou, d))
            if d > 1e-6:
                raise C.ContractError(
                    "%s: val mIoU 与历史期望差 %.3e > 1e-6 —— 环境/权重/管线不一致，"
                    "拒绝出图（如需接受请显式从 settings 删除 expected_miou）"
                    % (spec["name"], d))
        elif spec.get("expected_miou") is not None:
            log("SMOKE: %s 只跑了 %d/%d 张 —— 跳过 expected_miou 比对"
                "（历史期望 %.12f 属于整轮 val 口径）"
                % (spec["name"], limit, C.EXPECTED_VAL_IMAGES,
                   float(spec["expected_miou"])))

        all_per_image[spec["name"]] = per_image
        if limit != C.EXPECTED_VAL_IMAGES:
            log("SMOKE: %s 只处理 %d/%d 张 —— 不写 contract"
                % (spec["name"], limit, C.EXPECTED_VAL_IMAGES))
            continue

        contract = {
            "model_name": spec["name"],
            "pipeline_family": PIPELINE_FAMILY,
            "split": C.SPLIT,
            "n_images": len(stems),
            "stems": stems,
            "eval_contract": C.build_eval_contract(
                pipeline_family=PIPELINE_FAMILY,
                eval_loader="trainer.data.ddr_dataset.DDRSegDataset + trainer.eval",
                inference=("whole_image" if ctx["eval_mode"] == "whole_image"
                           else "sliding_window(window=%s,stride=%s)"
                                % (ctx["window_size"], ctx["window_stride"])),
                tta="none"),
            "pipeline_definition": {
                "name": spec["name"],
                "family": ctx["family"],
                "stage": ctx["stage"],
                "eval_mode": ctx["eval_mode"],
                "eval_mode_source": ctx["eval_mode_source"],
                "window_size": ctx["window_size"],
                "window_stride": ctx["window_stride"],
                "detail_branch": False,
                "tta": "none",
                # 键名不得用 `probabilities`/`probability`：它们是
                # expert_mask_contract.FORBIDDEN_CONTRACT_KEYS 的保留键
                # （"禁止保存概率"的硬黑名单），用文字说明也会被 fail closed。
                "probability_source": "sigmoid(logits.float()) 后严格 >0.5",
            },
            "identity": {
                "checkpoints": [{"role": spec["name"], "path": spec["checkpoint"],
                                 "sha256": spec["sha256"], "size": spec["size"],
                                 "format": ctx["checkpoint_format"],
                                 "arm_id": ctx["checkpoint_arm_id"]}],
                "configs": [{"path": spec["config"], "sha256": spec["config_sha256"],
                             "config_text_sha256": ctx["config_text_sha256"],
                             "arm_id": ctx["config_arm_id"]}],
                "code_sha256": {
                    "export_fullmodel_masks.py": C.sha256_file(os.path.abspath(__file__)),
                    "expert_mask_contract.py": C.sha256_file(C.self_script_path()),
                    "trainer/eval.py": C.sha256_file(os.path.join(_PROJECT_ROOT, "trainer", "eval.py")),
                    "trainer/metrics.py": C.sha256_file(os.path.join(_PROJECT_ROOT, "trainer", "metrics.py")),
                    "trainer/checkpoint.py": C.sha256_file(os.path.join(_PROJECT_ROOT, "trainer", "checkpoint.py")),
                    "trainer/data/ddr_dataset.py": C.sha256_file(
                        os.path.join(_PROJECT_ROOT, "trainer", "data", "ddr_dataset.py")),
                },
                "settings_file_sha256": C.sha256_file(settings_path),
                "data_root": ddr_root,
                "reference_root": ctx["reference_root"],
                "imagenet_init_applied_before_checkpoint": ctx["imagenet_init_applied"],
                "config_overrides_applied": ctx["cfg_overrides"],
            },
            "gt": {
                "gt_contract_sha256": gt_man["gt_contract_sha256"],
                "gt_semantic_fingerprint": gt_man["gt_semantic_fingerprint"],
                "data_root": gt_man["data_root"],
                "split": C.SPLIT,
            },
            "per_image": per_image,
            "global_counts": {C.CLASSES[c]: list(counts[c]) for c in range(C.NUM_CLASSES)},
            "metrics": {"mIoU_lesion4": miou, "per_class_iou": per_class,
                        "per_class_detail": detail,
                        "canonical_report": {
                            k: report[k] for k in
                            ("protocol", "threshold", "n_images", "mIoU_lesion4",
                             "mF1", "mPPV", "mRecall", "mAUPR", "excluded_classes")
                        }},
            "self_checks": dict(checks, counts_from_masks_verified=True),
            "blockers": list(raw.get("known_blockers", [])),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        C.finalize_model_dir(model_dir, contract)
        log("%s mIoU=%.12f per_class=%s" % (spec["name"], miou,
                                            {k: (None if v is None else round(v, 6))
                                             for k, v in per_class.items()}))
        if ctx["eval_mode"] == "sliding_window":
            log("%s: 注意本次为滑窗口径（window=%s stride=%s，来源=%s）——"
                "与整图结果不可混用" % (spec["name"], ctx["window_size"],
                                  ctx["window_stride"], ctx["eval_mode_source"]))
        del model, ds, acc
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    if args.limit is not None and int(args.limit) < C.EXPECTED_VAL_IMAGES:
        print(json.dumps({"smoke": True, "images": int(args.limit),
                          "models": args.models}, ensure_ascii=False, indent=2))
        return 0

    gt_man = C.load_gt_contract(gt_dir)["contract"]
    C.atomic_write_json(os.path.join(out_root, "RUN_MANIFEST_fullmodel.json"), {
        "schema": C.RUN_MANIFEST_SCHEMA,
        "script": "export_fullmodel_masks.py",
        "script_sha256": C.sha256_file(os.path.abspath(__file__)),
        "split": C.SPLIT,
        "n_images": int(gt_man["n_images"]),
        "models": list(args.models),
        "model_dir": out_dir,
        "gt_contract_sha256": gt_man["gt_contract_sha256"],
        "gt_semantic_fingerprint": gt_man["gt_semantic_fingerprint"],
        "settings_sha256": C.sha256_file(settings_path),
        "complete": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    log("DONE fullmodel: %s" % out_dir)
    return 0


def main_entry():
    try:
        return main()
    except C.PreflightBlocked as e:
        C.log("BLOCKED: %s" % e)
        return 3
    except C.ContractError as e:
        C.log("CONTRACT FAILURE: %s" % e)
        return 2
    except BaseException:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main_entry())
