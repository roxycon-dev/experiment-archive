#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""export_hrplus_expert_masks — 导出 HRPlus 家族 6 条**完整预测管线**的 val 二值掩膜。

管线（全部冻结，无任何可扫参的权重）：
  1. champion    = BASE_0 + H4 后融合 MA 校正头（D×3+E×2+B×1+E4×1 概率加权 + hflip TTA）
  2. BASE_0      = 同一 4 成员概率加权 + hflip TTA，**不接** H4 头
  3. D_blend     = 0.5 · P_student(D) + 0.5 · P_reference(D)（B 冻结基座）
  4. C2000       = C 臂 node checkpoint（class-balanced 采样续训）独立 student 概率
  5. MIX_C2000   = 0.75 · champion + 0.25 · C2000（历史冻结常数）
  6. S4000       = S 臂 node checkpoint（class→image→instance 条件采样）独立 student 概率

实现原则（与历史评测脚本同源）：
- champion 一律走既有 `runtime.ChampionEnsemble`：成员加载、hflip TTA、
  7 通道 H4 头、`corrected_class_prob` 的 f7 构造全部复用，不重写、不改动；
- C/S 一律走既有 `run_protocol.build_student` + `eval_candidates.load_node_checkpoint`
  （strict load + 逐张量 bitwise 回验），候选概率只取 `_student_prob`，
  **绝不**用 `model.inference()`（那是 0.5 reference + 0.5 student 的混合）；
- 每张图**只做一次成员前向**：`pre_head_probs()` 同时给出 pf(BASE_0) / ps / pr，
  H4 头由本脚本在共享张量上复用 `corrected_class_prob` 施加（并在首图上与
  `champion.predict()` 做逐位等价自检）；
- 概率全程 float32：sigmoid / 融合 / H4 全部在 float32 完成，然后**严格 p>0.5**；
- 落盘只有 1 bit/像素 packbits + 计数 + 哈希，**不保存任何概率**；
- 逐图掩膜的位内容与 `runtime.per_image_counts` 的计数必须逐类完全一致，
  否则 fail closed（这是"位图 → 计数"链路的机械证明）。

只处理 val（76 张）。split=test、路径含 test 分量、图像数 != 76 全部拒绝。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import expert_mask_contract as C  # noqa: E402

PIPELINES = ("champion", "BASE_0", "D_blend", "C2000", "MIX_C2000", "S4000")
MIX_CHAMPION = 0.75          # 冻结常数：历史定义在 eval_candidates.py 中同样冻结
MIX_CANDIDATE = 0.25
assert abs(MIX_CHAMPION + MIX_CANDIDATE - 1.0) < 1e-12
BLEND_W = 0.5                # D_blend = 0.5*student + 0.5*reference（历史 diag 同式）
PIPELINE_FAMILY = "hrplus"
EXPECTED_CANDIDATE_STEPS = {"C2000": ("C", 2000), "S4000": ("S", 4000)}
REQUIRED_KEYS = ("repo", "data_root", "config", "champion", "expected_sha256",
                 "fusionaware_src_dir", "fusionaware_exec_dir", "cache_dir")


# ---------------------------------------------------------------------------
# settings / import plumbing
# ---------------------------------------------------------------------------

def load_raw_settings(path: str) -> dict:
    p = C.safe_path(path, "--settings")
    if not os.path.isfile(p):
        raise C.PreflightBlocked("settings 文件不存在: %s" % p)
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise C.ContractError("settings 缺少必填键: %s" % missing)
    return raw


def setup_imports(raw: dict):
    """把 FusionAware 源码目录 / eval_candidates 目录 / mmseg repo 加进 sys.path。"""
    src = C.safe_path(raw["fusionaware_src_dir"], "fusionaware_src_dir")
    exd = C.safe_path(raw["fusionaware_exec_dir"], "fusionaware_exec_dir")
    repo = C.safe_path(raw["repo"], "repo")
    for d, what in ((src, "runtime.py/run_protocol.py/fusion_core.py"),
                    (exd, "eval_candidates.py"), (repo, "mmseg repo")):
        if not os.path.isdir(d):
            raise C.PreflightBlocked("%s 目录不存在: %s（需要 %s）" % (what, d, what))
        if d not in sys.path:
            sys.path.insert(0, d)
    for name in ("runtime.py", "run_protocol.py", "fusion_core.py"):
        if not os.path.isfile(os.path.join(src, name)):
            raise C.PreflightBlocked("缺少 %s（fusionaware_src_dir=%s）" % (name, src))
    if not os.path.isfile(os.path.join(exd, "eval_candidates.py")):
        raise C.PreflightBlocked("缺少 eval_candidates.py（fusionaware_exec_dir=%s）" % exd)
    rt = importlib.import_module("runtime")
    rp = importlib.import_module("run_protocol")
    ec = importlib.import_module("eval_candidates")
    require_api(rt, rp, ec)
    return rt, rp, ec


def require_api(rt, rp, ec) -> None:
    """冻结的加载契约：服务器端源码若版本不符，preflight 直接 BLOCKED 而不是猜。

    这些符号是历史 champion / candidate evaluator 的真实接口（见 README 证据表）：
      runtime.ChampionEnsemble.{pre_head_probs, corrected_class_prob, predict}
      runtime.{CLASSES, per_image_counts, _unwrap_img, _unwrap_meta, inject_repo}
      run_protocol.{validate_settings, build_student}
      eval_candidates.{student_prob, load_node_checkpoint, validate_prob, check_stem, rt}
    """
    need_rt = ("ChampionEnsemble", "CLASSES", "per_image_counts", "_unwrap_img",
               "_unwrap_meta", "inject_repo", "load_config")
    need_rp = ("validate_settings", "build_student")
    need_ec = ("student_prob", "load_node_checkpoint", "validate_prob", "check_stem", "rt")
    need_ce = ("pre_head_probs", "corrected_class_prob", "predict")
    miss = [("runtime.%s" % a) for a in need_rt if not hasattr(rt, a)]
    miss += [("run_protocol.%s" % a) for a in need_rp if not hasattr(rp, a)]
    miss += [("eval_candidates.%s" % a) for a in need_ec if not hasattr(ec, a)]
    ce = getattr(rt, "ChampionEnsemble", None)
    miss += [("ChampionEnsemble.%s" % a) for a in need_ce
             if ce is None or not hasattr(ce, a)]
    if miss:
        raise C.PreflightBlocked(
            "服务器端 FusionAware 源码与冻结接口不符，缺少: %s。"
            "本脚本按历史 runtime.run_protocol.eval_candidates 的真实接口编写，"
            "接口不匹配时**拒绝猜**：请确认 fusionaware_src_dir/exec_dir 指向"
            "历史交接版本（README 有阻塞说明）。" % miss)


def resolve_cfg(settings: dict, rt):
    """从 `rp.validate_settings()` 的返回值中解析 mmseg Config（`settings['_cfg']`）。

    根因：`run_protocol.validate_settings()`（历史交接版 run_protocol.py:79-110）只往
    settings 里写 `_hashes` / `_abs` / `_pending`，**从不写 `_cfg`**；`_cfg` 仅在
    `run_protocol.build_student()` 内部作为副作用被赋值（run_protocol.py:250）。
    历史 `eval_candidates.py` 先 `build_student` 再用 `settings['_cfg']`，所以顺序
    天然成立；本脚本却在 build_student 之前就要用 `cfg.data[val]` 建 val dataset，
    直接索引 `settings['_cfg']` 必然 KeyError。这里按 `build_student` 的**同一构造**
    （`rt.load_config` + `source_checkpoint` 绝对路径补丁）显式解析并写回，消除对
    build_student 调用顺序的隐式依赖。
    """
    cfg = settings.get("_cfg")
    if cfg is None:
        cfg = rt.load_config(settings["repo"], settings["config"])
        cfg.model["source_checkpoint"] = settings["_abs"]["reference_checkpoint"]
        settings["_cfg"] = cfg
    return cfg


def hash_cs_candidates(raw: dict, needed=()) -> list:
    """校验 C/S node checkpoint 的声明一致性；**只对本次需要的候选**检查存在性与 sha。

    未选中的候选只登记声明（arm/steps 冻结一致性必须成立），不阻塞本次导出。
    """
    specs = raw.get("cs_candidates")
    if not isinstance(specs, list) or not specs:
        raise C.ContractError("settings.cs_candidates 必须是至少 1 条 {name,path,arm,steps,sha256}")
    needed = set(needed)
    seen, out = set(), []
    for spec in specs:
        name = str(spec.get("name", ""))
        if not name or name in seen:
            raise C.ContractError("cs_candidates: 名称缺失或重复 %r" % name)
        arm, steps = spec.get("arm"), spec.get("steps")
        if (name in EXPECTED_CANDIDATE_STEPS
                and EXPECTED_CANDIDATE_STEPS[name] != (arm, int(steps))):
            raise C.ContractError(
                "cs_candidates[%s]: arm/steps=%r/%r 与冻结定义 %r 不符"
                % (name, arm, steps, EXPECTED_CANDIDATE_STEPS[name]))
        if arm not in ("C", "S"):
            raise C.ContractError("cs_candidates[%s]: arm 必须是 C/S，得到 %r" % (name, arm))
        rec = {"name": name, "arm": arm, "steps": int(steps),
               "path": None, "sha256": None, "size": None, "checked": False}
        if name in needed:
            p = C.safe_path(spec["path"], "cs_candidates[%s].path" % name)
            if not os.path.isfile(p):
                raise C.PreflightBlocked("cs candidate checkpoint 缺失: %s" % p)
            sha = C.sha256_file(p)
            want = spec.get("sha256")
            if want and want != sha:
                raise C.ContractError("cs_candidates[%s]: sha256 不符（声明 %s… 实际 %s…）"
                                      % (name, str(want)[:12], sha[:12]))
            rec.update(path=p, sha256=sha, size=os.path.getsize(p), checked=True)
        seen.add(name)
        out.append(rec)
    if seen != {"C2000", "S4000"}:
        raise C.ContractError(
            "cs_candidates 必须恰含 %s，得到 %s" % (sorted({"C2000", "S4000"}), sorted(seen)))
    return out


# ---------------------------------------------------------------------------
# 概率工具（float32 硬性检查）
# ---------------------------------------------------------------------------

def to_fp32_numpy(t, where: str, torch) -> np.ndarray:
    """张量 → CPU float32 numpy；**.float() 之前**就拦 fp16/bfloat16。"""
    if not torch.is_tensor(t):
        raise C.ContractError("%s: 期望 torch.Tensor，得到 %r" % (where, type(t).__name__))
    C.check_float32_like(t, where)
    a = t.detach().to("cpu").to(torch.float32).contiguous().numpy()
    if a.ndim == 4:
        if a.shape[0] != 1:
            raise C.ContractError("%s: batch 维必须为 1，得到 %r" % (where, a.shape))
        a = a[0]
    if a.ndim != 3 or a.shape[0] != C.NUM_CLASSES:
        raise C.ContractError("%s: 概率必须是 (4,H,W)，得到 %r" % (where, a.shape))
    return np.ascontiguousarray(a, dtype=np.float32)


def champion_shared_prob(ensemble, pf, ps, pr, img, meta, torch, F):
    """用**已共享**的 pf/ps/pr 复现 `ChampionEnsemble.predict()` 的通道循环。

    与 predict() (runtime.py:775-791) 的唯一区别是不重算成员前向；通道构造逐行一致
    （含 heads 判定、img_f 的双线性插值尺寸、`corrected_class_prob` 的 f7 构造）。
    首图会与 predict() 做逐位等价自检。
    """
    import runtime as rt
    with torch.no_grad():
        ori = tuple(meta[0]["ori_shape"][:2])
        img_f = F.interpolate(img, size=ori, mode="bilinear", align_corners=False)[0]
        chans = []
        for c in range(C.NUM_CLASSES):
            cname = rt.CLASSES[c]
            if cname in ensemble.heads:
                chans.append(ensemble.corrected_class_prob(
                    c, pf, ps, pr, img_f, ensemble.heads[cname]).unsqueeze(0))
            else:
                chans.append(pf[c:c + 1])
        return torch.cat(chans, dim=0)


def main():
    ap = argparse.ArgumentParser(
        description="导出 HRPlus 6 条管线的 val 二值掩膜（packbits 契约，含原子写与自检）")
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out-root", default=None, help="覆盖 settings.export.out_root")
    ap.add_argument("--pipelines", nargs="+", default=list(PIPELINES), choices=list(PIPELINES))
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke：只跑前 N 张并**不写 contract**（分析端会拒绝该目录）")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    log = C.log

    raw = load_raw_settings(args.settings)
    settings_path = C.safe_path(args.settings, "--settings")
    rt, rp, ec = setup_imports(raw)

    settings = rp.validate_settings(settings_path)
    if settings.get("_pending"):
        raise C.PreflightBlocked(
            "settings 中引用的资产不存在（rp.validate_settings 报告）: %s"
            % settings["_pending"])

    export_block = raw.get("export")
    if not isinstance(export_block, dict) or not export_block.get("out_root"):
        raise C.ContractError(
            "settings.export.out_root 未填写（两个导出脚本必须共享同一个 out_root，"
            "共享 GT 才能被复用）")
    out_root = C.safe_path(args.out_root or export_block["out_root"], "out_root")
    data_root = C.safe_path(raw["data_root"], "data_root")
    C.assert_val_only(C.SPLIT,
                      paths=(data_root, out_root, raw["fusionaware_src_dir"]),
                      what="export_hrplus")
    if args.device != "cuda":
        raise C.ContractError(
            "本脚本必须用 CUDA：ChampionEnsemble 内部固定 .cuda()（--device=%s 不支持）"
            % args.device)

    import torch
    import torch.nn.functional as F

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    cs_specs = hash_cs_candidates(
        raw, needed=[n for n in args.pipelines if n in EXPECTED_CANDIDATE_STEPS])
    out_dir = C.ensure_dir(os.path.join(out_root, "hrplus"))
    model_dirs = {
        name: C.ensure_dir(os.path.join(out_dir, name)) for name in args.pipelines
    }
    gt_dir = os.path.join(out_root, "_shared", "gt")

    # ---- 数据集（与 eval_candidates 完全一致的 val 构建方式）----
    rt.inject_repo(settings["repo"])
    resolve_cfg(settings, rt)   # validate_settings 返回里没有 _cfg，必须显式解析
    from mmseg.datasets import build_dataset
    d = copy.deepcopy(settings["_cfg"].data[C.SPLIT])
    d.test_mode = True
    d.data_root = data_root
    d.img_dir = os.path.join(data_root, "images", C.SPLIT)
    d.ann_dir = os.path.join(data_root, "labels", C.SPLIT)
    ds = build_dataset(d)
    C.assert_val_only(C.SPLIT, n_images=len(ds), what="mmseg val dataset")
    gt_maps = ds.get_gt_seg_maps()
    log("val dataset: %d 张（契约要求 %d）" % (len(ds), C.EXPECTED_VAL_IMAGES))

    # ---- 共享 GT：先按 ori_shape 表写/校验一次，再由两个导出脚本共用 ----
    import cv2
    stems_by_meta, image_hw = [], {}
    for i in range(len(ds)):
        meta_i = rt._unwrap_meta(ds[i]["img_metas"])
        stem_i = ec.check_stem(
            os.path.splitext(os.path.basename(meta_i[0]["ori_filename"]))[0], "stem %d" % i)
        stems_by_meta.append(stem_i)
        h, w = (int(x) for x in meta_i[0]["ori_shape"][:2])
        if stem_i in image_hw and tuple(image_hw[stem_i]) != (h, w):
            raise C.ContractError("样本 %s 的 ori_shape 在同一 val 集内不一致" % stem_i)
        image_hw[stem_i] = (h, w)
    if len(set(stems_by_meta)) != len(stems_by_meta):
        raise C.ContractError("val 样本 ID 出现重复")
    gt_man = C.write_or_verify_gt(
        gt_dir, data_root=data_root, split=C.SPLIT, stems=stems_by_meta,
        label_reader=lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED),
        image_hw=image_hw)
    stems = sorted(gt_man["stems"])

    # ---- 模型：champion（既有 ChampionEnsemble）+ C/S（既有 build_student）----
    champion = rt.ChampionEnsemble(settings, settings["champion"])
    log("champion built: members=%s head=%s"
        % ([m["name"] for m in settings["champion"]["members"]],
           settings["champion"]["head"]))

    cand_models = {}
    for spec in cs_specs:
        if spec["name"] not in args.pipelines:
            continue
        model, _ = rp.build_student(settings)          # fresh build + strict B/D init
        model.eval()
        info = ec.load_node_checkpoint(model, spec["path"], spec["name"], spec["arm"])
        if int(info["steps"]) != spec["steps"]:
            raise C.ContractError(
                "candidate %s: payload steps=%r != settings 声明 %r（拒绝：必须是历史定义的那个点）"
                % (spec["name"], info["steps"], spec["steps"]))
        for p_ in model.parameters():
            p_.requires_grad_(False)
        cand_models[spec["name"]] = model
        log("candidate %s loaded: arm=%s steps=%s seed=%s sha=%s…"
            % (spec["name"], info["arm"], info["steps"], info.get("seed"),
               spec["sha256"][:12]))
    torch.cuda.empty_cache()

    # ---- 每图：一次成员前向，六条管线共享 ----
    per_image = {name: {} for name in args.pipelines}
    global_counts = {name: [[0, 0, 0] for _ in range(C.NUM_CLASSES)]
                     for name in args.pipelines}
    checks = {"champion_shared_bitwise_equal": None,
              "champion_shared_max_abs_diff": None,
              "non_ma_channels_bitwise_equal": None,
              "mask_bits_match_runtime_counts": True,
              "dataset_gt_matches_shared_gt": True,
              "images_processed": 0}

    limit = len(ds) if args.limit is None else min(int(args.limit), len(ds))
    t0 = time.time()
    for i in range(limit):
        s = ds[i]
        img = rt._unwrap_img(s["img"]).unsqueeze(0).cuda()
        meta = rt._unwrap_meta(s["img_metas"])
        stem = stems_by_meta[i]
        ori = tuple(int(x) for x in meta[0]["ori_shape"][:2])
        if tuple(image_hw[stem]) != ori:
            raise C.ContractError("样本 %s: ori_shape 本次与前次不一致" % stem)

        gt_np = C.load_gt_npz(os.path.join(gt_dir, C.MASK_FILENAME % stem),
                              where="gt[%s]" % stem)
        gt_bits = gt_np["bits"].reshape(C.NUM_CLASSES, -1)
        gt_ds = np.asarray(gt_maps[i]).astype(np.int64)
        if gt_ds.shape != ori:
            gt_ds = cv2.resize(gt_ds.astype(np.uint8), (ori[1], ori[0]),
                               interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if not np.all(np.isin(gt_ds, (0, 1, 2, 3, 4, 255))):
            raise C.ContractError("样本 %s: val GT 值域越界" % stem)
        gt_ds_bits = np.stack([gt_ds == (c + 1) for c in range(C.NUM_CLASSES)],
                              axis=0).reshape(C.NUM_CLASSES, -1)
        if not np.array_equal(gt_ds_bits, gt_bits):
            checks["dataset_gt_matches_shared_gt"] = False
            raise C.ContractError(
                "样本 %s: mmseg 数据集 GT 与共享 GT 不一致 —— 两个导出脚本的 GT 必须同源"
                % stem)

        with torch.no_grad():
            pf, ps, pr = champion.pre_head_probs(img, meta)
            pf = pf.float()
            ps = ps.float()
            pr = pr.float()
            cand_prob = {}
            for name, model in cand_models.items():
                cand_prob[name] = ec.student_prob(model, img, meta).float()
            champ = champion_shared_prob(  # noqa: F821 (定义见下)
                champion, pf, ps, pr, img, meta, torch, F)

        probs = {
            "champion": champ,
            "BASE_0": pf,
            "D_blend": BLEND_W * ps + (1.0 - BLEND_W) * pr,
            "C2000": cand_prob.get("C2000"),
            "S4000": cand_prob.get("S4000"),
        }
        probs["MIX_C2000"] = (MIX_CHAMPION * champ + MIX_CANDIDATE * cand_prob["C2000"]
                              if "C2000" in cand_prob else None)

        # ---- 首图自检：共享路径 vs champion.predict() 逐位等价 ----
        if i == 0 and "champion" in args.pipelines:
            ref = champion.predict(img, meta).float()
            diff = float((ref - champ).abs().max().item())
            checks["champion_shared_max_abs_diff"] = diff
            checks["champion_shared_bitwise_equal"] = bool(torch.equal(ref, champ))
            if not checks["champion_shared_bitwise_equal"]:
                raise C.ContractError(
                    "共享 pf/ps/pr 复现的 champion 与 champion.predict() 不逐位相等"
                    "（max|Δ|=%.3e）—— 单次前向共享路径不可信，拒绝出图" % diff)
            checks["non_ma_channels_bitwise_equal"] = bool(torch.equal(champ[:3], pf[:3]))
            if not checks["non_ma_channels_bitwise_equal"]:
                raise C.ContractError(
                    "champion 的 EX/HE/SE 通道与 BASE_0 不逐位相等 —— H4 只应改 MA")
            log("首图自检通过：champion 共享路径 bitwise-equal（max|Δ|=%.1e），"
                "非 MA 三通道与 BASE_0 逐位一致" % diff)

        for name in args.pipelines:
            prob = probs.get(name)
            if prob is None:
                raise C.ContractError("管线 %s 的依赖候选未加载（--pipelines 组合不完整）" % name)
            prob_np = to_fp32_numpy(prob, "%s[%s]" % (name, stem), torch)
            ec.validate_prob(prob_np, ori, "%s[%s]" % (name, stem))
            bits = C.binarize_strict(prob_np, "%s[%s]" % (name, stem))
            # Each pipeline owns its mask files.  A shared out_dir/stem path
            # would let later candidates overwrite earlier predictions.
            rec = C.save_mask_npz(
                os.path.join(model_dirs[name], C.MASK_FILENAME % stem), bits, ori)
            cnt_rt = ec.rt.per_image_counts(prob_np, gt_ds)
            cnt_bits = C.counts_from_bits(bits.reshape(C.NUM_CLASSES, -1), gt_bits)
            if cnt_rt != cnt_bits:
                checks["mask_bits_match_runtime_counts"] = False
                raise C.ContractError(
                    "管线 %s 样本 %s: packbits 重算计数 %r != runtime.per_image_counts %r"
                    % (name, stem, cnt_bits, cnt_rt))
            rec["counts"] = {C.CLASSES[c]: list(cnt_rt[c]) for c in range(C.NUM_CLASSES)}
            per_image[name][stem] = rec
            for c in range(C.NUM_CLASSES):
                for j in range(3):
                    global_counts[name][c][j] += int(cnt_rt[c][j])
        checks["images_processed"] += 1
        if (i + 1) % 10 == 0 or (i + 1) == limit:
            log("export %d/%d (%.1fs)" % (i + 1, limit, time.time() - t0))

        if not checks["mask_bits_match_runtime_counts"]:
            raise C.ContractError("位图与 runtime 计数链路出现不一致（见上）")
        del img, meta, gt_bits, gt_ds_bits, pf, ps, pr, champ, probs, cand_prob
        torch.cuda.empty_cache()

    if limit != C.EXPECTED_VAL_IMAGES:
        log("SMOKE: 只处理了 %d/%d 张 —— 按契约**不写 contract**，该目录不可被分析端使用。"
            % (limit, C.EXPECTED_VAL_IMAGES))
        print(json.dumps({"smoke": True, "images": limit, "checks": checks},
                         ensure_ascii=False, indent=2))
        return 0

    # ---- contract（每个管线一份，最后原子写；先写掩膜后写契约 ⇒ 半截产物不可用）----
    identity = build_identity(settings, settings_path, cs_specs, raw)
    for name in args.pipelines:
        counts = global_counts[name]
        miou, per_class, detail = C.iou_from_counts(counts)
        contract = {
            "model_name": name,
            "pipeline_family": PIPELINE_FAMILY,
            "split": C.SPLIT,
            "n_images": len(stems),
            "stems": stems,
            "eval_contract": C.build_eval_contract(
                pipeline_family=PIPELINE_FAMILY,
                eval_loader="mmseg_test_pipeline(ori_shape 还原)",
                inference="whole_image",
                tta="hflip TTA（4 成员各 2 次前向）" if name != "C2000" and name != "S4000"
                    else "none（候选只取独立 student 概率）"),
            "pipeline_definition": pipeline_definition(name),
            "identity": identity,
            "gt": {
                "gt_contract_sha256": gt_man["gt_contract_sha256"],
                "gt_semantic_fingerprint": gt_man["gt_semantic_fingerprint"],
                "data_root": gt_man["data_root"],
                "split": C.SPLIT,
            },
            "per_image": per_image[name],
            "global_counts": {C.CLASSES[c]: list(counts[c]) for c in range(C.NUM_CLASSES)},
            "metrics": {"mIoU_lesion4": miou, "per_class_iou": per_class,
                        "per_class_detail": detail},
            "self_checks": dict(checks, counts_from_masks_verified=True,
                                mask_count_total=sum(1 for _ in per_image[name])),
            "blockers": list(raw.get("known_blockers", [])),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        C.finalize_model_dir(model_dirs[name], contract)
        log("pipeline %-10s mIoU=%.6f  per_class=%s"
            % (name, miou, {k: (None if v is None else round(v, 6))
                            for k, v in per_class.items()}))

    C.atomic_write_json(os.path.join(out_root, "RUN_MANIFEST_hrplus.json"), {
        "schema": C.RUN_MANIFEST_SCHEMA,
        "script": "export_hrplus_expert_masks.py",
        "script_sha256": C.sha256_file(
            os.path.join(_HERE, "export_hrplus_expert_masks.py")),
        "split": C.SPLIT,
        "n_images": len(stems),
        "models": list(args.pipelines),
        "model_dir": out_dir,
        "gt_contract_sha256": gt_man["gt_contract_sha256"],
        "gt_semantic_fingerprint": gt_man["gt_semantic_fingerprint"],
        "settings_sha256": C.sha256_file(settings_path),
        "complete": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    log("DONE hrplus: %s" % out_dir)
    return 0


def build_identity(settings, settings_path, cs_specs, raw) -> dict:
    """模型/配置/权重 SHA256（含 C/S checkpoint 与脚本自身）。"""
    import runtime as rt
    cfg = resolve_cfg(settings, rt)   # 不假设 build_student 已跑过（--pipelines 可不含 C/S）
    cfg_text_sha = hashlib.sha256(cfg.dump().encode("utf-8")).hexdigest()
    return {
        "weights_sha256": settings["_hashes"],
        "config_text_sha256": cfg_text_sha,
        "cs_candidates": [dict(s) for s in cs_specs],
        "code_sha256": {
            # `_HERE` was frozen at import time.  The legacy runtime changes cwd
            # to settings.repo, so resolving a relative __file__ here is unsafe.
            "export_hrplus_expert_masks.py": C.sha256_file(
                os.path.join(_HERE, "export_hrplus_expert_masks.py")),
            "expert_mask_contract.py": C.sha256_file(
                os.path.join(_HERE, "expert_mask_contract.py")),
            "eval_candidates.py": C.sha256_file(
                os.path.join(raw["fusionaware_exec_dir"], "eval_candidates.py")),
            "runtime.py": C.sha256_file(
                os.path.join(raw["fusionaware_src_dir"], "runtime.py")),
            "run_protocol.py": C.sha256_file(
                os.path.join(raw["fusionaware_src_dir"], "run_protocol.py")),
            "fusion_core.py": C.sha256_file(
                os.path.join(raw["fusionaware_src_dir"], "fusion_core.py")),
        },
        "settings_file_sha256": C.sha256_file(settings_path),
        "repo": settings["repo"],
        "data_root": settings["data_root"],
    }


def pipeline_definition(name: str) -> dict:
    base = {
        "members": ["D×3", "E×2", "B×1", "E4×1"],
        "member_probability_weighting": "3:2:1:1（概率级加权，分母 2·Σw 含 hflip TTA）",
        "tta": "hflip",
        "four_member_forward_count_per_image": 8,
    }
    if name == "champion":
        return dict(base, name=name, head="H4（后融合 MA 校正头，7ch→16→16→1，Δ=6·tanh）",
                    head_channels="仅 MA", blend=None,
                    note="= BASE_0 + H4 头；EX/HE/SE 与 BASE_0 逐位相同")
    if name == "BASE_0":
        return dict(base, name=name, head=None,
                    note="4 成员概率加权 + hflip TTA，不接任何校正头")
    if name == "D_blend":
        return {"name": name, "head": None,
                "definition": "0.5 · P_student(D) + 0.5 · P_reference(D)",
                "student": "D 成员 _student_prob（无 flip）",
                "reference": "D 成员 _reference_prob == B 冻结基座（无 flip）",
                "weight": BLEND_W,
                "history": "val mIoU 0.41614841513965567（HE 单类历史最优 0.44727）",
                "note": "与 HRDecoder_审查轮 diag_train_inference_contract.py 的 blend 同式"}
    if name == "C2000":
        return {"name": name, "head": None, "tta": "none",
                "definition": "C 臂 node checkpoint 的独立 student 概率",
                "arm": "C", "steps": 2000,
                "loader": "run_protocol.build_student + eval_candidates.load_node_checkpoint",
                # 键名必须是 probability_source：`probability` 属于
                # expert_mask_contract.FORBIDDEN_CONTRACT_KEYS 的保留键（硬黑名单），
                # 用它会触发"禁止保存概率"的误判而 fail closed（历史 exit=2 根因）。
                "probability_source": "_student_prob（绝不使用 model.inference()）"}
    if name == "S4000":
        return {"name": name, "head": None, "tta": "none",
                "definition": "S 臂 node checkpoint 的独立 student 概率",
                "arm": "S", "steps": 4000,
                "loader": "run_protocol.build_student + eval_candidates.load_node_checkpoint",
                # 同 C2000：保留键 `probability` 会被载荷禁令误伤，改用 _source 命名。
                "probability_source": "_student_prob（绝不使用 model.inference()）"}
    if name == "MIX_C2000":
        return {"name": name, "head": "H4（经 champion 分支带入）",
                "definition": "0.75 · champion + 0.25 · C2000",
                "mix_champion": MIX_CHAMPION, "mix_candidate": MIX_CANDIDATE,
                "weight_scan": False,
                "history": "val mIoU 0.4286569005579753，且被 locked_candidate.json 锁定",
                "note": "与 eval_candidates.py 的冻结 mix 同式（0.75/0.25，不可扫参）"}
    raise C.ContractError("未知管线 %r" % name)


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
