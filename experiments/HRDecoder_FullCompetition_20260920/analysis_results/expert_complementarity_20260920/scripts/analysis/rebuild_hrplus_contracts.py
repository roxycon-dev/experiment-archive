#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rebuild_hrplus_contracts — **离线**从已有 76 张 val 掩膜补写 HRPlus contract.json。

背景（为什么需要这个脚本）
--------------------------
`export_hrplus_expert_masks.py` 的第六步（写 contract）在 C2000 上 fail closed：
`pipeline_definition("C2000")` 把**纯文字说明**放在保留键 `"probability"` 下，
被 `expert_mask_contract.assert_no_probability_payload` 的键名硬黑名单误判为
概率载荷 ⇒ `hrplus_exit_code.txt = 2`。此时六个目录的 76 张掩膜**都已写完**，
只有 C2000 / MIX_C2000 / S4000 三份 contract 没落盘（champion/BASE_0/D_blend 已写）。

掩膜本身是完全可信的证据（已通过运行时逐图自检），没有任何理由为了补一份 JSON
重跑 GPU。本脚本因此**只读掩膜、只写 JSON**：

- 逐图记录（file_sha256 / semantic_sha256 / n_pixels / orig_hw / packed_bytes /
  counts）**全部从磁盘掩膜重算**，不依赖任何内存状态；
- `identity` 与 `self_checks` 两个块在原导出里是**整轮共享的同一份内存对象**
  （`build_identity(...)` 与 `checks` 在循环外构造一次），所以对 C2000 / MIX_C2000 /
  S4000 而言，它们与同轮写出的 champion contract 逐字节相同 —— 直接复用
  reference contract 中的这两块，就是"原run本该写出的内容"，不是重新发明。
- **不使用 GPU、不联网、不读 test、不重算任何概率**；`per_image` 里的 counts 由
  掩膜位重算（而原 run 已机械证明"掩膜位计数 == runtime.per_image_counts"）。

自证（--verify）
----------------
对已经有 contract 的管线（champion/BASE_0/D_blend）用同一套离线逻辑重建，并与
磁盘上的真实 contract 做**深度逐字段比对**。除 `created_at` 外必须完全一致；
任何差异都说明"离线重建不可信"，脚本 fail closed（exit 2），拒绝写任何东西。
这是把"离线补写"从"我猜"变成"机械可验证"的关键一步。

用法
----
    python analysis/rebuild_hrplus_contracts.py --settings analysis/expert_eval_settings.json \
        --verify --write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import expert_mask_contract as C  # noqa: E402
import export_hrplus_expert_masks as E  # noqa: E402

DEFAULT_PIPELINES = ("C2000", "MIX_C2000", "S4000")
VERIFY_PIPELINES = ("champion", "BASE_0", "D_blend")
# 重建结果与磁盘 contract 允许不同的键（时间戳；其余一律必须一致）
IGNORED_DIFF_KEYS = ("created_at",)


def deep_diff(a, b, path="", out=None):
    """返回 [(path, a_val, b_val)]，递归比对 dict/list/标量。"""
    if out is None:
        out = []
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append((path or "<root>", a, b))
        return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORED_DIFF_KEYS:
                continue
            p = "%s/%s" % (path, k)
            if k not in a or k not in b:
                out.append((p, a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                deep_diff(a[k], b[k], p, out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((path + "/<len>", len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_diff(x, y, "%s[%d]" % (path, i), out)
    else:
        if a != b:
            out.append((path, a, b))
    return out


def tta_string(name: str) -> str:
    """与 export_hrplus_expert_masks.main() 里的取值逐字一致。"""
    if name in ("C2000", "S4000"):
        return "none（候选只取独立 student 概率）"
    return "hflip TTA（4 成员各 2 次前向）"


def rebuild_per_image(model_dir, gt_dir, stems):
    """从磁盘掩膜 + 共享 GT 重算逐图记录与累计 counts。"""
    per_image, counts = {}, [[0, 0, 0] for _ in range(C.NUM_CLASSES)]
    for stem in stems:
        mp = os.path.join(model_dir, C.MASK_FILENAME % stem)
        if not os.path.isfile(mp):
            raise C.ContractError("掩膜缺失: %s" % mp)
        m = C.load_mask_npz(mp, where="%s[%s]" % (os.path.basename(model_dir), stem))
        g = C.load_gt_npz(os.path.join(gt_dir, C.MASK_FILENAME % stem),
                          where="gt[%s]" % stem)
        if tuple(m["orig_hw"]) != tuple(g["orig_hw"]):
            raise C.ContractError("样本 %s: 掩膜 orig_hw %r != GT orig_hw %r"
                                 % (stem, tuple(m["orig_hw"]), tuple(g["orig_hw"])))
        cnt = C.counts_from_bits(m["bits"].reshape(C.NUM_CLASSES, -1),
                                 g["bits"].reshape(C.NUM_CLASSES, -1))
        rec = {
            "mask_file": C.MASK_FILENAME % stem,
            "file_sha256": m["file_sha256"],
            "semantic_sha256": C.semantic_sha256(
                m["packed"], m["n_pixels"], m["orig_hw"], kind="model-mask"),
            "n_pixels": int(m["n_pixels"]),
            "orig_hw": [int(m["orig_hw"][0]), int(m["orig_hw"][1])],
            "packed_bytes": int(m["packed"].shape[1]),
            "counts": {C.CLASSES[c]: list(cnt[c]) for c in range(C.NUM_CLASSES)},
        }
        per_image[stem] = rec
        for c in range(C.NUM_CLASSES):
            for j in range(3):
                counts[c][j] += int(cnt[c][j])
    return per_image, counts


def build_contract(name, per_image, counts, *, stems, identity, self_checks,
                   gt_man, blockers):
    miou, per_class, detail = C.iou_from_counts(counts)
    return {
        "model_name": name,
        "pipeline_family": E.PIPELINE_FAMILY,
        "split": C.SPLIT,
        "n_images": len(stems),
        "stems": list(stems),
        "eval_contract": C.build_eval_contract(
            pipeline_family=E.PIPELINE_FAMILY,
            eval_loader="mmseg_test_pipeline(ori_shape 还原)",
            inference="whole_image",
            tta=tta_string(name)),
        "pipeline_definition": E.pipeline_definition(name),
        "identity": identity,
        "gt": {
            "gt_contract_sha256": gt_man["gt_contract_sha256"],
            "gt_semantic_fingerprint": gt_man["gt_semantic_fingerprint"],
            "data_root": gt_man["data_root"],
            "split": C.SPLIT,
        },
        "per_image": per_image,
        "global_counts": {C.CLASSES[c]: list(counts[c]) for c in range(C.NUM_CLASSES)},
        "metrics": {"mIoU_lesion4": miou, "per_class_iou": per_class,
                    "per_class_detail": detail},
        "self_checks": dict(self_checks),
        "blockers": list(blockers),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser(description="离线补写 HRPlus contract（不重跑 GPU）")
    ap.add_argument("--settings", required=True)
    ap.add_argument("--pipelines", nargs="+", default=list(DEFAULT_PIPELINES))
    ap.add_argument("--verify-pipelines", nargs="+", default=list(VERIFY_PIPELINES))
    ap.add_argument("--verify", action="store_true",
                    help="先用已有 contract 的管线做逐字段自证，通过才继续")
    ap.add_argument("--write", action="store_true", help="真正写 contract.json")
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()
    log = C.log

    settings_path = C.safe_path(args.settings, "--settings")
    with open(settings_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out_root = C.safe_path(args.out_root or raw["export"]["out_root"], "out_root")
    data_root = C.safe_path(raw["data_root"], "data_root")
    C.assert_val_only(C.SPLIT, paths=(out_root, data_root), what="rebuild_contracts")

    out_dir = os.path.join(out_root, "hrplus")
    gt_dir = os.path.join(out_root, "_shared", "gt")
    gt_man = C.load_gt_contract(gt_dir)["contract"]
    stems = sorted(gt_man["stems"])
    if len(stems) != C.EXPECTED_VAL_IMAGES:
        raise C.ContractError("stems 数量 %d != %d" % (len(stems), C.EXPECTED_VAL_IMAGES))
    log("共享 GT 契约载入: %d 张，fingerprint=%s" % (len(stems),
       gt_man["gt_semantic_fingerprint"][:12]))

    # ---- reference contract：identity / self_checks 是整轮共享对象，逐字复用 ----
    ref_pipe = args.verify_pipelines[0]
    ref_path = os.path.join(out_dir, ref_pipe, C.CONTRACT_FILENAME)
    if not os.path.isfile(ref_path):
        raise C.PreflightBlocked("reference contract 缺失: %s" % ref_path)
    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    identity = ref["identity"]
    self_checks = ref["self_checks"]
    log("reference = %s（identity/self_checks 整轮共享，逐字复用）" % ref_pipe)
    assert ref["gt"]["gt_contract_sha256"] == gt_man["gt_contract_sha256"], \
        "reference contract 的 GT 契约与共享 GT 不一致"
    assert ref["n_images"] == len(stems) and list(ref["stems"]) == stems, \
        "reference contract 的 stems 与共享 GT 不一致"

    blockers = raw.get("known_blockers", [])
    results = {}

    # ---- 自证：用离线逻辑重建已有 contract 的管线，逐字段比对 ----
    if args.verify:
        bad = 0
        for name in args.verify_pipelines:
            mdir = os.path.join(out_dir, name)
            cpath = os.path.join(mdir, C.CONTRACT_FILENAME)
            per_image, counts = rebuild_per_image(mdir, gt_dir, stems)
            built = build_contract(name, per_image, counts, stems=stems,
                                   identity=identity, self_checks=self_checks,
                                   gt_man=gt_man, blockers=blockers)
            if os.path.isfile(cpath):
                with open(cpath, "r", encoding="utf-8") as f:
                    on_disk = json.load(f)
                diff = deep_diff(built, on_disk)
                if diff:
                    bad += 1
                    log("VERIFY FAIL %-10s: %d 处差异" % (name, len(diff)))
                    for p, a, b in diff[:20]:
                        log("    %s: rebuilt=%r disk=%r" % (p, a, b))
                else:
                    log("VERIFY OK   %-10s: 与磁盘 contract 逐字段一致（除 created_at）"
                        % name)
            else:
                log("VERIFY N/A  %-10s: 磁盘无 contract（跳过比对）" % name)
            results[name] = built
        if bad:
            raise C.ContractError(
                "离线重建与磁盘真实 contract 存在差异 ⇒ 重建逻辑不可信，拒绝写任何 contract")
        log("自证通过：离线重建 == 原 run 产物（仅 created_at 不同）")

    # ---- 补写 / 报告目标管线的数值 ----
    report = {}
    for name in args.pipelines:
        mdir = os.path.join(out_dir, name)
        per_image, counts = rebuild_per_image(mdir, gt_dir, stems)
        contract = build_contract(name, per_image, counts, stems=stems,
                                  identity=identity, self_checks=self_checks,
                                  gt_man=gt_man, blockers=blockers)
        m = contract["metrics"]
        report[name] = {
            "mIoU_lesion4": m["mIoU_lesion4"],
            "per_class_iou": m["per_class_iou"],
            "n_masks": len(per_image),
        }
        log("target %-10s mIoU=%.12f  per_class=%s  masks=%d"
            % (name, m["mIoU_lesion4"],
               {k: (None if v is None else round(v, 6))
                for k, v in m["per_class_iou"].items()}, len(per_image)))
        if args.write:
            C.finalize_model_dir(mdir, contract)
            # 写回后立刻用分析端读回路子完整校验一遍
            got = C.read_model_dir(mdir, expected_family=E.PIPELINE_FAMILY)
            assert abs(got["contract"]["metrics"]["mIoU_lesion4"]
                       - m["mIoU_lesion4"]) < 1e-15
            log("READBACK OK %-10s（read_model_dir 通过）" % name)

    # ---- RUN_MANIFEST（原 run 没走到这一步）----
    if args.write:
        C.atomic_write_json(os.path.join(out_root, "RUN_MANIFEST_hrplus.json"), {
            "schema": C.RUN_MANIFEST_SCHEMA,
            "script": "rebuild_hrplus_contracts.py",
            "script_sha256": C.sha256_file(os.path.join(_HERE, "rebuild_hrplus_contracts.py")),
            "masks_produced_by": {
                "script": "export_hrplus_expert_masks.py",
                "script_sha256": identity["code_sha256"].get("export_hrplus_expert_masks.py"),
                "note": "掩膜由原 run 产出；本次只用离线重建补齐 contract.json",
            },
            "contract_writer_fix": {
                "file": "expert_mask_contract.py",
                "bug": "pipeline_definition 的元数据字段名 'probability' 撞上保留键黑名单",
                "fix": "元数据字段改名 probability_source；保留键黑名单不放松，"
                       "另加语义级浮点数组禁令",
            },
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
        C.atomic_write_json(os.path.join(out_root, "REBUILD_REPORT.json"), {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gpu_used": False,
            "test_used": False,
            "verified_against_disk": bool(args.verify),
            "verified_pipelines": list(args.verify_pipelines),
            "rebuilt_pipelines": report,
        })
        log("RUN_MANIFEST_hrplus.json / REBUILD_REPORT.json 已写出")

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
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
