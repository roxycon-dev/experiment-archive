#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase A2/A4：checkpoint 身份 + split 边界（prompt §3.2 / §3.4）。

要点：
  - torch>=2.6 的 torch.load 默认 weights_only=True，会拒绝含 numpy scalar 的
    mmseg checkpoint。此处显式 weights_only=False（本地可信文件），并把这一
    环境差异记录进报告 —— 服务器是 torch 2.0.1，不会有这个问题。
  - checkpoint 对比不只比键名，还逐键比 shape 与 dtype，定位字节数差异来源。
  - split 只做完整性/不相交审计，不使用 test 标签统计指导任何设计。
"""
import argparse
import hashlib
import json
import os
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
    return h.hexdigest()


def load_ckpt(p):
    import torch
    try:
        return torch.load(p, map_location="cpu", weights_only=False), "weights_only=False"
    except TypeError:
        return torch.load(p, map_location="cpu"), "legacy_default"


def summarize_ckpt(p, tag):
    import torch
    ck, mode = load_ckpt(p)
    out = {"tag": tag, "path": p, "load_mode": mode,
           "bytes": os.path.getsize(p),
           "sha256": sha256_file(p)}
    if isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
        out["container_keys"] = sorted(ck.keys())
        out["meta"] = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                           else "<%s>" % type(v).__name__)
                       for k, v in (ck.get("meta", {}) or {}).items()}
        out["has_optimizer"] = "optimizer" in ck
        if out["has_optimizer"]:
            opt = ck["optimizer"]
            out["optimizer_keys"] = sorted(opt.keys()) if isinstance(opt, dict) else "<%s>" % type(opt).__name__
            st = opt.get("state", {}) if isinstance(opt, dict) else {}
            out["optimizer_param_groups"] = len(opt.get("param_groups", [])) if isinstance(opt, dict) else None
            out["optimizer_state_entries"] = len(st) if hasattr(st, "__len__") else None
    else:
        sd = ck
        out["container_keys"] = "RAW_STATE_DICT"
    keys = list(sd.keys())
    out["n_keys"] = len(keys)
    out["top_prefixes"] = sorted(set(k.split(".")[0] for k in keys))
    out["has_ma_refiner"] = any(k.startswith("ma_refiner.") for k in keys)
    out["has_reference"] = any(k.startswith("reference.") for k in keys)
    out["n_ma_refiner"] = sum(1 for k in keys if k.startswith("ma_refiner."))
    out["n_reference"] = sum(1 for k in keys if k.startswith("reference."))
    # 参数张量总元素数与总字节数（用于解释文件大小差异）
    import torch as _t
    tot_el = tot_by = 0
    for k in keys:
        v = sd[k]
        if _t.is_tensor(v):
            tot_el += v.numel()
            tot_by += v.numel() * v.element_size()
    out["total_param_elements"] = tot_el
    out["total_param_bytes"] = tot_by
    out["_sd"] = sd
    return out


def key_shape_map(sd):
    m = {}
    for k, v in sd.items():
        if hasattr(v, "shape"):
            m[k] = (tuple(v.shape), str(v.dtype))
        else:
            m[k] = (None, type(v).__name__)
    return m


def compare_ckpt(a, b):
    ka, kb = set(a["_sd"].keys()), set(b["_sd"].keys())
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)
    ma, mb = key_shape_map(a["_sd"]), key_shape_map(b["_sd"])
    shape_diff = []
    for k in sorted(ka & kb):
        if ma[k] != mb[k]:
            shape_diff.append({"key": k, "a": ma[k], "b": mb[k]})
    return {"only_in_a": only_a[:50], "n_only_in_a": len(only_a),
            "only_in_b": only_b[:50], "n_only_in_b": len(only_b),
            "common": len(ka & kb), "shape_or_dtype_diff": shape_diff[:50],
            "n_shape_diff": len(shape_diff)}


def split_analysis(paths):
    out = {}
    for label, p in paths.items():
        if not p or not os.path.isfile(p):
            out[label] = {"status": "MISSING", "path": p}
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        rec = {"status": "FOUND", "path": p, "sha256": sha256_file(p),
               "bytes": os.path.getsize(p), "type": type(d).__name__}
        if isinstance(d, dict):
            ks = list(d.keys())
            rec["n_entries"] = len(ks)
            rec["sample_entry"] = {ks[0]: d[ks[0]]} if ks else None
            # 统计 value 的结构
            from collections import Counter
            vt = Counter(type(v).__name__ for v in d.values())
            rec["value_types"] = dict(vt)
            if vt.get("list"):
                ln = Counter(len(v) for v in d.values() if isinstance(v, list))
                rec["value_list_lengths"] = dict(ln)
                # 若 list 第一项是 split 名，统计各 split
                first = Counter(str(v[0]) for v in d.values()
                                if isinstance(v, list) and v)
                if len(first) <= 10:
                    rec["first_element_counts"] = dict(first)
        elif isinstance(d, list):
            rec["n_entries"] = len(d)
            rec["sample_entry"] = d[0] if d else None
        out[label] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-assignment", required=True)
    ap.add_argument("--data-manifest", required=True)
    ap.add_argument("--ckpts", action="append", default=[],
                    help="name=relpath，可重复")
    args = ap.parse_args()

    ckpts = [c.split("=", 1) for c in args.ckpts]
    results = {}
    for name, rel in ckpts:
        p = os.path.join(args.ckpt_root, rel)
        if not os.path.isfile(p):
            results[name] = {"status": "MISSING", "rel": rel}
            print("✗ %-18s MISSING" % name)
            continue
        try:
            s = summarize_ckpt(p, name)
            results[name] = s
            print("✓ %-18s keys=%-6d prefixes=%s  ma_refiner=%s reference=%s  params=%.1fMB"
                  % (name, s["n_keys"], s["top_prefixes"], s["has_ma_refiner"],
                     s["has_reference"], s["total_param_bytes"] / 1e6))
        except Exception as e:
            results[name] = {"status": "ERROR", "rel": rel, "reason": repr(e)}
            print("✗ %-18s ERROR %r" % (name, e))

    comp = {}
    if "b_weights" in results and "d_student_only" in results \
            and "n_keys" in results.get("b_weights", {}) \
            and "n_keys" in results.get("d_student_only", {}):
        comp["b_weights_vs_d_student_only"] = compare_ckpt(
            results["b_weights"], results["d_student_only"])
        c = comp["b_weights_vs_d_student_only"]
        print("\n== b_weights vs d_student_only ==")
        print("   common=%d  only_in_b=%d  only_in_d=%d  shape_diff=%d"
              % (c["common"], c["n_only_in_a"], c["n_only_in_b"], c["n_shape_diff"]))
        if c["n_only_in_a"]:
            print("   only_in_b 样例:", c["only_in_a"][:6])
        if c["n_only_in_b"]:
            print("   only_in_d 样例:", c["only_in_b"][:6])
        if c["n_shape_diff"]:
            for x in c["shape_diff"][:6]:
                print("   shape差:", x)
        # 容器级差异（用于解释文件大小差异）
        cb = results["b_weights"].get("container_keys")
        cd = results["d_student_only"].get("container_keys")
        print("   container: b=%s  d=%s" % (cb, cd))
        print("   meta: b=%s" % json.dumps(results["b_weights"].get("meta"),
                                           ensure_ascii=False)[:160])
        print("         d=%s" % json.dumps(results["d_student_only"].get("meta"),
                                           ensure_ascii=False)[:160])
        print("   has_optimizer: b=%s  d=%s" % (results["b_weights"].get("has_optimizer"),
                                                results["d_student_only"].get("has_optimizer")))
        print("   文件字节: b=%d  d=%d  diff=%d"
              % (results["b_weights"]["bytes"], results["d_student_only"]["bytes"],
                 results["b_weights"]["bytes"] - results["d_student_only"]["bytes"]))
        print("   参数字节: b=%d  d=%d  diff=%d"
              % (results["b_weights"]["total_param_bytes"],
                 results["d_student_only"]["total_param_bytes"],
                 results["b_weights"]["total_param_bytes"]
                 - results["d_student_only"]["total_param_bytes"]))

    splits = split_analysis({
        "split_assignment": args.split_assignment,
        "data_manifest": args.data_manifest,
    })
    print("\n== split ==")
    for k, v in splits.items():
        if v.get("status") != "FOUND":
            print("  -", k, v)
            continue
        print("  %-18s type=%s n=%s value_types=%s"
              % (k, v.get("type"), v.get("n_entries"), v.get("value_types")))
        if v.get("value_list_lengths"):
            print("       list 长度分布:", v["value_list_lengths"])
        if v.get("first_element_counts"):
            print("       首元素分布:", v["first_element_counts"])
        if v.get("sample_entry"):
            print("       样例:", json.dumps(v["sample_entry"], ensure_ascii=False)[:200])

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "generated_at": datetime.now(CST).isoformat(),
        "torch_weights_only_note": (
            "torch>=2.6 defaults weights_only=True and rejects mmseg checkpoints "
            "containing numpy scalars. Server uses torch 2.0.1 (default False). "
            "This is an environment contract difference, recorded per prompt §4.1."),
        "checkpoints": {k: {kk: vv for kk, vv in v.items() if kk != "_sd"}
                        for k, v in results.items()},
        "checkpoint_comparison": comp,
        "splits": splits,
    }
    p = os.path.join(args.out_dir, "phaseA_identity_split.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)
    print("\n写出:", p)


if __name__ == "__main__":
    main()
