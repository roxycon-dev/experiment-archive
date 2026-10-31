#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 D checkpoint 预处理成"只含 student"的版本，供 J0/J1 的 load_from 使用。

为什么必须这样做（子代理审计发现的机制性缺陷，不是猜测）：
  D 的 state_dict 里有 1877 个 `reference.*` 键，而 `self.reference` 恰恰也是 `reference.*`。
  mmseg 的 `load_checkpoint(..., strict=False)` **只抑制不匹配报错，不阻止匹配键被覆盖**
  → strict=False 会把 D 的 reference 写进我们本该恒为 B 的冻结 reference。
  实测这份 iter_20000 的 reference 与 B 逐张量全等（1877/1877，0 差异），所以"这次没出事"，
  但那只是"恰好相等"，不是机制保证 —— 一旦换了 reference 未冻结的 D，污染会静默发生。

本脚本：剔除 `reference.*`，并丢弃 optimizer 状态（load_from 不恢复优化器，留着白占 ~350 MB）。
保留 meta 与其余全部键，载入后 reference 只可能来自构造函数 strict 载入的 B。

用法：python prep_D_ckpt.py <in.pth> <out.pth>
"""
import sys

import torch


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ck = torch.load(src, map_location="cpu")
    print("输入:", src)
    print("  顶层键:", sorted(ck.keys()))
    sd = ck["state_dict"]
    n_all = len(sd)
    ref = [k for k in sd if k.startswith("reference.")]
    rest = {k: v for k, v in sd.items() if not k.startswith("reference.")}
    print("  原始参数键: %d（其中 reference.* = %d，剔除后 = %d）" % (n_all, len(ref), len(rest)))
    assert ref, "该 ckpt 没有 reference.* 键，请确认是否传对了文件"
    assert not any(k.startswith("reference.") for k in rest)
    out = {"meta": ck.get("meta", {}), "state_dict": rest,
           "stripped": {"reference_keys": len(ref), "source": src,
                        "note": "reference.* 已剔除：reference 必须只来自构造函数的 B 严格载入"}}
    torch.save(out, dst)
    import os
    print("输出:", dst, "%.1f MB" % (os.path.getsize(dst) / 1048576))
    # 复核
    chk = torch.load(dst, map_location="cpu")
    bad = [k for k in chk["state_dict"] if k.startswith("reference.")]
    assert not bad
    meta = chk.get("meta", {})
    print("复核: reference.* = 0 ✓ ; meta.iter =", meta.get("iter"), "meta.epoch =", meta.get("epoch"))
    print("OK")


if __name__ == "__main__":
    main()
