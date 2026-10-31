#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""契约条目 → 运行时证据 的机器对账表（contract audit ledger）。

存在理由（2026-09-17 D-006 的直接产物）：
  之前我核对契约的方式是"回忆我实现了什么，然后在日志里找它的证据"——
  于是**漏实现的条目永远不会被检查到**（val_eval 从未被调用，T0/T1 跑完 8000 步
  却零 val 记录，没人报警）。本脚本把方向反过来：
      契约要求出现的输出模式 → 必须在日志里找到；找不到就是 FAIL。
  「证据缺席」被变成红灯，而不是沉默。

对每条契约要求，检查方式 = 在日志里搜期望模式；文件类要求 = 在 out-dir 里查存在性。
尚未跑到的条目记为 PENDING（不误报），但**只要 iter 已越过该 val 点仍无记录，就是 FAIL**。

用法：
  python3 contract_audit.py --log T0v2.log --variant T0 --seed 1 --iters 8000 \
      --out-dir /root/autodl-tmp/audit/audit_20260916_A [--json out.json]
退出码：0 = 全通过或仅 PENDING；1 = 有 FAIL
"""
import argparse
import json
import os
import re
import sys

VAL_POINTS = (0, 1000, 2000, 4000, 6000, 8000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--variant", required=True, choices=["T0", "T1", "U0"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    try:
        txt = open(a.log, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        print("FAIL  日志不存在：%s" % a.log)
        sys.exit(1)
    lines = txt.splitlines()
    done = "训练完成，head 已保存" in txt
    iters_seen = [int(m.group(1)) for m in re.finditer(r"iter=(\d+)", txt)]
    max_iter = max(iters_seen) if iters_seen else 0

    checks = []          # (name, status, detail)

    def logcheck(name, pat, detail_ok="", required=True):
        ok = re.search(pat, txt, re.M) is not None
        if ok:
            checks.append((name, "PASS", detail_ok))
        else:
            if not done and max_iter < a.iters:
                checks.append((name, "PENDING", "日志尚未结束且未跑到该点"))
            else:
                checks.append((name, "FAIL", "契约要求此输出，日志中不存在"))

    # --- 契约 §统一规格 / §8 可观测条目 ---
    logcheck("数据集规模打印", r"^\[%s\] train=\d+ val=\d+" % a.variant,
             "train=454 val=76")
    logcheck("warmup 峰值 lr=1e-3", r"iter=\d+\s+lr=1\.00e-03")

    # val 点（契约明文 0/1000/2000/4000/6000/8000）—— D-006 漏掉的就是这一族
    for p in [q for q in VAL_POINTS if q <= a.iters]:
        found = re.search(r"^\[val\] iter=\s*%d\s" % p, txt, re.M)
        if found:
            m = re.search(r"^\[val\] iter=\s*%d\s+base0_MA_IoU=([\d.]+) head_MA_IoU=([\d.]+)" % p,
                          txt, re.M)
            checks.append(("val@%d 点" % p, "PASS",
                           ("base0=%.6f head=%.6f" % (float(m.group(1)), float(m.group(2)))
                            if m else "存在")))
        else:
            # 只有「日志已结束」或「iter 已越过该点」才能判 FAIL；
            # 否则是训练行先打印、val 正在跑（val 比 iter 行晚约 2.5 min）的正常竞态。
            if done or max_iter > p:
                checks.append(("val@%d 点" % p, "FAIL",
                               "无 val 记录（日志%s）" % ("已结束" if done else "已跑到 iter=%d" % max_iter)))
            else:
                checks.append(("val@%d 点" % p, "PENDING", "尚未跑到该点"))

    if a.variant == "U0":
        logcheck("U0 参数量证据", r"\[U0\] 参数量=\d+")
        logcheck("U0 结构证据 shapes", r"\[U0\] 结构证据 shapes")
    logcheck("采样来源计数", r"采样来源计数:")
    logcheck("全 ignore 跳过步计数", r"全 ignore 跳过步 = \d+ / \d+")
    logcheck("中心区无 MA 阳性计数", r"中心区无 MA 阳性的 batch = \d+ / \d+")
    logcheck("终末 lr=1e-5", r"iter=%d\s+lr=1\.00e-05" % a.iters,
             "契约要求的 LR 终点")

    # --- 产物存在性 ---
    if a.out_dir:
        arts = [("head_%s_s%d.pth" % (a.variant, a.seed)),
                ("val_curve_%s_s%d.json" % (a.variant, a.seed)),
                ("val_counts_%s_s%d_it%d.json" % (a.variant, a.seed, a.iters))]
        if a.seed == 1:
            arts.append("head_%s.pth" % a.variant)
        for f in arts:
            p = os.path.join(a.out_dir, f)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                checks.append(("产物 %s" % f, "PASS", "%d 字节" % os.path.getsize(p)))
            else:
                checks.append(("产物 %s" % f, "PENDING" if not done else "FAIL",
                               "不存在或为空"))

    nf = sum(1 for c in checks if c[1] == "FAIL")
    npd = sum(1 for c in checks if c[1] == "PENDING")
    nps = sum(1 for c in checks if c[1] == "PASS")
    print("=" * 78)
    print("契约对账：%s  (variant=%s seed=%d iters=%d, 日志%s)"
          % (a.log, a.variant, a.seed, a.iters, "已结束" if done else "进行中 max_iter=%d" % max_iter))
    print("=" * 78)
    for name, st, det in checks:
        mark = {"PASS": " ok ", "FAIL": "FAIL", "PENDING": " .. "}[st]
        print("[%s] %-28s %s" % (mark, name, det))
    print("-" * 78)
    print("通过 %d / 失败 %d / 待跑 %d   -> %s"
          % (nps, nf, npd, "REJECT（存在契约条目缺失）" if nf else
             ("INCOMPLETE" if npd else "ACCEPT")))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"log": a.log, "variant": a.variant, "seed": a.seed,
                       "iters": a.iters, "complete": done, "max_iter": max_iter,
                       "pass": nps, "fail": nf, "pending": npd,
                       "verdict": "REJECT" if nf else ("INCOMPLETE" if npd else "ACCEPT"),
                       "checks": [{"item": n, "status": s, "detail": d} for n, s, d in checks]},
                      f, indent=1, ensure_ascii=False)
    sys.exit(1 if nf else 0)


if __name__ == "__main__":
    main()
