#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 2 号机上给 repo 的 mmseg/apis/train.py 打最小补丁：把 cfg.custom_hooks 注册进 runner。

为什么需要：本仓库的 train_segmentor 是定制版，**全文没有 custom_hooks 字样**
（上游 mmseg 有这段逻辑，这里被删了）→ 我们的 FreezeScopeHook 永远不会被注册
（已由 smoke 实证：日志的 hook 列表里没有它、证据文件未生成）。
补丁内容 = 把上游 mmseg 那 5 行原样加回，只在 register_training_hooks 之后。
幂等：带 J0J1_PATCH 标记，重复执行不会叠加。
"""
import io
import re
import sys

P = "/root/box1_sync/repo/mmseg/apis/train.py"
MARK = "J0J1_PATCH"

s = io.open(P, encoding="utf-8").read()
if MARK in s:
    print("已打过补丁，跳过")
    sys.exit(0)

lines = s.split("\n")
start = None
for i, ln in enumerate(lines):
    if "runner.register_training_hooks(" in ln:
        start = i
        break
assert start is not None, "找不到 register_training_hooks 调用点"

# 找到这次调用的闭合行（以 ')' 结尾的那一行）
end = None
for j in range(start, min(start + 12, len(lines))):
    if lines[j].rstrip().endswith(")"):
        end = j
        break
assert end is not None, "找不到 register_training_hooks 调用的闭合行"

ins = [
    "",
    "    # ---- %s begin（补回上游 mmseg 的 custom_hooks 支持，本仓库定制版删掉了）----" % MARK,
    "    _custom_hooks = cfg.get('custom_hooks', None)",
    "    if _custom_hooks is None:",
    "        _custom_hooks = []",
    "    for _hook_cfg in _custom_hooks:",
    "        runner.register_hook_from_cfg(_hook_cfg)",
    "    # ---- %s end ----" % MARK,
]
lines[end + 1:end + 1] = ins
io.open(P, "w", encoding="utf-8").write("\n".join(lines))
print("补丁已写入，上下文：")
print("\n".join(lines[max(0, start - 1):end + len(ins) + 2]))
