#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 train_T0T1.py 打补丁：实现契约要求的 val 点 + 逐图计数 + E1 续训支持。

补丁内容（不含任何会改变 seed=1 训练轨迹的改动）：
  1. 真正调用 val_eval()：val 点固定为 0/1000/2000/4000/6000/8000（按 --iters 截断）
     —— 修复 D-006「val_eval 只定义未调用，--val-every 未使用」
  2. 每个 val 点导出**逐图 TP/FP/FN**（`val_counts_{variant}_s{seed}_it{iter}.json`），
     供 E1 判定门做图像级配对 bootstrap（复用 R0 的冻结规则）
  3. checkpoint 名带 seed，杜绝多 seed 互相覆盖；seed==1 额外写一份无后缀别名保持契约名
  4. checkpoint 内存 `rng_state`，并新增 `--init` / `--fixed-lr`：
     E1 续训时恢复采样 RNG 状态 → 是**真续训**（不重放前 8000 步的采样序列）
  5. val_eval 采用 no_grad、不触碰任何 RNG → 训练轨迹与加 val 前逐位一致
"""
import io
import os
import sys

P = r"D:/dataset/experiment/HRDecoder_审查轮_20260916/tools/train_T0T1.py"
s = io.open(P, encoding="utf-8").read()
orig = s

def rep(old, new, tag):
    global s
    assert s.count(old) == 1, "锚点不唯一或缺失（%s）：count=%d" % (tag, s.count(old))
    s = s.replace(old, new, 1)
    print("  补丁 OK: %s" % tag)


# ---- 1) 新增命令行参数 ----
rep(
    '    ap.add_argument("--smoke", type=int, default=0, help=">0 时只跑前 N 步做排错")\n',
    '    ap.add_argument("--smoke", type=int, default=0, help=">0 时只跑前 N 步做排错")\n'
    '    ap.add_argument("--init", default="", help="从该 checkpoint 续训（E1）；会校验 variant 一致")\n'
    '    ap.add_argument("--fixed-lr", type=float, default=-1.0,\n'
    '                    help=">0 时全程用该固定学习率（E1 续训用），否则用 lr_at 时程")\n',
    "argparse --init/--fixed-lr",
)

# ---- 2) val_eval 记录逐图计数 ----
rep(
    '        agg = {"base0": [0, 0, 0], "head": [0, 0, 0]}\n',
    '        agg = {"base0": [0, 0, 0], "head": [0, 0, 0]}\n'
    '        per_img = []          # 逐图逐 tag 的 TP/FP/FN（MA 类），供配对 bootstrap\n',
    "per_img 初始化",
)

rep(
    '                gg = (g == 4); ig = (g == 255)\n'
    '                for tag, pr in (("base0", prob_b), ("head", prob_h)):\n'
    '                    p = pr > TAU\n'
    '                    agg[tag][0] += int((p & gg & ~ig).sum())\n'
    '                    agg[tag][1] += int((p & ~gg & ~ig).sum())\n'
    '                    agg[tag][2] += int((~p & gg).sum())\n',
    '                gg = (g == 4); ig = (g == 255)\n'
    '                row = {"img": int(vi), "H": int(H), "W": int(W)}\n'
    '                for tag, pr in (("base0", prob_b), ("head", prob_h)):\n'
    '                    p = pr > TAU\n'
    '                    tp = int((p & gg & ~ig).sum())\n'
    '                    fp = int((p & ~gg & ~ig).sum())\n'
    '                    fn = int((~p & gg).sum())\n'
    '                    agg[tag][0] += tp; agg[tag][1] += fp; agg[tag][2] += fn\n'
    '                    row[tag] = [tp, fp, fn]\n'
    '                per_img.append(row)\n',
    "逐图计数累积",
)

rep(
    '            out[k] = tp / den if den else 0.0\n'
    '        return out\n',
    '            out[k] = tp / den if den else 0.0\n'
    '        return out, per_img\n',
    "val_eval 返回值",
)

# ---- 3) VAL_POINTS + log_val（契约 val 点）----
rep(
    '    hist = []\n'
    '    skip_all_ignore = 0     # 中心区全为 ignore -> 按契约跳过更新\n',
    '    # 契约 val 点：0/1000/2000/4000/6000/8000（按 --iters 截断）\n'
    '    VAL_POINTS = tuple(p for p in (0, 1000, 2000, 4000, 6000, 8000) if p <= args.iters)\n'
    '\n'
    '    def log_val(k):\n'
    '        """跑一次 val 并把「聚合 IoU + 逐图计数」落盘。纯推理，不动 RNG。"""\n'
    '        vo, vper = val_eval()\n'
    '        hist.append({"iter": int(k), "base0_MA_IoU": vo["base0"],\n'
    '                     "head_MA_IoU": vo["head"],\n'
    '                     "d_MA_IoU": vo["head"] - vo["base0"]})\n'
    '        print("[val] iter=%-5d base0_MA_IoU=%.6f head_MA_IoU=%.6f d_MA_IoU=%+.6f"\n'
    '              % (k, vo["base0"], vo["head"], vo["head"] - vo["base0"]), flush=True)\n'
    '        jp = os.path.join(args.out_dir, "val_counts_%s_s%d_it%d.json"\n'
    '                          % (args.variant, args.seed, k))\n'
    '        with open(jp, "w") as f:\n'
    '            json.dump({"variant": args.variant, "seed": args.seed, "iter": int(k),\n'
    '                       "cls_index": 4, "tau": TAU, "n_val": int(n_va),\n'
    '                       "agg": {"base0": vo["base0"], "head": vo["head"]},\n'
    '                       "per_image": vper}, f)\n'
    '        print("[val] 逐图计数已写 %s" % jp, flush=True)\n'
    '        return vo\n'
    '\n'
    '    hist = []\n'
    '    skip_all_ignore = 0     # 中心区全为 ignore -> 按契约跳过更新\n',
    "VAL_POINTS + log_val",
)

# ---- 4) --init 续训 + iter 0 的 val ----
rep(
    '    rng = np.random.RandomState(args.seed)\n'
    '    t0 = time.time()\n',
    '    rng = np.random.RandomState(args.seed)\n'
    '    if args.init:\n'
    '        ck = torch.load(args.init, map_location="cpu", weights_only=False)\n'
    '        sd = ck["head"] if isinstance(ck, dict) and "head" in ck else ck\n'
    '        ckv = ck.get("variant") if isinstance(ck, dict) else None\n'
    '        assert ckv is None or ckv == args.variant, (\n'
    '            "续训臂不一致：checkpoint variant=%s 但 --variant=%s" % (ckv, args.variant))\n'
    '        head.load_state_dict(sd, strict=True)\n'
    '        n_on = sum(1 for v in sd.values()\n'
    '                   if torch.is_tensor(v) and v.is_floating_point() and float(v.abs().sum()) > 0)\n'
    '        print("[init] 续训自 %s：step=%s variant=%s 非零张量=%d"\n'
    '              % (args.init, ck.get("step") if isinstance(ck, dict) else None, ckv, n_on),\n'
    '              flush=True)\n'
    '        if isinstance(ck, dict) and ck.get("rng_state") is not None:\n'
    '            rng.set_state(ck["rng_state"])\n'
    '            print("[init] 采样 RNG 状态已恢复 -> 真续训（不重放前段采样序列）", flush=True)\n'
    '        else:\n'
    '            print("[init] !! checkpoint 缺 rng_state，采样序列将从起点重放", flush=True)\n'
    '    if 0 in VAL_POINTS:\n'
    '        log_val(0)          # 契约要求 iter=0 的 val 点（零初始化头，应与 base0 相等）\n'
    '    t0 = time.time()\n',
    "--init 续训 + val@0",
)

# ---- 5) 固定学习率（E1）----
rep(
    '        lr = lr_at(it)\n',
    '        lr = args.fixed_lr if args.fixed_lr > 0 else lr_at(it)\n',
    "fixed-lr",
)

# ---- 6) 全 ignore 跳过步也要出 val 点（未更新，状态等价）----
rep(
    '                print("  [skip] iter=%d 中心区全 ignore，本步不更新（累计 %d）"\n'
    '                      % (it, skip_all_ignore), flush=True)\n'
    '            continue\n',
    '                print("  [skip] iter=%d 中心区全 ignore，本步不更新（累计 %d）"\n'
    '                      % (it, skip_all_ignore), flush=True)\n'
    '            if it in VAL_POINTS:\n'
    '                log_val(it)     # 该步未更新，状态等价于 it-1 步后的头\n'
    '            continue\n',
    "skip 分支的 val",
)

# ---- 7) 循环内 val 点 ----
rep(
    '                  % (it, lr, loss, bce, dice, gnorm, time.time() - t0), flush=True)\n'
    '        if args.smoke and it >= args.smoke:\n',
    '                  % (it, lr, loss, bce, dice, gnorm, time.time() - t0), flush=True)\n'
    '        if it in VAL_POINTS:\n'
    '            log_val(it)\n'
    '        if args.smoke and it >= args.smoke:\n',
    "循环内 val 点",
)

# ---- 8) 归档：val 曲线 + 带 seed 的 checkpoint + rng_state ----
rep(
    '    torch.save({"head": head.state_dict(), "step": args.iters, "variant": args.variant},\n'
    '               os.path.join(args.out_dir, "head_%s.pth" % args.variant))\n'
    '    print("训练完成，head 已保存", flush=True)\n',
    '    # --- val 曲线归档 ---\n'
    '    cvp = os.path.join(args.out_dir, "val_curve_%s_s%d.json" % (args.variant, args.seed))\n'
    '    with open(cvp, "w") as f:\n'
    '        json.dump({"variant": args.variant, "seed": args.seed, "iters": args.iters,\n'
    '                   "fixed_lr": args.fixed_lr, "init": args.init,\n'
    '                   "val_points": list(VAL_POINTS), "curve": hist}, f, indent=1)\n'
    '\n'
    '    # --- checkpoint：名带 seed，另存 rng_state 以便 E1 真续训 ---\n'
    '    payload = {"head": head.state_dict(), "step": args.iters, "variant": args.variant,\n'
    '               "seed": args.seed, "fixed_lr": args.fixed_lr, "init": args.init,\n'
    '               "rng_state": rng.get_state(), "val_curve": hist}\n'
    '    ckp = os.path.join(args.out_dir, "head_%s_s%d.pth" % (args.variant, args.seed))\n'
    '    torch.save(payload, ckp)\n'
    '    if args.seed == 1:\n'
    '        torch.save(payload, os.path.join(args.out_dir, "head_%s.pth" % args.variant))\n'
    '    print("训练完成，head 已保存: %s（seed==1 另存无后缀别名）" % ckp, flush=True)\n'
    '    if hist:\n'
    '        print("val 曲线: " + " | ".join(\n'
    '            "it%d:%.4f" % (h["iter"], h["head_MA_IoU"]) for h in hist), flush=True)\n',
    "归档 val 曲线 + checkpoint",
)

# ---- 9) 文档字符串补一句 ----
rep(
    'T1 采样：每 batch 2 均匀 + 1 MA 阳性锚定 + 1 train BASE_0 MA 误报锚定。\n"""\n',
    'T1 采样：每 batch 2 均匀 + 1 MA 阳性锚定 + 1 train BASE_0 MA 误报锚定。\n'
    '\n'
    '（2026-09-17 补丁）真正调用 val_eval：val 点 0/1000/2000/4000/6000/8000，\n'
    '每个点导出逐图 TP/FP/FN；checkpoint 名带 seed 并存 rng_state；\n'
    '新增 --init/--fixed-lr 供 E1 配对续训。\n'
    '"""\n',
    "docstring",
)

assert s != orig
io.open(P, "w", encoding="utf-8").write(s)
print("已写入 %s（%d -> %d 字节）" % (P, len(orig), len(s)))
