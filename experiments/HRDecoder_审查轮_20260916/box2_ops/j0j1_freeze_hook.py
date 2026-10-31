#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""J0/J1 冻结范围 hook（把 j0j1_preflight.py 已验证的接口搬进真实训练）。

放置位置：repo 根目录，供 config 以
`custom_imports=dict(imports=['hrplus_patch','j0j1_freeze_hook'])` 导入。

接口约定（与 j0j1_preflight.py 一致，不重新发明）：
  * 冻结全部 params；J0 = 只解冻 decode_head；J1 = decode_head + HRNet 最后 stage（本仓库 = stage4）
  * 所有 BatchNorm 恒为 eval
  * 起点由 config 的 `load_from` 指定（预检用的那个 D checkpoint）

──────────────── 2026-09-17 子代理审计后的修复（3 处致命 + 2 处加固）────────────────
F1 [致命] 单卡 train.py 会用 MMDataParallel 包一层：runner.model.backbone 会 AttributeError，
        且 named_parameters() 带 `module.` 前缀导致 startswith 断言全假失败。
        → 所有方法开头用 mmcv.parallel.is_module_wrapper 解包。
F2 [致命] IterBasedRunner.train() 每个 iter 都调 model.train()，会把 before_run 里的 BN.eval() 撤销。
        （本模型 hr100_refinement.train() 自己会锁 BN，属双保险）→ 在 before_train_iter 里重设。
F3 [保命] after_train_iter 的钩子顺序由 priority 决定：OptimizerHook=ABOVE_NORMAL(40) 先跑，
        本 hook 必须 >= NORMAL(50)，否则 grad 已被 zero_grad 清空 → 断言假失败。
        → config 里显式 priority='NORMAL'（本文件默认即 NORMAL）。
F4 [加固] IterBasedRunner.run() 无 try/except：断言失败时 after_run 不执行 → 证据丢失。
        → 首步梯度证据当场落盘，不等 after_run。
F5 [加固] 增加 init 探针：student 必须与 reference 不同（证明 load_from 真的生效），
        并打印可核验的指纹。
"""
import hashlib
import json
import os
import re

import torch
import torch.nn as nn
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


def _unwrap(model):
    """F1：单卡训练时 runner.model = MMDataParallel，必须解包才能访问 .backbone。"""
    return model.module if is_module_wrapper(model) else model


def _locate_last_stage(backbone):
    """HRNet 的 stage 命名不统一（本仓库 = layer1 + stage2 + stage3 + stage4），
    显式枚举后按数字排序，不猜前缀（契约 §10.1）。"""
    top = sorted({n for n, _ in backbone.named_children()})
    ordered = sorted({n for n in top if re.fullmatch(r"(layer|stage)\d+", n)},
                     key=lambda x: int(re.search(r"\d+", x).group()))
    assert ordered, "未能定位 HRNet 的 stage 序列，需人工确认：%s" % top
    return ordered, ordered[-1]


def _bn_fp(model):
    """BN 运行统计（含 num_batches_tracked）的指纹；训练前后必须一致。"""
    h = hashlib.sha256()
    for n, b in sorted(model.named_buffers()):
        if ".bn" not in n and "norm" not in n and "batchnorm" not in n:
            continue
        h.update(n.encode())
        h.update(b.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _probe(mod, limit=3, n_el=4096):
    """F5：便宜的身份探针（前 limit 个浮点张量的前 n_el 个元素）。"""
    h = hashlib.sha256()
    c = 0
    for n, p in sorted(mod.named_parameters()):
        if not p.is_floating_point():
            continue
        flat = p.detach().cpu().reshape(-1)[:n_el]
        h.update(n.encode())
        h.update(flat.numpy().tobytes())
        c += 1
        if c >= limit:
            break
    return h.hexdigest()[:16]


@HOOKS.register_module()
class FreezeScopeHook(Hook):
    def __init__(self, arm="J0", expect_trainable=None, expect_extra=None):
        assert arm in ("J0", "J1"), arm
        self.arm = arm
        self.expect_trainable = expect_trainable
        self.expect_extra = expect_extra
        self.evidence = {"arm": arm}
        self._grad_logged = False

    # ---- 运行前：施加冻结范围 + 断言 + 证据 ----
    def before_run(self, runner):
        m = _unwrap(runner.model)
        self._m = m
        stages, last = _locate_last_stage(m.backbone)
        self._last = last
        self.evidence["hrnet_stage_sequence"] = stages
        self.evidence["j1_last_stage"] = last

        # F5 init 探针：student 与 reference 必须不同（否则 load_from 没生效）
        ps, pr = _probe(m), _probe(m.reference)
        self.evidence["init_probe"] = {"student": ps, "reference": pr, "differ": ps != pr}
        print("[FreezeScope %s] init 探针 student=%s reference=%s 不同=%s"
              % (self.arm, ps, pr, ps != pr), flush=True)

        for p in m.parameters():
            p.requires_grad_(False)
        for p in m.decode_head.parameters():
            p.requires_grad_(True)
        if self.arm == "J1":
            for nm, p in m.backbone.named_parameters():   # nm 相对 backbone，如 stage4.0...
                if nm.split(".")[0] == last:
                    p.requires_grad_(True)
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eval()

        names = sorted(n for n, p in m.named_parameters() if p.requires_grad)
        dec = [n for n in names if n.startswith("decode_head.")]
        bbn = [n for n in names if n.startswith("backbone.")]
        elems = sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.evidence.update({
            "trainable_tensors": len(names),
            "trainable_elems": int(elems),
            "decoder_tensors": len(dec),
            "backbone_tensors": len(bbn),
            "sample": names[:8],
            "bn_all_eval": all(not mod.training for mod in m.modules()
                               if isinstance(mod, nn.modules.batchnorm._BatchNorm)),
            "reference_frozen": all(not p.requires_grad for p in m.reference.parameters()),
        })
        print("[FreezeScope %s] trainable 张量=%d（decoder %d + backbone %d）元素=%d BN全eval=%s reference冻结=%s"
              % (self.arm, len(names), len(dec), len(bbn), elems,
                 self.evidence["bn_all_eval"], self.evidence["reference_frozen"]), flush=True)
        print("[FreezeScope %s] HRNet stages=%s 最后stage=%s" % (self.arm, stages, last), flush=True)

        if self.arm == "J0":
            assert not bbn, "J0 不应有 backbone 张量被解冻：%s" % bbn[:5]
        else:
            bad = [n for n in bbn if n.split(".")[1] != last]
            assert not bad, "J1 只允许解冻 %s，违规：%s" % (last, bad[:5])
            assert len(bbn) > 0, "J1 没有解冻到任何 backbone 张量"
        if self.expect_trainable is not None:
            assert len(names) == self.expect_trainable, (
                "trainable 张量数 %d != 预定 %d" % (len(names), self.expect_trainable))
        assert self.evidence["bn_all_eval"], "BN 未全部处于 eval"
        assert self.evidence["reference_frozen"], "reference 未被冻结"
        assert ps != pr, "student 与 reference 完全相同 —— load_from 可能没生效"
        self.evidence["bn_fp_before"] = _bn_fp(m)
        self._flush(runner)

    # ---- F2：每个 iter 开头重锁 BN（IterBasedRunner.train() 会撤销 before_run 的设置）----
    def before_train_iter(self, runner):
        m = _unwrap(runner.model)
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm) and mod.training:
                mod.eval()

    # ---- 首个训练步：梯度证据（F4：当场落盘）----
    def after_train_iter(self, runner):
        if self._grad_logged:
            return
        self._grad_logged = True
        m = _unwrap(runner.model)
        last = self._last

        def gsum(pred):
            s, n = 0.0, 0
            for nm, p in m.named_parameters():
                if pred(nm) and p.grad is not None:
                    s += float(p.grad.abs().sum())
                    n += 1
            return s, n

        g_dec, n_dec = gsum(lambda nm: nm.startswith("decode_head."))
        g_last, n_last = gsum(lambda nm: nm.startswith("backbone.") and nm.split(".")[1] == last)
        g_oth, n_oth = gsum(lambda nm: nm.startswith("backbone.") and nm.split(".")[1] != last)
        self.evidence["grad_first_step"] = {
            "decoder_grad_sum": g_dec, "decoder_tensors_with_grad": n_dec,
            "last_stage_grad_sum": g_last, "last_stage_tensors_with_grad": n_last,
            "other_backbone_grad_sum": g_oth, "other_backbone_tensors_with_grad": n_oth,
        }
        print("[FreezeScope %s] 首步梯度: decoder=%.4f(%d) %s=%.4f(%d) 其他backbone=%.6f(%d)"
              % (self.arm, g_dec, n_dec, last, g_last, n_last, g_oth, n_oth), flush=True)
        self._flush(runner)          # F4
        assert g_dec > 0, "decoder 应有非零梯度"
        assert g_oth == 0.0, "未授权 backbone 段出现梯度 %.6f" % g_oth
        if self.arm == "J1":
            assert g_last > 0, "J1 的 %s 未收到梯度 —— 反传没真正到 backbone" % last

    def after_run(self, runner):
        m = _unwrap(runner.model)
        after = _bn_fp(m)
        self.evidence["bn_fp_after"] = after
        self.evidence["bn_buffers_unchanged"] = (after == self.evidence.get("bn_fp_before"))
        self.evidence["finished"] = True
        self._flush(runner)
        print("[FreezeScope %s] 结束：BN buffers 不变=%s"
              % (self.arm, self.evidence["bn_buffers_unchanged"]), flush=True)

    def _flush(self, runner):
        wd = getattr(runner, "work_dir", None) or "."
        os.makedirs(wd, exist_ok=True)
        p = os.path.join(wd, "J0J1_evidence_%s.json" % self.arm)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.evidence, f, indent=1, ensure_ascii=False)
        os.replace(tmp, p)
