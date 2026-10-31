# -*- coding: utf-8 -*-
"""trainer.engine — H0/H1/U0/U1 共享的训练循环（唯一训练状态机）。

硬性语义（每一条都有对应的 smoke 用例，见 tests/run_training_smoke.py）：

1. **梯度累积以 optimizer update 计数**：每 `grad_accum` 个 micro-step 才
   `optimizer.step()` 一次；`update` 只在真正 step 后 +1。累积窗口跨 epoch
   连续（不因 epoch 边界丢弃半窗梯度，也不额外补一次 step）。
2. **scheduler 也按 update**：LR 乘子由 `trainer.optim.lr_multiplier(update)`
   在每次 step 前计算，`scheduler_state.update` 与 `update` 恒等；
   断点恢复后 LR 轨迹与不中断跑完全一致。
3. **AMP**：`mixed_precision=True` 时用 `torch.cuda.amp.GradScaler`
   （CPU 上自动退化为无 scaler 的 fp32 路径，绝不假装有 AMP）。
   裁剪在 `scaler.unscale_` 之后执行，顺序与 PyTorch 官方推荐一致。
4. **BN 策略**：每次 `model.train()` 之后立刻 `freeze_bn_stats(backbone)`，
   让 HRNet 主干统计冻结而 affine 仍可训练；ConvNeXt 臂无 BN，返回 0。
5. **验证**：只接受 `trainer.eval.WholeImageEvaluator`（整图/滑窗还原后累计），
   formal 模式缺 evaluator 直接 fail closed（禁止 crop 指标冒充 val）。
6. **checkpoint**：模型/优化器/scheduler/scaler/update/micro_step/seed/
   config SHA256/数据 manifest SHA256/RNG 全量落盘，原子写；
   恢复时校验 arm/stage/两个 SHA256，校准产物不得作为正式训练恢复源。
7. **日志与非有限 loss**：每遇到非有限（NaN/Inf）loss 立即 GateError（在
   backward 之前，绝不把坏梯度带进优化器/checkpoint）；`log_every` 节点除 LR
   外还记录最近 `log_every` 个 update 的平均总 loss 与各标量 loss parts
   （tensor parts 安全 detach→float，仅日志语义，不改变 loss/梯度/随机流）。
"""

from __future__ import annotations

import math
import os
import json
import time
from collections import deque

import torch

from .checkpoint import (
    build_checkpoint_payload,
    load_checkpoint,
    resume_state,
    save_checkpoint,
    verify_resume_compatibility,
)
from .eval import require_whole_image_evaluator
from .gate import GateError
from .metrics import append_metrics_csv, save_metrics_json
from .optim import (
    apply_lr,
    audit_param_groups,
    base_lrs_of,
    clip_gradients,
    freeze_bn_stats,
    lr_multiplier,
    scheduler_state,
)
from .paths import ensure_parent_dir


def _make_grad_scaler(enabled: bool):
    """构造 GradScaler（兼容 torch 新旧 API；新栈避免 FutureWarning）。"""
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


class TrainState:
    """训练状态计数（与 checkpoint 一一对应，不允许有第二份真相）。"""

    def __init__(self, *, update: int = 0, micro_step: int = 0, epoch: int = 0):
        self.update = int(update)
        self.micro_step = int(micro_step)
        self.epoch = int(epoch)

    def as_dict(self) -> dict:
        return {"update": self.update, "micro_step": self.micro_step, "epoch": self.epoch}


class StageRunner:
    """单阶段训练器（external_pretrain 与 ddr_finetune 共用同一实现）。"""

    def __init__(
        self,
        *,
        model,
        dataset,
        loss_fn,
        optimizer,
        cfg,                     # trainer.config 解析出的 EffectiveTrainConfig（dict 形式）
        device,
        evaluator=None,
        val_dataset=None,
        loader=None,
        scaler=None,
        logger=print,
    ):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.cfg = dict(cfg)
        self.device = device
        self.evaluator = evaluator
        self.val_dataset = val_dataset
        self.logger = logger
        self.state = TrainState()

        self.param_audit = audit_param_groups(model, optimizer)
        self.base_lrs = base_lrs_of(optimizer.param_groups)
        self.backbone_prefixes = tuple(self.cfg.get("backbone_prefixes") or ())
        self.backbone_norm_eval = bool(self.cfg.get("backbone_norm_eval", False))

        amp_on = bool(self.cfg.get("mixed_precision")) and self._cuda()
        if scaler is not None:
            self.scaler = scaler
        else:
            self.scaler = _make_grad_scaler(amp_on)
        self.amp_enabled = amp_on

        self.loader = loader if loader is not None else self._build_loader()
        self.history = []            # [(update, {group: lr})]，用于 LR 轨迹断言
        self.train_log = []          # [(micro_step, update, {loss parts})]
        self.best = {"mIoU_lesion4": float("-inf"), "update": None}
        self.eval_reports = []
        self.started_at = time.monotonic()
        self.stop_reason = None

        self.grad_accum = int(self.cfg["grad_accum"])
        self.total_updates = int(self.cfg["total_updates"])
        if self.grad_accum <= 0 or self.total_updates <= 0:
            raise GateError(
                f"grad_accum/total_updates 必须为正: {self.grad_accum}/{self.total_updates}"
            )

        # log_every 节点的 loss 聚合缓冲（仅日志语义）：环形保留最近 log_every
        # 个 optimizer update 的窗口摘要；log_every=0（关闭）时不缓冲不增长。
        self.log_every = int(self.cfg.get("log_every") or 0)
        self._loss_window = deque(maxlen=self.log_every) if self.log_every > 0 else None

    # ------------------------------------------------------------------
    def _cuda(self) -> bool:
        return torch.device(self.device).type == "cuda"

    def _build_loader(self):
        sampler = None
        if self.cfg.get('source_mixing') == 'source_balanced_sampling':
            from .data.external_dataset import source_balanced_weights
            weights = source_balanced_weights(self.dataset.records)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True)
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.cfg["micro_batch"]),
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=int(self.cfg.get("num_workers", 0)),
            pin_memory=self._cuda(),
        )

    def _infinite_batches(self):
        """无限 micro-batch 流：epoch 边界不打断累积窗口（无半窗丢弃/补步）。

        `set_epoch` 必须在 **创建新迭代器之前** 调用，这样 num_workers>0 时
        新 worker 进程才会拿到推进后的 epoch（增强随机流因此逐 epoch 变化）。
        """
        if len(self.loader) == 0:
            raise GateError(
                f"DataLoader 为空（dataset={len(self.dataset)} 样本，"
                f"micro_batch={self.cfg['micro_batch']}，drop_last=True）"
            )
        self.dataset.set_epoch(self.state.epoch)
        while True:
            for batch in self.loader:
                yield batch
            self.state.epoch += 1
            self.dataset.set_epoch(self.state.epoch)

    def _set_train_mode(self) -> int:
        """train() 后立即冻结 backbone BN 统计；返回冻结的 BN 模块数。"""
        self.model.train()
        if self.backbone_norm_eval:
            return freeze_bn_stats(self.model, backbone_prefixes=self.backbone_prefixes)
        return 0

    # ------------------------------------------------------------------
    def _forward_loss(self, batch) -> tuple:
        image = batch["image"].to(self.device, non_blocking=True).float()
        target = batch["target"].to(self.device, non_blocking=True).float()
        valid = batch["valid"].to(self.device, non_blocking=True).float()
        if self.amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(image)
                loss, parts = self.loss_fn(logits, target, valid)
        else:
            logits = self.model(image)
            loss, parts = self.loss_fn(logits, target, valid)
        if not torch.isfinite(loss).all():
            raise GateError('Nonfinite training loss: engineering failure, not a negative experiment')
        return loss, parts

    def _optimizer_step(self) -> dict:
        """执行一次 optimizer update（含 LR 计算、裁剪、scaler），update += 1。"""
        mult = lr_multiplier(
            self.state.update,
            total_updates=self.total_updates,
            warmup_updates=int(self.cfg["warmup_updates"]),
            cosine_min_ratio=float(self.cfg["cosine_min_ratio"]),
        )
        applied = apply_lr(self.optimizer, self.base_lrs, mult)
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        grad_norm = clip_gradients(self.model, self.cfg.get("grad_clip"))
        if self.scaler is not None:
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < old_scale:
                self.optimizer.zero_grad(set_to_none=True)
                self.amp_skips = getattr(self, 'amp_skips', 0) + 1
                if self.amp_skips > 20:
                    raise GateError('Repeated AMP overflow: calibration requires repair')
                return {'skipped': True}
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.update += 1
        self.history.append((self.state.update, dict(applied)))
        return {"lr_mult": mult, "lr": applied, "grad_norm": grad_norm, 'skipped': False}

    # ------------------------------------------------------------------
    def train_one_update_window(self, batch_iter):
        """跑满一个累积窗口（grad_accum 个 micro-step）并执行一次 update。

        返回 (did_step, last_parts)；batch_iter 耗尽时返回 (False, None)。
        """
        parts_acc = []
        for _ in range(self.grad_accum):
            try:
                batch = next(batch_iter)
            except StopIteration:
                if not parts_acc:
                    return False, None
                break
            self.state.micro_step += 1
            loss, parts = self._forward_loss(batch)
            # 累积窗口内每步除以 grad_accum ⇒ 一个 update 的梯度等于
            # effective_batch 上的平均损失（与"不累积直接大 batch"等价）。
            scaled_loss = loss / float(self.grad_accum)
            if self.scaler is not None:
                # AMP must scale every accumulated micro-loss before backward;
                # otherwise GradScaler has no scale/growth state and the first
                # unscale_() at optimizer step fails.
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            parts_acc.append(parts)
        if not parts_acc:
            return False, None
        step = self._optimizer_step()
        if step.get('skipped'):
            return self.train_one_update_window(batch_iter)
        if self._loss_window is not None:
            self._loss_window.append(float(parts_acc[-1].get('loss_total', 0)))
        return True, parts_acc[-1]

    # ------------------------------------------------------------------
    def run(self, *, resume_from: str | None = None, allow_config_mismatch: bool = False,
            allow_manifest_mismatch: bool = False, max_updates: int | None = None) -> dict:
        total = self.total_updates if max_updates is None else min(self.total_updates, int(max_updates))
        warnings = []
        if resume_from:
            warnings = self.resume(resume_from, allow_config_mismatch=allow_config_mismatch,
                                  allow_manifest_mismatch=allow_manifest_mismatch)
        require_whole_image_evaluator(self.evaluator, mode=self.cfg.get("mode", "calibration"))

        self.optimizer.zero_grad(set_to_none=True)
        frozen_bn = self._set_train_mode()
        self.logger(
            f"[engine] stage={self.cfg['stage']} mode={self.cfg['mode']} "
            f"updates {self.state.update}->{total} grad_accum={self.grad_accum} "
            f"micro_batch={self.cfg['micro_batch']} amp={self.amp_enabled} "
            f"frozen_bn={frozen_bn} params={self.param_audit['n_trainable_tensors']}"
        )

        eval_points = sorted(float(p) for p in (self.cfg.get("eval_progress_points") or [1.0]))
        next_eval = next((p for p in eval_points if p * total > self.state.update), None)

        stream = self._infinite_batches()
        while self.state.update < total:
            did, _ = self.train_one_update_window(stream)
            if not did:
                raise GateError("micro-batch 流意外结束（数据加载失败），拒绝静默结束训练")
            upd = self.state.update
            if self.cfg.get("log_every") and upd % int(self.cfg["log_every"]) == 0:
                avg_loss = sum(self._loss_window)/max(1,len(self._loss_window)) if self._loss_window else None
                self.logger(f"[engine] update {upd}/{total} loss={avg_loss} lr={self.history[-1][1]}")

            if self.cfg.get('periodic_save_updates') and upd % self.cfg['periodic_save_updates'] == 0:
                self.save(metrics_at_save={}, tag='last')
            stop_file = os.path.join(self.cfg.get('out_dir') or '.', 'STOP')
            if os.path.exists(stop_file):
                self.stop_reason = 'operator_stop_at_update_boundary'
                self.save(metrics_at_save={}, tag='last')
                break

            if next_eval is not None and upd >= next_eval * total:
                report = self.evaluate(update=upd)
                if report["mIoU_lesion4"] > self.best["mIoU_lesion4"]:
                    self.best = {"mIoU_lesion4": report["mIoU_lesion4"], "update": upd}
                    self.save(metrics_at_save=report, tag="best")
                self.save(metrics_at_save=report, tag="last")
                self._set_train_mode()
                next_eval = next((p for p in eval_points if p * total > upd), None)
                from .screening import decide_stop
                decision = decide_stop(self.eval_reports, total, self.cfg.get('screening') or {})
                if decision['stop']:
                    self.stop_reason = decision['reason']
                    self.logger(f'[engine] STOP: {self.stop_reason}')
                    break

        self.save(metrics_at_save={"mIoU_lesion4": self.best["mIoU_lesion4"]}, tag="final")
        result = {
            'status': 'stopped' if self.stop_reason else 'complete',
            'stop_reason': self.stop_reason,
            'elapsed_seconds': time.monotonic()-self.started_at,
            "stage": self.cfg["stage"],
            "mode": self.cfg["mode"],
            "updates_done": self.state.update,
            "micro_steps": self.state.micro_step,
            "best": self.best,
            "resume_warnings": warnings,
            "checkpoints": self.checkpoint_paths(),
        }
        if self.cfg.get('out_dir'):
            save_metrics_json(result, os.path.join(self.cfg['out_dir'], 'completion_summary.json'))
        return result

    def dry_run_check(self, batch=None):
        """校准/dry-run 用：一次真实前向 + 反向，证明形状、loss 与反传通路可用。

        不做 optimizer.step()、不写 checkpoint —— dry-run 不得产出可被误用的产物。
        """
        if batch is None:
            batch = next(iter(self.loader))
        self._set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        loss, parts = self._forward_loss(batch)
        loss.backward()
        n_grad = sum(1 for p in self.model.parameters() if p.requires_grad and p.grad is not None)
        self.optimizer.zero_grad(set_to_none=True)
        return {"loss": float(loss.detach()), "parts": parts,
                "tensors_with_grad": n_grad, "amp": self.amp_enabled}

    # ------------------------------------------------------------------
    def evaluate(self, *, update: int) -> dict:
        require_whole_image_evaluator(self.evaluator, mode=self.cfg.get("mode", "calibration"))
        report = self.evaluator.run()
        report.update({"update": update, "stage": self.cfg["stage"], "mode": self.cfg["mode"]})
        self.eval_reports.append(report)
        out_dir = self.cfg.get("out_dir")
        if out_dir:
            save_metrics_json(report, os.path.join(out_dir, f"eval_u{update}.json"))
            append_metrics_csv(report, os.path.join(out_dir, "eval_history.csv"))
        self.logger(
            f"[engine] eval update={update} mIoU_lesion4={report['mIoU_lesion4']:.4f} "
            f"n_images={report['n_images']}"
        )
        return report

    # ------------------------------------------------------------------
    def checkpoint_paths(self) -> dict:
        out_dir = self.cfg.get("out_dir")
        if not out_dir:
            return {}
        return {t: os.path.join(out_dir, f"ckpt_{t}.pth") for t in ("last", "best", "final")}

    def save(self, *, metrics_at_save, tag: str) -> str:
        out_dir = self.cfg.get("out_dir")
        if not out_dir:
            raise GateError("cfg.out_dir 未设置：拒绝在没有明确落盘目录时保存 checkpoint")
        payload = build_checkpoint_payload(
            model=self.model,
            optimizer=self.optimizer,
            scheduler_state=scheduler_state(
                self.state.update, self.base_lrs,
                total_updates=self.total_updates,
                warmup_updates=int(self.cfg["warmup_updates"]),
                cosine_min_ratio=float(self.cfg["cosine_min_ratio"]),
            ),
            scaler=self.scaler,
            update=self.state.update,
            micro_step=self.state.micro_step,
            total_updates=self.total_updates,
            seed=int(self.cfg["seed"]),
            config_sha256=self.cfg["config_sha256"],
            data_manifest_sha256=self.cfg.get("data_manifest_sha256"),
            manifest_path=self.cfg.get("manifest_path"),
            arm_id=self.cfg["arm_id"],
            family=self.cfg["family"],
            stage=self.cfg["stage"],
            mode=self.cfg["mode"],
            is_calibration=bool(self.cfg.get("is_calibration")),
            metrics_at_save=metrics_at_save,
            extra={"tag": tag, "best": dict(self.best), 'epoch': self.state.epoch},
        )
        path = save_checkpoint(os.path.join(out_dir, f"ckpt_{tag}.pth"), payload)
        return path

    def resume(self, path: str, *, allow_config_mismatch=False, allow_manifest_mismatch=False) -> list:
        ckpt = load_checkpoint(path)
        warnings = verify_resume_compatibility(
            ckpt,
            arm_id=self.cfg["arm_id"],
            stage=self.cfg["stage"],
            config_sha256=self.cfg["config_sha256"],
            data_manifest_sha256=self.cfg.get("data_manifest_sha256"),
            allow_config_mismatch=allow_config_mismatch,
            allow_manifest_mismatch=allow_manifest_mismatch,
            allow_calibration_checkpoint=bool(self.cfg.get("is_calibration")),
        )
        restored = resume_state(ckpt, model=self.model, optimizer=self.optimizer, scaler=self.scaler)
        # update 直接采用 checkpoint 值（绝不重放/累加）⇒ 断点后计数连续。
        self.state.update = restored["update"]
        self.state.micro_step = restored["micro_step"]
        self.state.epoch = int(ckpt.get("epoch", 0))
        self.best = dict(ckpt.get("best") or self.best)
        for w in warnings:
            self.logger(f"[engine] resume warning: {w}")
        return warnings


def build_effective_config(
    *,
    cfg: dict,
    stage: str,
    mode: str,
    stage_cfg: dict,
    batch: dict,
    seed: int,
    total_updates: int,
    backbone_prefixes,
    backbone_norm_eval: bool,
    config_sha256_value: str,
    data_manifest_sha256,
    manifest_path,
    out_dir: str | None,
    extra: dict | None = None,
) -> dict:
    """把配置 + CLI 解析结果归一成 StageRunner 需要的扁平字典（唯一真相）。"""
    eff = {
        "arm_id": cfg["arm_id"],
        "family": cfg["structure"]["family"],
        "stage": stage,
        "mode": mode,
        "is_calibration": mode == "calibration",
        "seed": int(seed),
        "total_updates": int(total_updates),
        "micro_batch": int(batch["micro_batch"]),
        "grad_accum": int(batch["grad_accum"]),
        "effective_batch": int(batch["effective_batch"]),
        "warmup_updates": int(round(float(stage_cfg.get("warmup_ratio", 0.0)) * int(total_updates))),
        "cosine_min_ratio": float(stage_cfg.get("cosine_min_ratio", 0.01)),
        "grad_clip": stage_cfg.get("grad_clip"),
        "mixed_precision": bool(stage_cfg.get("mixed_precision", False)),
        "eval_progress_points": list(stage_cfg.get("eval_progress_points") or [1.0]),
        "log_every": int(stage_cfg.get("log_every", 0) or 0),
        "num_workers": int(stage_cfg.get("num_workers", 0) or 0),
        'source_mixing': stage_cfg.get('source_mixing'),
        'screening': stage_cfg.get('screening') or {},
        'periodic_save_updates': int(stage_cfg.get('periodic_save_updates', 0)),
        "backbone_prefixes": tuple(backbone_prefixes),
        "backbone_norm_eval": bool(backbone_norm_eval),
        "config_sha256": config_sha256_value,
        "data_manifest_sha256": data_manifest_sha256,
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "out_dir": None if out_dir is None else ensure_parent_dir(os.path.join(out_dir, "placeholder.pth")),
    }
    if eff["out_dir"] is not None:
        eff["out_dir"] = os.path.dirname(eff["out_dir"])
    if extra:
        eff.update(extra)
    return eff
