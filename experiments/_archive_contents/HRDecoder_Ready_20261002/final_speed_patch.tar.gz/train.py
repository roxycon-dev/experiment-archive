#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train.py — H0/H1/U0/U1 的**唯一**训练入口（上传服务器的独立入口）。

用法（两个阶段共用同一入口与同一份共享核心）：

    # 阶段一：外部病灶预训练（仅 H1/U1）
    python train.py --config configs/H1_hrdecoder_ext_pretrain.yaml \
        --stage external_pretrain --mode calibration

    # 阶段二：DDR 适配（四臂）
    python train.py --config configs/H0_hrdecoder_ddr_only.yaml \
        --stage ddr_finetune --mode formal

    # 断点恢复
    python train.py --config ... --stage ddr_finetune --resume work_dirs/H0/ckpt_last.pth

硬性闸门（任一不满足即非零退出，绝不降级放行）：

* `--mode formal`：配置里**不允许存在任何** REQUIRED/TBD 占位符
  （readiness 与 calibration 两层都要求清空）。正式训练不接受"跑起来再说"。
* `--mode calibration`：只要求 readiness 层清空；calibration 层的
  micro_batch / grad_accum / gradient_checkpointing 可由 CLI 覆盖，
  产出的 checkpoint 标记 `is_calibration=true`，**不得**用于正式训练恢复。
* train/val split 只能是 train/val；test 一律硬拒绝（`--train-split`/`--val-split`
  与配置字段都会被检查）。checkpoint 选择只用 val。
* formal 模式必须提供整图验证 evaluator（trainer.eval.WholeImageEvaluator）；
  禁止用训练 crop 的指标冒充 val。
* 模型依赖不可用时以 EXIT_DEPENDENCY_BLOCKED(3) 退出，**不使用假模型**。

退出码：0=成功；1=契约/闸门失败；2=用法或 IO 致命错误；3=依赖阻塞（DEPENDENCY_BLOCKED）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from trainer.config import (  # noqa: E402
    CALIBRATION_OVERRIDES,
    apply_calibration_overrides,
    assert_no_test_split,
    checkpoint_selection_splits,
    config_digest,
    ddr_data_config,
    effective_batch_check,
    gate_config,
    load_yaml,
    metrics_config,
    resolve_stage_section,
    validate_config,
)
from trainer.gate import GateError, sha256_text  # noqa: E402
from trainer.models import DependencyBlocked  # noqa: E402
from trainer.paths import ensure_dir  # noqa: E402
from trainer.data.common import ContractError  # noqa: E402

EXIT_OK = 0
EXIT_CONTRACT = 1
EXIT_USAGE_IO = 2
EXIT_DEPENDENCY_BLOCKED = 3

# CLI 允许覆盖的字段前缀（校准层 + 数据/资产路径）。
# 明确**不允许**覆盖 loss / metrics / checkpoint_selection / test_usage /
# channel_order 等科学语义字段 —— 那些只能改配置文件并重跑 preflight。
OVERRIDE_ALLOWED_PREFIXES = (
    "data.ddr_root",
    "data.ddr_normalization",
    "data.train_crop",
    "structure.reference_root",
    "structure.library_version",
    "structure.encoder_weights_file",
    "structure.encoder_weights_sha256",
    "structure.encoder_weights_source",
    "structure.training_entry",
    "initialization.imagenet_weights",
    "initialization.external_pretrain_checkpoint",
    "pretrain_stage.external_manifest",
    "pretrain_stage.manifest_base_dir",
    "pretrain_stage.init_checkpoint_output",
    "pretrain_stage.overfit_monitor",
    "pretrain_stage.micro_batch",
    "pretrain_stage.grad_accum",
    "pretrain_stage.gradient_checkpointing",
    "pretrain_stage.seed",
    "pretrain_stage.updates",
    "pretrain_stage.dev_ratio",
    "ddr_finetune_stage.micro_batch",
    "ddr_finetune_stage.grad_accum",
    "ddr_finetune_stage.gradient_checkpointing",
    "ddr_finetune_stage.seed",
    "ddr_finetune_stage.updates",
    "ddr_finetune_stage.num_workers",
    "ddr_finetune_stage.log_every",
    "ddr_finetune_stage.eval_progress_points",
    "metrics.eval_mode",
    "metrics.window_size",
    "metrics.window_stride",
    "runtime.out_dir",
    "runtime.eval_max_images",
)


def _parse_override(text: str):
    if "=" not in text:
        raise GateError(f"--override 需要 key=value 形式，收到 {text!r}")
    key, raw = text.split("=", 1)
    key = key.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def _override_allowed(key: str) -> bool:
    if key in CALIBRATION_OVERRIDES_FULL:
        return True
    return any(key == p or key.startswith(p + ".") for p in OVERRIDE_ALLOWED_PREFIXES)


CALIBRATION_OVERRIDES_FULL = tuple(f"ddr_finetune_stage.{k}" for k in CALIBRATION_OVERRIDES) + tuple(
    f"pretrain_stage.{k}" for k in CALIBRATION_OVERRIDES
)


def apply_overrides(cfg: dict, overrides: list) -> tuple:
    applied = {}
    for text in overrides or []:
        key, value = _parse_override(text)
        if not _override_allowed(key):
            raise GateError(
                f"--override 不允许作用于 {key!r}（只允许校准层与数据/资产路径字段；"
                "loss/metrics/checkpoint_selection/test_usage/channel_order 等"
                "科学语义字段必须改配置文件并重跑 preflight）"
            )
        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            if not isinstance(cur.get(p), dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
        applied[key] = value
    return cfg, applied


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="H0/H1/U0/U1 统一训练入口（fail closed；REQUIRED/TBD 未清空不得正式开训）"
    )
    ap.add_argument("--config", required=True, help="单臂配置 YAML")
    ap.add_argument("--stage", required=True, choices=("external_pretrain", "ddr_finetune"))
    ap.add_argument("--mode", default="formal", choices=("formal", "calibration"),
                    help="formal=正式训练（禁止任何占位符）；calibration=短时 GPU 校准")
    ap.add_argument("--out-dir", default=None, help="输出目录（ckpt/指标/日志）")
    ap.add_argument("--resume", default=None, help="断点恢复 checkpoint 路径")
    ap.add_argument("--micro-batch", type=int, default=None, help="校准覆盖（仅 calibration）")
    ap.add_argument("--grad-accum", type=int, default=None, help="校准覆盖（仅 calibration）")
    ap.add_argument("--gradient-checkpointing", default=None,
                    choices=("true", "false"), help="校准覆盖（仅 calibration）")
    ap.add_argument("--train-split", default=None, help="覆盖训练 split（test 会被硬拒绝）")
    ap.add_argument("--val-split", default=None, help="覆盖验证/选点 split（test 会被硬拒绝）")
    ap.add_argument("--max-updates", type=int, default=None, help="只跑前 N 次 update（校准用）")
    ap.add_argument("--eval-max-images", type=int, default=None, help="验证只用前 N 张（校准用）")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="覆盖允许清单内的配置字段（可重复）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做闸门/构建/一次前向反向检查，不进入训练循环")
    ap.add_argument("--allow-config-mismatch", action="store_true")
    ap.add_argument("--allow-manifest-mismatch", action="store_true")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------

def build_dataset_for_stage(cfg, *, stage, stage_cfg, data_cfg, train_split, val_split,
                            make_augment):
    """按阶段返回 (train_dataset, eval_dataset, manifest_sha256, manifest_path)。"""
    from trainer.data.ddr_dataset import DDRSegDataset
    from trainer.data.external_dataset import (
        ExternalLesionDataset,
        deterministic_dev_split,
    )
    from trainer.data.common import ddr_split_fingerprint, file_sha256

    if stage == "ddr_finetune":
        train_ds = DDRSegDataset(
            data_cfg["ddr_root"], train_split,
            mean=data_cfg["mean"], std=data_cfg["std"],
            augment=make_augment(data_cfg["train_crop"]),
        )
        eval_ds = DDRSegDataset(
            data_cfg["ddr_root"], val_split,
            mean=data_cfg["mean"], std=data_cfg["std"], augment=None,
        )
        fp = ddr_split_fingerprint(data_cfg["ddr_root"], (train_split, val_split))
        return train_ds, eval_ds, fp, data_cfg["ddr_root"]

    # external_pretrain
    manifest = stage_cfg.get("external_manifest")
    if not isinstance(manifest, str) or not manifest:
        raise GateError(
            "pretrain_stage.external_manifest 未填写（外部预训练必须显式给出 manifest 路径）"
        )
    base_dir = stage_cfg.get("manifest_base_dir")
    forbidden = tuple(stage_cfg.get("forbidden_splits") or ("test",))
    allowed = tuple(stage_cfg.get("allowed_splits") or ("train",))
    dev_ratio = float(stage_cfg.get("dev_ratio", 0.1))
    pool = ExternalLesionDataset(
        manifest, mean=data_cfg["mean"], std=data_cfg["std"],
        manifest_base_dir=base_dir, allowed_splits=allowed,
        forbidden_splits=forbidden, augment=None,
    )
    dev_ids, train_ids = deterministic_dev_split(
        pool.records, dev_ratio=dev_ratio, seed=int(stage_cfg.get("seed", 42))
    )
    train_ds = ExternalLesionDataset(
        manifest, mean=data_cfg["mean"], std=data_cfg["std"],
        manifest_base_dir=base_dir, allowed_splits=allowed,
        forbidden_splits=forbidden, sample_ids=train_ids,
        augment=make_augment(data_cfg["train_crop"]), split_label="external_train",
    )
    train_ds.draw_counts = {}
    eval_ds = ExternalLesionDataset(
        manifest, mean=data_cfg["mean"], std=data_cfg["std"],
        manifest_base_dir=base_dir, allowed_splits=allowed,
        forbidden_splits=forbidden, sample_ids=dev_ids,
        augment=None, split_label="external_train_derived_dev",
    )
    return train_ds, eval_ds, file_sha256(manifest), manifest


def make_augment_factory(stage_cfg, *, train_crop):
    from trainer.data.augment import TrainAugment

    if train_crop is None or int(train_crop) <= 0:
        raise GateError(
            "data.train_crop 未配置：训练 crop 尺寸必须显式给出"
            "（不允许猜测；它同时决定增强分辨率与有效前景采样率）"
        )
    aug_cfg = stage_cfg.get("augmentation") or {}

    def factory(crop_size):
        return TrainAugment(
            crop_size=int(crop_size or train_crop),
            resize_long_sides=aug_cfg.get('resize_long_sides'),
            resize_ratio_range=aug_cfg.get("resize_ratio_range", (0.5, 2.0)),
            cat_max_ratio=aug_cfg.get("random_crop_cat_max_ratio", 0.75),
            flip_ratio=aug_cfg.get("flip_ratio", 0.5),
            rotate_degrees=aug_cfg.get("rotate_degrees", (-45.0, 45.0)),
            photometric=aug_cfg.get("photometric", "light"),
        )

    return factory


def build_criterion(cfg):
    from trainer.loss import MaskedBinaryCEDice

    loss_cfg = cfg.get("loss") or {}
    if loss_cfg.get("type") != "binary_per_class":
        raise GateError(f"loss.type 必须为 binary_per_class，当前 {loss_cfg.get('type')!r}")
    if "dice" not in str(loss_cfg.get("recipe", "")).lower():
        raise GateError(f"loss.recipe 必须包含 dice（当前 {loss_cfg.get('recipe')!r}）")
    return MaskedBinaryCEDice(
        smooth=float(loss_cfg.get("smooth", 1e-5)),
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return _run(args)
    except DependencyBlocked as e:
        print(f"[train] DEPENDENCY_BLOCKED: {e}", file=sys.stderr)
        return EXIT_DEPENDENCY_BLOCKED
    except (GateError, ContractError) as e:
        print(f"[train] CONTRACT FAILURE: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_CONTRACT
    except FileNotFoundError as e:
        print(f"[train] IO FATAL: {e}", file=sys.stderr)
        return EXIT_USAGE_IO


def _run(args) -> int:
    from trainer.eval import WholeImageEvaluator
    from trainer.models import build_model
    from trainer.optim import build_optimizer, build_param_groups
    from trainer.seed import seed_everything

    cfg_path = os.path.abspath(args.config)
    cfg = load_yaml(cfg_path)
    if cfg.get('calibrate_hr_ddr_once') and args.stage=='ddr_finetune' and args.mode=='formal':
        if args.resume:raise GateError('Do not change micro grouping after a formal run starts')
        from tools.select_hr_ddr_speed import select
        candidate_out=args.out_dir or (cfg.get('runtime') or {}).get('out_dir')
        if not candidate_out:raise GateError('Speed calibration needs the concrete formal output directory')
        select(cfg_path,candidate_out);cfg=load_yaml(cfg_path)
    validate_config(cfg, cfg_path)
    stage = args.stage
    resolve_stage_section(cfg, stage)

    # ---- CLI 覆盖（仅允许清单内字段）----
    cfg, overridden = apply_overrides(cfg, args.override)

    # ---- 校准覆盖（formal 模式禁止）----
    calib_over = {}
    if args.micro_batch is not None:
        calib_over["micro_batch"] = int(args.micro_batch)
    if args.grad_accum is not None:
        calib_over["grad_accum"] = int(args.grad_accum)
    if args.gradient_checkpointing is not None:
        calib_over["gradient_checkpointing"] = args.gradient_checkpointing == "true"
    if args.mode == "formal" and any(
        v is not None for v in (args.micro_batch, args.grad_accum, args.gradient_checkpointing)
    ):
        raise GateError(
            "formal 模式禁止用 --micro-batch/--grad-accum/--gradient-checkpointing 覆盖："
            "正式训练的批量/累积/重计算必须来自配置文件本身（否则 config hash 不再"
            "对应实际跑的配置）。请先在 calibration 模式校准并回填配置。"
        )
    if calib_over:
        cfg, applied_calib = apply_calibration_overrides(cfg, stage=stage, overrides=calib_over)
        overridden.update({f"{stage}.{k}": v for k, v in applied_calib.items()})

    # ---- 两层闸门 ----
    gate = gate_config(cfg, mode=args.mode, stage=stage, path=cfg_path)
    print(f"[train] gate: readiness={gate['readiness_state']} "
          f"calibration={gate['calibration_state']} mode={args.mode}")

    # ---- split / test 闸门 ----
    data_cfg = ddr_data_config(cfg)
    train_split = args.train_split or data_cfg["train_split"]
    val_split = args.val_split or data_cfg["val_split"]
    assert_no_test_split(cfg, split=train_split, role="训练 split")
    assert_no_test_split(cfg, split=val_split, role="验证/选点 split")
    if train_split == val_split:
        raise GateError(f"train/val split 不得相同: {train_split!r}")
    checkpoint_selection_splits(cfg)
    mcfg = metrics_config(cfg)
    section = resolve_stage_section(cfg, stage)

    batch = effective_batch_check(section, stage=stage)
    seed = int(section.get("seed", 42))
    total_updates = int(args.max_updates or section.get("updates", 0))
    if total_updates <= 0:
        raise GateError(f"{stage}.updates 必须为正整数，当前 {section.get('updates')!r}")
    if args.num_workers is not None:
        section["num_workers"] = int(args.num_workers)
    if args.mode == 'formal' and (args.max_updates is not None or args.eval_max_images is not None or
                                 (cfg.get('runtime') or {}).get('eval_max_images') is not None):
        raise GateError('Formal runs require the locked schedule and the complete evaluation split')

    out_dir = args.out_dir or (cfg.get("runtime") or {}).get("out_dir") or os.path.join(
        "work_dirs", f"{cfg['arm_id']}_{stage}_{args.mode}"
    )
    out_dir = ensure_dir(out_dir)

    print(f"[train] arm={cfg['arm_id']} family={cfg['structure']['family']} stage={stage} "
          f"mode={args.mode} updates={total_updates} train_split={train_split} "
          f"val_split={val_split} out_dir={out_dir}")
    print(f"[train] batch: effective={batch['effective_batch']} "
          f"micro={batch['micro_batch']} accum={batch['grad_accum']}")

    fast_kernels = bool(cfg.get('execution_fast_kernels', False))
    seed_everything(seed, deterministic=not fast_kernels)
    if fast_kernels:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- 模型（依赖不可用 → DependencyBlocked，绝不用假模型）----
    gc_flag = section.get("gradient_checkpointing")
    gc_flag = None if gc_flag is None or isinstance(gc_flag, str) else bool(gc_flag)
    model, meta = build_model(cfg, stage=stage, gradient_checkpointing=gc_flag)
    model.to("cuda" if _cuda_available() else "cpu")
    print(f"[train] model={meta['family']} backbone_prefixes={meta['backbone_prefixes']} "
          f"norm_eval={meta['backbone_norm_eval']}")
    if meta.get("weight_report"):
        rep = meta["weight_report"]
        print(f"[train] weight_report({rep['label']}): loaded={rep['loaded_keys']} "
              f"missing={rep['missing']} unexpected={rep['unexpected']} remapped={rep['remapped']}")

    # ---- 数据 ----
    if meta['family']=='hrdecoder' and stage=='external_pretrain':
        import torch
        model.eval()
        with torch.no_grad():
            probe=model.eval_forward(torch.zeros(1,3,2048,2048,device=_device()))
            if probe.shape!=(1,4,2048,2048) or not torch.isfinite(probe).all():
                raise GateError('HR global/local full-FOV engineering preflight failed')
        del probe
        model.train()
        print('[train] HR full-FOV inference engineering preflight passed (no metric, no checkpoint)')
    make_augment = make_augment_factory(section, train_crop=data_cfg["train_crop"])
    train_ds, eval_ds, manifest_sha, manifest_path = build_dataset_for_stage(
        cfg, stage=stage, stage_cfg=section, data_cfg=data_cfg,
        train_split=train_split, val_split=val_split, make_augment=make_augment,
    )
    print(f"[train] data: train={len(train_ds)} eval={len(eval_ds)} "
          f"manifest_sha256={manifest_sha[:12]}…")

    # ---- loss / 优化器 ----
    criterion = build_criterion(cfg)
    param_groups = build_param_groups(
        model,
        backbone_lr=float(section.get("backbone_lr", 1e-4)),
        decoder_lr=float(section.get("decoder_lr", 1e-4)),
        weight_decay=float(section.get("weight_decay", 1e-2)),
        backbone_prefixes=tuple(meta["backbone_prefixes"]),
    )
    optimizer = build_optimizer(section, param_groups)

    # ---- evaluator（formal 必须有整图口径）----
    evaluator = WholeImageEvaluator(
        model, eval_ds,
        device=_device(),
        mean=data_cfg["mean"], std=data_cfg["std"],
        threshold=mcfg["threshold"],
        mode=mcfg["eval_mode"],
        window_size=mcfg["window_size"],
        window_stride=mcfg["window_stride"],
        max_images=args.eval_max_images or (cfg.get("runtime") or {}).get("eval_max_images"),
        resize_to=(cfg.get('metrics') or {}).get('resize_to'),
    )

    # ---- 生效配置 + 训练器 ----
    from trainer.engine import StageRunner, build_effective_config

    eff = build_effective_config(
        cfg=cfg, stage=stage, mode=args.mode, stage_cfg=section, batch=batch,
        seed=seed, total_updates=total_updates,
        backbone_prefixes=meta["backbone_prefixes"],
        backbone_norm_eval=meta["backbone_norm_eval"],
        config_sha256_value=_effective_digest(cfg),
        data_manifest_sha256=manifest_sha,
        manifest_path=manifest_path,
        out_dir=out_dir,
        extra={
            "config_file_sha256": config_digest(cfg_path),
            "config_overrides": overridden,
            "train_split": train_split,
            "val_split": val_split,
            "source_family": meta["family"],
        },
    )
    runner_type=StageRunner;runner_extra={}
    if section.get('pseudo_manifest'):
        if stage!='ddr_finetune':raise GateError('Pseudo crops are allowed only in DDR followups')
        from trainer.semi import SemiRunner,PseudoDataset
        from trainer.data.common import file_sha256
        pm=section['pseudo_manifest']
        if file_sha256(pm)!=section['pseudo_manifest_sha256']:raise GateError('Pseudo manifest identity changed')
        pseudo=PseudoDataset(pm,data_cfg['mean'],data_cfg['std'],make_augment(data_cfg['train_crop']))
        eff['data_manifest_sha256']=sha256_text(str(manifest_sha)+file_sha256(pm))
        runner_type=SemiRunner
        runner_extra=dict(pseudo_dataset=pseudo,pseudo_micro_batch=section.get('pseudo_micro_batch',2),pseudo_weight=.25)
    runner = runner_type(
        model=model, dataset=train_ds, loss_fn=criterion, optimizer=optimizer,
        cfg=eff, device=_device(), evaluator=evaluator,
        **runner_extra,
    )
    print(f"[train] param audit: groups={runner.param_audit['groups']} "
          f"frozen_tensors={runner.param_audit['n_frozen_tensors']}")

    if args.dry_run:
        chk = runner.dry_run_check()
        print(f"[train] dry-run forward/backward OK: loss={chk['loss']:.6f} "
              f"parts={chk['parts']} tensors_with_grad={chk['tensors_with_grad']}")
        print("[train] dry-run 完成，未写任何 checkpoint")
        return EXIT_OK

    result = runner.run(
        resume_from=args.resume,
        allow_config_mismatch=args.allow_config_mismatch,
        allow_manifest_mismatch=args.allow_manifest_mismatch,
    )
    print("[train] done: " + json.dumps(result, ensure_ascii=False, default=str))
    return EXIT_OK


def _effective_digest(cfg: dict) -> str:
    """生效配置的规范 JSON SHA256（含 CLI 覆盖，故与文件字节哈希不同）。"""
    return sha256_text(json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str))


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    sys.exit(main())
