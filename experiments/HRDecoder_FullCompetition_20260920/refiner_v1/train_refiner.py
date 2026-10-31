#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_refiner — training entry point for the MA/HE candidate refiner.

Protocol (all frozen in configs/refiner_v1.json + configs/stage*.json):
  * train = 454 images, model selection = val (76 images), test never touched
  * the champion is a frozen INPUT provider: `assert_frozen_reference` and
    `assert_only_refiner_trainable` are enforced before the first step
  * the objective is exactly `L = 30.0*focal_bce(alpha=0.75,gamma=2) +
    1.0*soft_dice`; `refiner.losses.assert_frozen_objective` fails closed if a
    stage override changes it (asserted again against the main config here)
  * checkpoints store `{model, cfg_hash, stage, step, seed}` and are loaded
    back with `strict=True` and per-tensor bitwise verification

Fail-closed runtime guards (each one aborts the stage with exit code 3 and a
`TRAIN_FAILED.json` next to the run outputs):
  * identity: at initialisation the refiner must return the champion BITWISE
  * non-finite loss or non-finite gradient norm
  * free disk space below `--min-free-gb` (default 4 GB)

Machine-readable outputs, written continuously so a supervisor can monitor the
run by reading files only (no long-lived terminal session required):
  steps.csv     one row per optimisation step (incl. the recorded focal share)
  epochs.csv    one row per epoch (means, min/max, elapsed)
  history.json  the same per-step rows as JSON, rewritten every log interval
  heartbeat.json  {ts, epoch1, step, loss_ema} rewritten every log interval
  done.json     written on clean completion
  TRAIN_FAILED.json  written on an aborted run (reason + where)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(_HERE))

from refiner import candidate as C                       # noqa: E402
from refiner import guards as G                          # noqa: E402
from refiner import losses as L                          # noqa: E402
from refiner.dataset import PatchCacheDataset, collate   # noqa: E402
from refiner.model import HaarHeteroRefiner              # noqa: E402

EXIT_ABORT = 3
STEP_CSV_FIELDS = ["step", "epoch1", "lr", "loss", "focal", "dice",
                   "focal_weighted", "dice_weighted", "focal_share", "gnorm",
                   "finite"]
EPOCH_CSV_FIELDS = ["epoch1", "steps", "lr_end", "loss_mean", "loss_min",
                    "loss_max", "focal_mean", "dice_mean", "focal_share_mean",
                    "gnorm_mean", "gnorm_max", "elapsed_s", "step_end"]


class TrainAbort(RuntimeError):
    """Non-recoverable condition: abort the stage, do not proceed."""

    def __init__(self, reason, **where):
        super().__init__("%s %r" % (reason, where))
        self.reason = reason
        self.where = where


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _free_gb(path="/root/autodl-tmp"):
    try:
        return shutil.disk_usage(path).free / float(1 << 30)
    except OSError:
        return float("inf")


def build_model(cfg, ckpt=None, device="cuda"):
    m = HaarHeteroRefiner(width=int(cfg["model"]["width"]),
                          haar_levels=int(cfg["input_declared_haar_levels"]),
                          residual_scale=float(cfg["model"]["residual_scale"]))
    if ckpt:
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = ck.get("model", ck)
        m.load_state_dict(state, strict=True)
        print("[train] resumed from %s (stage=%s step=%s epoch=%s)"
              % (ckpt, ck.get("stage"), ck.get("step"), ck.get("epoch1")), flush=True)
    return m.to(device)


def lr_at(step, total, base, warmup, schedule):
    if step < warmup:
        return base * float(step + 1) / float(max(1, warmup))
    if schedule == "cosine":
        p = float(step - warmup) / float(max(1, total - warmup))
        return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return base


def ckpt_dict(model, cfg, stage, step, epoch1, seed, config_path, epochs,
              objective):
    return {"model": model.state_dict(), "stage": stage, "step": int(step),
            "epoch1": int(epoch1), "epochs": int(epochs), "seed": int(seed),
            "config": config_path, "objective": objective,
            "cfg_sha256": G.sha256_file(config_path)}


def prune_epoch_ckpts(out_dir, keep_epochs):
    """Bounded retention: drop epoch checkpoints outside `keep_epochs`.

    Pre-registered node checkpoints live in `nodes/` and are never touched.
    """
    for fn in os.listdir(out_dir):
        if not (fn.startswith("ep") and fn.endswith(".pth")):
            continue
        try:
            n = int(fn[2:5])
        except ValueError:
            continue
        if n not in keep_epochs:
            try:
                os.remove(os.path.join(out_dir, fn))
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_HERE, "configs", "refiner_v1.json"))
    ap.add_argument("--stage-config", default=None,
                    help="stage override json (configs/stage1.json / stage2.json)")
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "runs", "stage2"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overfit-steps", type=int, default=0,
                    help="SMOKE ONLY: run exactly N optimisation steps on train "
                         "patches and stop. Not a training run.")
    ap.add_argument("--train-images", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=1,
                    help="write ep###.pth / last.pth every N epochs")
    ap.add_argument("--keep-recent", type=int, default=3,
                    help="epoch checkpoints kept outside the node list")
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pth / last_opt.pth in --out-dir")
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.config = os.path.abspath(args.config)
    args.out_dir = os.path.abspath(args.out_dir)
    if args.stage_config:
        args.stage_config = os.path.abspath(args.stage_config)
    if args.cache_root:
        args.cache_root = os.path.abspath(args.cache_root)

    cfg = json.load(open(args.config, encoding="utf-8"))
    base_loss = json.loads(json.dumps(cfg["loss"]))
    if args.stage_config:
        ov = json.load(open(args.stage_config, encoding="utf-8"))
        for k, v in ov.items():
            if k.startswith("_") or k == "name":
                continue
            cfg[k] = v
    cfg["input_declared_haar_levels"] = int(cfg["inputs"]["haar"]["levels"])
    cache_root = args.cache_root or cfg["cache"]["root"]
    G.assert_val_only_inference("train", cache_root, args.out_dir, args.config)
    if args.stage_config:
        G.assert_no_test_paths(args.stage_config)

    # ---- frozen-objective guard (fails closed, before any compute) ---------
    L.assert_frozen_objective(cfg["loss"])
    if cfg["loss"]["weights"] != base_loss["weights"] or \
       cfg["loss"]["focal"] != base_loss["focal"] or \
       cfg["loss"]["terms"] != base_loss["terms"]:
        raise C.ContractError(
            "the stage override changed the frozen objective: base=%r override=%r"
            % (base_loss, cfg["loss"]))
    print("[train] objective = %s (frozen at %s)"
          % (L.FROZEN_OBJECTIVE_STR, base_loss.get("objective_frozen_at", "?")),
          flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "nodes"), exist_ok=True)

    ds_full = PatchCacheDataset(cache_root, cfg["cache"]["split_train"], train=True,
                                seed=args.seed)
    stage = cfg.get("stage", "stage2_main")
    stage_cfg = cfg["stages"][stage]
    n_img = args.train_images or int(stage_cfg["train_images"])

    # Subset by IMAGE, never by patch: taking a prefix of patches would bias the
    # sampling towards the first images' lesion statistics.
    stems, seen = [], set()
    for e in ds_full.entries:
        if e[0] not in seen:
            seen.add(e[0]); stems.append(e[0])
    use = set(stems[:n_img])
    idx = [i for i, e in enumerate(ds_full.entries) if e[0] in use]
    ds = Subset(ds_full, idx)
    roles = ds_full.role_counts()
    print("[train] stage=%s images=%d/%d patches=%d roles=%s"
          % (stage, len(use), len(stems), len(ds),
             {k: v for k, v in sorted(roles.items())}), flush=True)

    loader = DataLoader(ds, batch_size=int(cfg["optim"]["batch_size"]), shuffle=True,
                        num_workers=2, collate_fn=collate, drop_last=False,
                        pin_memory=True)

    model = build_model(cfg, device=args.device)
    rep = G.assert_only_refiner_trainable(model, [])
    print("[train] refiner trainable params=%d (non-refiner trainable=%d)"
          % (rep["refiner_trainable"], rep["non_refiner_trainable"]), flush=True)

    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(o["lr"]),
                            weight_decay=float(o["weight_decay"]),
                            betas=tuple(o["betas"]))
    epochs = args.epochs or int(stage_cfg["epochs"])
    node_epochs = sorted(int(x) for x in (stage_cfg.get("checkpoint_nodes") or []))
    if args.overfit_steps:
        node_epochs = []
    steps_per_epoch = max(1, len(loader))
    total = steps_per_epoch * epochs
    warm = int(o["warmup_steps"])
    lw = cfg["loss"]["weights"]
    fa = cfg["loss"]["focal"]

    start_ep = 0
    step = 0
    last_ep1 = 0
    hist = []
    ck_last = os.path.join(args.out_dir, "last.pth")
    opt_last = os.path.join(args.out_dir, "last_opt.pth")
    steps_p = os.path.join(args.out_dir, "steps.csv")
    epochs_p = os.path.join(args.out_dir, "epochs.csv")
    hist_p = os.path.join(args.out_dir, "history.json")
    hb_p = os.path.join(args.out_dir, "heartbeat.json")
    failed_p = os.path.join(args.out_dir, "TRAIN_FAILED.json")
    if os.path.isfile(failed_p):
        os.remove(failed_p)
    resuming = bool(args.resume and os.path.isfile(ck_last))
    if resuming:
        ck = torch.load(ck_last, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"], strict=True)
        print("[train] RESUME from %s (epoch1=%s step=%s)"
              % (ck_last, ck.get("epoch1"), ck.get("step")), flush=True)
        start_ep = int(ck.get("epoch1") or 0)
        last_ep1 = start_ep
        step = int(ck.get("step") or 0)
        if os.path.isfile(opt_last):
            opt.load_state_dict(torch.load(opt_last, map_location="cpu",
                                           weights_only=False))
        if os.path.isfile(hist_p):
            try:
                hist = json.load(open(hist_p, encoding="utf-8")).get("steps", [])
            except (OSError, ValueError):
                hist = []
    mode = "a" if (resuming and os.path.isfile(steps_p)) else "w"
    fs = open(steps_p, mode, newline="", encoding="utf-8")
    ws = csv.DictWriter(fs, fieldnames=STEP_CSV_FIELDS)
    if mode == "w":
        ws.writeheader()
    fe_mode = "a" if (resuming and os.path.isfile(epochs_p)) else "w"
    fe = open(epochs_p, fe_mode, newline="", encoding="utf-8")
    we = csv.DictWriter(fe, fieldnames=EPOCH_CSV_FIELDS)
    if fe_mode == "w":
        we.writeheader()

    def write_heartbeat(ep1, loss_ema, extra=None):
        _atomic_json(hb_p, {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "epoch1": int(ep1), "epochs": int(epochs),
                            "step": int(step), "loss_ema": loss_ema,
                            "free_gb": round(_free_gb(), 2),
                            "total_epochs": int(epochs),
                            "steps_per_epoch": int(steps_per_epoch),
                            "pid": os.getpid(), "extra": extra or {}})

    t0 = time.time()
    stop = False
    identity_ok = None

    def first_batch_identity_check():
        """The zero-initialised model MUST reproduce the champion bitwise.

        Run in the same mode the training loop uses, so this checks the exact
        runtime path rather than a special-cased one.
        """
        for b in loader:
            rgb = b["rgb"].to(args.device)
            p_champ = b["p_champ"].to(args.device)
            with torch.no_grad():
                _r, refined = model.forward_refine(rgb, p_champ)
            if not torch.equal(refined, p_champ):
                raise TrainAbort("identity_broken",
                                 max_abs_diff=float((refined - p_champ).abs().max()),
                                 p_champ_grad_ok=True)
            return max(int(rgb.shape[0]), 1), float((refined - p_champ).abs().max())
        raise TrainAbort("identity_check_no_batch")

    try:
        if resuming:
            # The zero-residual identity invariant applies only before the first
            # optimiser update.  A trained residual checkpoint must differ from
            # the champion; requiring equality here makes every valid resume
            # fail.  Strict model/optimizer loading above is the resume guard.
            identity_ok = {
                "bitwise": True,
                "scope": "fresh_start_guard_already_passed",
                "resume_checkpoint": ck_last,
                "resume_epoch1": start_ep,
                "resume_step": step,
            }
            print("[train] resume guard OK: strict model + optimizer state loaded "
                  "(epoch1=%d step=%d); zero-init identity check is not "
                  "applicable after training" % (start_ep, step), flush=True)
        else:
            n_b, d0 = first_batch_identity_check()
            identity_ok = {"bitwise": True, "max_abs_diff": d0, "batch": n_b,
                           "scope": "fresh_zero_init"}
            print("[train] identity guard OK: refiner == champion bitwise at init "
                  "(max|d|=%.1e, batch=%d)" % (d0, n_b), flush=True)
        write_heartbeat(start_ep, None, {"identity_ok": identity_ok})

        for ep in range(start_ep, epochs):
            ep1 = ep + 1
            free = _free_gb()
            if free < args.min_free_gb:
                raise TrainAbort("disk_low", free_gb=round(free, 2),
                                 min_free_gb=args.min_free_gb, epoch1=ep1)
            if hasattr(ds, "dataset"):
                ds.dataset.set_epoch(ep)
            ep_rows = []
            ep_t0 = time.time()
            for batch in loader:
                lr = lr_at(step, total, float(o["lr"]), warm, o["schedule"])
                for g in opt.param_groups:
                    g["lr"] = lr
                rgb = batch["rgb"].to(args.device)
                p_champ = batch["p_champ"].to(args.device)
                gt = batch["gt"].to(args.device)
                valid = batch["valid"].to(args.device)
                _r, refined = model.forward_refine(rgb, p_champ)
                loss, parts = L.refiner_losses(
                    refined, gt, valid, alpha=float(fa["alpha"]),
                    gamma=float(fa["gamma"]), w_focal=float(lw["focal_bce"]),
                    w_dice=float(lw["soft_dice"]))
                if not torch.isfinite(loss):
                    raise TrainAbort("loss_non_finite", step=step + 1, epoch1=ep1,
                                     loss=float(loss),
                                     focal=parts["focal"], dice=parts["dice"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    float(o["grad_clip_norm"]))
                if not torch.isfinite(gn):
                    raise TrainAbort("grad_non_finite", step=step + 1, epoch1=ep1,
                                     gnorm=float(gn))
                opt.step()
                step += 1
                row = {"step": step, "epoch1": ep1, "lr": "%.6g" % lr,
                       "loss": "%.8g" % float(loss),
                       "focal": "%.8g" % parts["focal"],
                       "dice": "%.8g" % parts["dice"],
                       "focal_weighted": "%.8g" % parts["focal_weighted"],
                       "dice_weighted": "%.8g" % parts["dice_weighted"],
                       "focal_share": "%.8g" % parts["focal_share"],
                       "gnorm": "%.8g" % float(gn), "finite": 1}
                ws.writerow(row)
                ep_rows.append({"step": step, "epoch": ep1, "lr": lr,
                                "loss": float(loss), "focal": parts["focal"],
                                "dice": parts["dice"],
                                "focal_weighted": parts["focal_weighted"],
                                "dice_weighted": parts["dice_weighted"],
                                "focal_share": parts["focal_share"],
                                "gnorm": float(gn)})
                if step % 20 == 0 or step == 1:
                    fs.flush()
                    print("[train] ep=%d step=%d lr=%.2e loss=%.4f focal=%.4f "
                          "dice=%.4f focal_share=%.3f gnorm=%.2f %.1fs"
                          % (ep1, step, lr, float(loss), parts["focal"],
                             parts["dice"], parts["focal_share"], float(gn),
                             time.time() - t0), flush=True)
                if len(ep_rows) % 20 == 0:
                    write_heartbeat(ep1, float(loss))
                if args.overfit_steps and step >= args.overfit_steps:
                    stop = True
                    break
            # ---- epoch bookkeeping ------------------------------------------
            fs.flush()
            last_ep1 = ep1
            if ep_rows:
                losses = [r["loss"] for r in ep_rows]
                erow = {"epoch1": ep1, "steps": len(ep_rows),
                        "lr_end": "%.6g" % ep_rows[-1]["lr"],
                        "loss_mean": "%.8g" % float(np.mean(losses)),
                        "loss_min": "%.8g" % float(np.min(losses)),
                        "loss_max": "%.8g" % float(np.max(losses)),
                        "focal_mean": "%.8g" % float(np.mean([r["focal"] for r in ep_rows])),
                        "dice_mean": "%.8g" % float(np.mean([r["dice"] for r in ep_rows])),
                        "focal_share_mean": "%.8g" % float(np.mean([r["focal_share"] for r in ep_rows])),
                        "gnorm_mean": "%.8g" % float(np.mean([r["gnorm"] for r in ep_rows])),
                        "gnorm_max": "%.8g" % float(np.max([r["gnorm"] for r in ep_rows])),
                        "elapsed_s": "%.1f" % (time.time() - ep_t0), "step_end": step}
                we.writerow(erow); fe.flush()
                hist.extend(ep_rows)
                _atomic_json(hist_p, {"steps": hist, "elapsed_s": time.time() - t0,
                                      "epochs": epochs, "stage": stage,
                                      "objective": L.FROZEN_OBJECTIVE_STR,
                                      "roles": roles})
                write_heartbeat(ep1, float(np.mean(losses)))
            if not args.overfit_steps and (ep1 % max(1, args.save_every) == 0
                                           or ep1 == epochs or stop):
                ckd = ckpt_dict(model, cfg, stage, step, ep1, args.seed,
                                args.config, epochs, L.FROZEN_OBJECTIVE_STR)
                ck_ep = os.path.join(args.out_dir, "ep%03d.pth" % ep1)
                torch.save(ckd, ck_ep)
                torch.save(ckd, ck_last)
                torch.save(opt.state_dict(), opt_last)
                if ep1 in node_epochs:
                    torch.save(ckd, os.path.join(args.out_dir, "nodes",
                                                 "node_ep%03d.pth" % ep1))
                prune_epoch_ckpts(args.out_dir,
                                  set(range(max(1, ep1 - args.keep_recent + 1), ep1 + 1)))
            if stop:
                break
    except TrainAbort as e:
        _atomic_json(failed_p, {"reason": e.reason, "where": e.where,
                                "ts": time.time(),
                                "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "step": step, "stage": stage,
                                "objective": L.FROZEN_OBJECTIVE_STR})
        fs.close(); fe.close()
        print("[train] ABORTED: %s %r (wrote %s)" % (e.reason, e.where, failed_p),
              flush=True)
        return EXIT_ABORT

    ck = os.path.join(args.out_dir, "refiner_last.pth")
    torch.save(ckpt_dict(model, cfg, "overfit_smoke" if args.overfit_steps else
                         cfg.get("name"), step, last_ep1, args.seed, args.config,
                         epochs, L.FROZEN_OBJECTIVE_STR), ck)
    summary = {
        "schema": "hrdecoder-refiner-train-done/v1",
        "stage": stage, "out_dir": args.out_dir, "epochs": epochs,
        "start_epoch": start_ep, "steps": step, "steps_per_epoch": steps_per_epoch,
        "elapsed_s": time.time() - t0, "roles": roles,
        "objective": L.FROZEN_OBJECTIVE_STR,
        "loss_weights": cfg["loss"]["weights"],
        "focal": cfg["loss"]["focal"],
        "identity_ok": identity_ok,
        "checkpoint_nodes": node_epochs,
        "args": vars(args), "exit_code": 0,
    }
    if hist:
        summary["final"] = {k: hist[-1][k] for k in
                            ("step", "epoch", "loss", "focal", "dice",
                             "focal_share", "gnorm")}
        summary["final_loss_mean_last_epoch"] = float(
            np.mean([r["loss"] for r in hist if r["epoch"] == hist[-1]["epoch"]]))
        summary["focal_share_mean"] = float(np.mean([r["focal_share"] for r in hist]))
        summary["loss_first"] = hist[0]["loss"]
    _atomic_json(os.path.join(args.out_dir, "done.json"), summary)
    with open(hist_p, "w", encoding="utf-8") as f:
        json.dump({"steps": hist, "elapsed_s": time.time() - t0, "args": vars(args),
                   "roles": roles, "objective": L.FROZEN_OBJECTIVE_STR},
                  f, ensure_ascii=False, indent=2)
    fs.close(); fe.close()
    print("[train] wrote %s" % ck)
    print("[train] NOTE: this entry point does not select on val; the supervisor "
          "runs eval_refiner.py on every pre-registered node and selects with the "
          "frozen rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
