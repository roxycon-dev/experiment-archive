"""Locked v3 execution: one SGD step per two images, complete-batch denominators.

Only a persisted complete-boundary PAUSED checkpoint can resume. Interrupted or
failed runs cannot roll back to an older last.pt. Importing starts no training.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (SegmentationData, ResumableBatches, worker_setup, verify_training_files,
                  ASSIGNMENT_SHA, IDENTITY_SHA)
from losses import SparseLesionLoss
from metrics import accumulate, summarize
from model import NativePyramid, file_sha256, resize, W48_SHA256
from monitoring import HealthGuard

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ("model.py", "data.py", "losses.py", "metrics.py", "monitoring.py",
                "train.py", "evaluate.py", "verify.py", "protocol.json", "requirements.txt",
                "tests.py", "test_model_v3.py", "test_train_v3.py",
                "00_最终架构方案.md", "02_v3_最终架构设计_20261016.md")


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def acquire_run_lease(directory, identity):
    path = Path(directory) / "run_active.json"
    token = str(uuid.uuid4())
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"token": token, "pid": os.getpid(), "run_identity": identity,
                   "created_unix": time.time(), "stale_lease_requires_diagnosis": True}, stream)
        stream.flush(); os.fsync(stream.fileno())
    return path, token


def release_run_lease(lease):
    path, token = lease
    if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("token") == token:
        path.unlink()


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".writing")
    with temporary.open("wb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def source_identity():
    material = {name: file_sha256(HERE / name) for name in SOURCE_FILES}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest(), material


def runtime_identity():
    dependencies = {}
    for name in ("torch", "torchvision", "timm", "numpy", "opencv-python", "Pillow"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    cuda = torch.cuda.is_available()
    return {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": dependencies, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
            "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "precision": "BF16_forward_FP32_loss_and_master_weights"}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    if set(state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete RNG state")
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed")
        torch.cuda.set_rng_state_all(state["cuda"])


def learning_rate(index, protocol):
    total = protocol["updates"]
    if not 0 <= index < total:
        raise ValueError("Optimizer update outside the 40k budget")
    value = protocol["optimizer"]["lr"] * (1 - index / total)
    warmup = protocol["schedule"]["warmup_updates"]
    if index < warmup:
        ratio = protocol["schedule"]["warmup_ratio"]
        value *= ratio + (1 - ratio) * index / warmup
    return value


def make_optimizer(model, protocol):
    p = protocol["optimizer"]
    return torch.optim.SGD(model.parameters(), lr=p["lr"], momentum=p["momentum"],
                           weight_decay=p["weight_decay"])


def gradient_norms(model):
    """Read existing task gradients; no extra autograd pass or weight decay."""
    groups, missing, parameter_modules, missing_parameter_modules = {}, {}, {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = name.split(".")
        key = ".".join(parts[:3] if parts[:2] == ["backbone", "net"] else parts[:2])
        parameter_module = name.rsplit(".", 1)[0]
        if parameter.grad is None:
            missing[key] = missing.get(key, 0) + 1
            missing_parameter_modules[parameter_module] = missing_parameter_modules.get(parameter_module, 0) + 1
        else:
            value = parameter.grad.detach().float().norm(2)
            groups.setdefault(key, []).append(value)
            parameter_modules.setdefault(parameter_module, []).append(value)
    keys = sorted(groups)
    norms = (torch.stack([torch.stack(groups[key]).norm(2) for key in keys]).cpu().tolist()
             if keys else [])
    parameter_module_keys = sorted(parameter_modules)
    parameter_module_norms = (torch.stack([torch.stack(parameter_modules[key]).norm(2)
                                          for key in parameter_module_keys]).cpu().tolist()
                              if parameter_module_keys else [])
    direct = getattr(getattr(model, "decoder", None), "direct_classifiers", None)
    per_class = None
    if direct is not None:
        if len(direct) != 4:
            raise ValueError("Four direct semantic classifiers required")
        per_class = []
        for layer in direct:
            grad = layer.weight.grad
            per_class.append(None if grad is None else grad.detach().float().flatten(1).norm(dim=1).cpu().tolist())
    return {"modules": dict(zip(keys, norms)), "missing_gradient_parameters": missing,
            "parameter_modules": dict(zip(parameter_module_keys, parameter_module_norms)),
            "missing_gradient_parameter_modules": missing_parameter_modules,
            "direct_scale_class": per_class, "class_order": ["EX", "HE", "SE", "MA"],
            "scale_order": ["F1", "F2", "F3", "F4"]}


def update_once(model, optimizer, criterion, batch, *, micro_batch=1, clip_norm=5, amp=True):
    if len(batch["image"]) != 2 or micro_batch not in (1, 2):
        raise ValueError("Exactly two images and one optimizer step are required")
    denominator = criterion.batch_denominators(batch["label"])
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    metrics = {}
    for start in range(0, 2, micro_batch):
        image = batch["image"][start:start + micro_batch].to(device, non_blocking=True)
        label = batch["label"][start:start + micro_batch].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp and device.type == "cuda"):
            output = model(image)
        loss, values = criterion(output, label, denominators=denominator)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss; optimizer.step not called")
        loss.backward()  # shared complete-batch denominators; never multiply by 1/2
        for key, value in values.items():
            metrics[key] = metrics.get(key, 0.) + float(value)
        del output, loss, image, label
    before = gradient_norms(model)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    after = gradient_norms(model)
    norm_value = float(norm)
    metrics.update(grad_norm_before_clip=norm_value,
                   clip_coefficient=min(1., clip_norm / (norm_value + 1e-6)),
                   gradients_before_clip=before, gradients_after_clip=after,
                   supervision_positive_pixels=denominator["positive"].cpu().tolist(),
                   supervision_positive_images=denominator["positive_images"].cpu().tolist())
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return metrics


@torch.no_grad()
def evaluate(model, dataset, indices=None, training_probe=False, amp=True):
    indices = list(range(len(dataset))) if indices is None else list(indices)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    counts = {key: np.zeros((4, 3), dtype=np.int64) for key in ("Z", "S")}
    positives = {key: [[] for _ in range(4)] for key in counts}
    per_image = []
    started = time.monotonic()
    try:
        for index in indices:
            image, label, row = dataset.evaluation(index)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                output = model(image[None].to(device), diagnostics=True) if training_probe else model(image[None].to(device))
            item = {"sid": row["sid"], "heads": {}}
            for head, key in (("Z", "logits"), ("S", "auxiliary_logits")):
                source_logits = output[key].float()
                if head == "S":
                    # Keep the same 512 -> full canvas -> native geometry as Z's
                    # semantic contribution. Two discrete resizes cannot merge.
                    source_logits = resize(source_logits, output["logits"].shape[-2:])
                logits = resize(source_logits, label.shape)[0]
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Nonfinite evaluation logits")
                prediction = (logits.sigmoid() > .5).cpu().numpy()
                local = np.zeros((4, 3), dtype=np.int64)
                accumulate(local, prediction, label); counts[head] += local
                item["heads"][head] = local.tolist()
                if training_probe:
                    for c in range(4):
                        mask = torch.as_tensor(label == c + 1, device=device)
                        positives[head][c].append(logits[c][mask].cpu().numpy())
                del logits, prediction, source_logits
            item["counts"] = item["heads"]["Z"]
            if training_probe:
                item["forward_diagnostics"] = output["diagnostics"]
            per_image.append(item)
            del output, image
    finally:
        model.train(was_training)
    heads = {key: summarize(value, len(indices)) for key, value in counts.items()}
    if training_probe:
        quantiles = [0., .1, .25, .5, .75, .9, 1.]
        for head in heads:
            heads[head]["positive_logit_quantile_levels"] = quantiles
            heads[head]["positive_logit_quantiles"] = [np.quantile(np.concatenate(v), quantiles).tolist()
                if sum(len(a) for a in v) else None for v in positives[head]]
            heads[head]["positive_logit_median"] = [None if v is None else v[3]
                                                   for v in heads[head]["positive_logit_quantiles"]]
    result = dict(heads["Z"])
    result.update(heads=heads, seconds=time.monotonic() - started, split=dataset.split, per_image=per_image)
    if training_probe:
        result["scope"] = "fixed eight training whole images; forward-only stability diagnostic"
    return result


def resolve_probe_indices(train_data, preflight):
    selected = preflight["train_probe_sids"]
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError("Exactly eight unique fixed training probe images required")
    order = {row["sid"]: i for i, row in enumerate(train_data.rows)}
    indices = [order[sid] for sid in selected]
    presence = np.zeros(4, dtype=np.int64)
    for index in indices:
        _, label, _ = train_data.native(index)
        presence += [bool((label == c + 1).any()) for c in range(4)]
    if (presence < 2).any():
        raise ValueError("Each class requires at least two positive whole probe images")
    return indices


def sampler_cursor(n, completed, seed):
    epoch, offset = divmod(completed * 2, n)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    return {"n": n, "seed": seed, "next_update": completed, "epoch": epoch,
            "offset": offset, "epoch_order": order, "consumed_images": completed * 2,
            "augmentation_rule": "seed + 2 * zero_based_update + image_in_batch"}


def validate_resume(state, progress, expected_identity, pause_id, *, failure_present=False):
    if failure_present:
        raise RuntimeError("Terminal failure marker exists; rollback/resume forbidden")
    if state.get("status") != "PAUSED" or progress.get("status") != "PAUSED":
        raise RuntimeError("Resume requires an explicitly controlled PAUSED SGD boundary")
    if not pause_id or pause_id != state.get("pause_id") or pause_id != progress.get("pause_id"):
        raise RuntimeError("Resume must explicitly name the current pause_id")
    completed = state.get("completed")
    if not isinstance(completed, int) or not 0 < completed <= 40000:
        raise RuntimeError("Invalid paused update count")
    if any(value != completed for value in (state.get("executed_updates"), state.get("checkpoint_persisted_updates"),
               progress.get("executed_updates"), progress.get("checkpoint_persisted_updates"))):
        raise RuntimeError("Executed/persisted counters disagree; rollback forbidden")
    if progress.get("in_flight_update") is not None:
        raise RuntimeError("An optimizer attempt is unresolved")
    if state.get("run_identity") != expected_identity or progress.get("run_identity") != expected_identity:
        raise RuntimeError("Source/data/preflight/environment/microbatch identity changed")
    if state.get("sampler") != sampler_cursor(454, completed, 42):
        raise RuntimeError("Sampler cursor or permutation mismatch")
    for key in ("model", "optimizer", "rng", "loader_rng", "guard", "validation_history", "best"):
        if key not in state:
            raise RuntimeError("Incomplete paused state: " + key)
    if set(state["rng"]) != {"python", "numpy", "torch", "cuda"}:
        raise RuntimeError("Incomplete paused RNG")
    if state["guard"].get("last_supervision_update") != completed:
        raise RuntimeError("Incomplete supervision history")
    if completed == 40000 and (not state["validation_history"] or
                              state["validation_history"][-1].get("update") != 40000):
        raise RuntimeError("Endpoint pause requires its already-completed 40k validation")
    return completed


def validate_protocol(protocol):
    locked = {"version": 3, "updates": 40000, "global_batch": 2, "canvas": 2048,
              "seed": 42, "head_seed": 20261011, "train_probe_interval": 250, "full_val_interval": 2000,
              "train_probe_count": 8, "classes": ["EX", "HE", "SE", "MA"],
              "threshold": .5, "threshold_comparison": "strict_greater",
              "val_target_percent": 45.321831227388, "test_target_percent": 45.860383832668,
              "precision": "BF16_forward_FP32_loss_and_master_weights",
              "decoder_norm": "per_pixel_channel_LayerNorm_FP32_eps_1e-5",
              "decoder_parameters": 246132, "total_parameters": 65571252,
              "pretrained_sha256": W48_SHA256, "assignment_sha256": ASSIGNMENT_SHA,
              "data_identity_sha256": IDENTITY_SHA, "split_sizes": {"train": 454, "val": 76, "test": 227},
              "teacher": False, "TTA": False, "EMA": False,
              "primary_checkpoint": "fixed_40000_raw_weights"}
    if any(protocol.get(key) != value for key, value in locked.items()):
        raise ValueError("Formal v3 budget, seed or cadence changed")
    optimizer = {"type": "SGD", "lr": .005, "momentum": .9, "weight_decay": .0005, "clip_norm": 5.}
    if protocol["optimizer"] != optimizer:
        raise ValueError("Formal optimizer changed")
    if protocol["schedule"] != {"type": "linear_poly", "warmup_updates": 1500, "warmup_ratio": 1e-6}:
        raise ValueError("Formal learning rate schedule changed")
    if protocol["loss"] != {"negative_ratio": 3, "min_negatives": 1024, "positive_weight": .25,
                             "negative_weight": .25, "dense_weight": .05, "auxiliary_weight": .2}:
        raise ValueError("Formal loss changed")
    if protocol["train_probe_collapse"] != {"minimum_update": 4000, "learned_recall_min": .05,
            "learned_consecutive": 2, "lost_recall_max": .001, "lost_consecutive": 2,
            "unlearned_zero_tp_consecutive": 3, "supervision_window_updates": 1000}:
        raise ValueError("Training health rules changed")
    if protocol["validation_collapse"] != {"minimum_update": 4000, "miou_drop_pp": 15.,
            "learned_recall_min": .05, "learned_consecutive": 2, "lost_recall_max": .001,
            "lost_consecutive": 2}:
        raise ValueError("Validation health rules changed")


def validate_preflight(preflight, digest, identity_sha, environment, micro_batch=1):
    if (preflight.get("status") != "PASS" or preflight.get("source_sha256") != digest
            or preflight.get("version") != 3 or preflight.get("protocol_version") != 3
            or preflight.get("mode") != "server_preflight" or preflight.get("formal_updates") != 0
            or preflight.get("test_accessed") is not False):
        raise RuntimeError("Formal run requires completed server v3 preflight on this exact source")
    declared_source = preflight.get("source_files")
    if (not isinstance(declared_source, dict) or set(declared_source) != set(SOURCE_FILES)
            or hashlib.sha256(json.dumps(declared_source, sort_keys=True).encode()).hexdigest() != digest):
        raise RuntimeError("Preflight source-file identity is incomplete or inconsistent")
    for stage in ("unit_tests", "data_audit", "backbone_integration", "precision_micro_parity",
                  "real_crop_learning", "full_resolution_smoke"):
        if preflight.get(stage, {}).get("status") != "PASS":
            raise RuntimeError("Missing real preflight gate: " + stage)
    if preflight.get("environment") != environment:
        raise RuntimeError("Preflight environment differs from formal execution")
    audit = preflight["data_audit"]
    if (audit.get("identity_sha256") != identity_sha or audit.get("test_pixels_read") is not False
            or len(audit.get("positive_image_coverage", [])) != 4
            or min(audit["positive_image_coverage"]) < 2):
        raise RuntimeError("Invalid audited data/probe identity")
    probe_sids = preflight.get("train_probe_sids", [])
    if len(probe_sids) != 8 or len(set(probe_sids)) != 8 or audit.get("train_probe_sids") != probe_sids:
        raise RuntimeError("Invalid frozen eight-image training probe")
    crop = preflight["real_crop_learning"]
    if (crop.get("discarded_updates") != 256 or crop.get("required_nodes") != [224, 256]
            or crop.get("required_each_class_iou_percent") != 20 or crop.get("weights_retained") is not False):
        raise RuntimeError("Crop gate did not use the locked 256-update discarded budget")
    gates = [row for row in crop.get("trajectory", []) if row.get("update") in (224, 256)]
    if (sorted(row["update"] for row in gates) != [224, 256]
            or any(len(row.get("IoU_percent", [])) != 4 or
                   not all(np.isfinite(v) and v >= 20 for v in row["IoU_percent"]) for row in gates)):
        raise RuntimeError("Both 224/256 crop nodes must pass all four classes")
    losses = np.asarray(crop.get("step_losses", []), dtype=np.float64)
    if (losses.shape != (256,) or not np.isfinite(losses).all()
            or not float(losses[-16:].mean()) < float(losses[:16].mean())
            or not np.isclose(crop.get("first_16_loss_mean", np.nan), losses[:16].mean())
            or not np.isclose(crop.get("last_16_loss_mean", np.nan), losses[-16:].mean())):
        raise RuntimeError("Crop gate lacks the required first/last sixteen loss decrease")
    crops = crop.get("crops", [])
    if (len(crops) != 8 or any(row.get("crop_yxhw", [])[-2:] != [256, 256] for row in crops)
            or any(sum(row.get("class_positive_pixels", [0] * 4)[c] > 0 for row in crops) < 2 for c in range(4))):
        raise RuntimeError("Crop bank does not have two positive 256 crops per class")
    full = preflight["full_resolution_smoke"]
    if (micro_batch != 1 or full.get("micro_batch") != micro_batch
            or full.get("discarded_updates") != 8 or full.get("canvas") != 2048
            or full.get("global_batch") != 2 or len(full.get("metrics", [])) != 8
            or full.get("all_BN_buffers_unchanged") is not True or full.get("weights_retained") is not False):
        raise RuntimeError("Eight real-2048 global-batch2 updates and frozen BN gate required")
    internal_gate = full.get("internal_parameter_module_gradient_gate", {})
    if (internal_gate.get("status") != "PASS" or not internal_gate.get("active_modules")
            or not all(active is True for active in internal_gate["active_modules"].values())):
        raise RuntimeError("Every residual parameter module must have task gradient after zero heads update")


def validate_execution_ledger(path, completed):
    observed = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            observed += 1
            if (entry.get("update") != observed or entry.get("executed_updates") != observed
                    or not 0 <= entry.get("checkpoint_persisted_updates", -1) <= observed):
                raise RuntimeError("Execution ledger is inconsistent; rollback forbidden")
    if observed != completed:
        raise RuntimeError("Execution ledger differs from paused checkpoint; rollback forbidden")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True); parser.add_argument("--identity", required=True)
    parser.add_argument("--pretrained", required=True); parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, choices=(1,), default=1,
                        help="Formal 2048 gate locks microbatch1; batch2 remains a unit/parity check")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-pause-id", help="Explicitly acknowledge the current saved health pause")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    if (out / "failure.json").exists():
        raise RuntimeError("Terminal failure marker exists; old checkpoints cannot resume this experiment")
    if args.resume and not (out / "last.pt").exists():
        raise FileNotFoundError("No own last.pt to resume")
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError("Fresh run needs an empty output directory")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Formal run requires a CUDA GPU with BF16 support")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("One global batch2 optimizer only; DDP is not enabled")
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    seed_all(protocol["seed"]); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    environment = runtime_identity()
    preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
    digest, source = source_identity()
    identity_sha = file_sha256(args.identity)
    validate_preflight(preflight, digest, identity_sha, environment, args.micro_batch)
    if file_sha256(args.pretrained) != protocol["pretrained_sha256"]:
        raise RuntimeError("Wrong pretrained file")
    verify_training_files(args.data_root, args.identity)
    run_identity = {"source_sha256": digest, "identity_sha256": identity_sha,
                    "pretrained_sha256": protocol["pretrained_sha256"],
                    "preflight_sha256": file_sha256(args.preflight), "environment": environment,
                    "micro_batch": args.micro_batch, "workers": args.workers}
    resume_state = None
    out.mkdir(parents=True, exist_ok=True)
    lease = acquire_run_lease(out, run_identity)
    progress, acknowledgement_error = {}, False
    try:
        # Recheck under the exclusive lease: another invocation may have
        # finished while this invocation was auditing the dataset.
        if not args.resume and any(p.name != "run_active.json" for p in out.iterdir()):
            raise FileExistsError("Fresh output became nonempty during startup; refusing to overwrite it")
        if args.resume:
            progress = json.loads((out / "progress.json").read_text(encoding="utf-8"))
            # A missing acknowledgement is a command error, before mutating run state.
            if not args.resume_pause_id or args.resume_pause_id != progress.get("pause_id"):
                acknowledgement_error = True
                raise RuntimeError("Resume requires --resume-pause-id naming the current PAUSED boundary")
            if progress.get("checkpoint_sha256") != file_sha256(out / "last.pt"):
                raise RuntimeError("Persisted checkpoint differs from boundary record")
            resume_state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
            validate_resume(resume_state, progress, run_identity, args.resume_pause_id)
            validate_execution_ledger(out / "train.jsonl", resume_state["completed"])
    except BaseException as error:
        # Wrong/missing acknowledgement alone has not invalidated a saved state.
        if args.resume and not acknowledgement_error:
            write_json(out / "failure.json", {"status": "FAILED", "phase": "resume_state_validation",
                "executed_updates": progress.get("executed_updates"),
                "checkpoint_persisted_updates": progress.get("checkpoint_persisted_updates"),
                "in_flight_update": progress.get("in_flight_update"), "resume_forbidden": True,
                "error_type": type(error).__name__, "error": str(error), "test_accessed": False})
        release_run_lease(lease)
        raise
    completed, persisted, best, history, pause_id = 0, 0, None, [], None
    pause_reasons = []
    in_flight, checkpoint_sha = None, None
    guard = HealthGuard(protocol)
    loader_generator = torch.Generator().manual_seed(42)
    model = optimizer = None

    def progress_record(status):
        return {"status": status, "completed": completed, "executed_updates": completed,
                "checkpoint_persisted_updates": persisted, "expected": 40000,
                "in_flight_update": in_flight, "checkpoint_sha256": checkpoint_sha,
                "pause_id": pause_id, "pause_reasons": pause_reasons,
                "run_identity": run_identity, "best": best}

    def snapshot(status):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed": completed,
                "executed_updates": completed, "checkpoint_persisted_updates": completed,
                "source_sha256": digest, "identity_sha256": identity_sha, "run_identity": run_identity,
                "environment": environment,
                "rng": rng_state(), "loader_rng": loader_generator.get_state(),
                "sampler": sampler_cursor(454, completed, protocol["seed"]),
                "best": best, "validation_history": history, "guard": guard.state_dict(),
                "status": status, "pause_id": pause_id, "pause_reasons": pause_reasons,
                "primary": "fixed_40000_raw_weights"}

    def persist(status):
        nonlocal persisted, checkpoint_sha
        save_checkpoint(out / "last.pt", snapshot(status))
        checkpoint_sha = file_sha256(out / "last.pt")
        persisted = completed
        write_json(out / "progress.json", progress_record(status))

    try:
        model = NativePyramid(args.pretrained, seed=protocol["seed"], head_seed=protocol["head_seed"]).cuda().train()
        optimizer = make_optimizer(model, protocol)
        criterion = SparseLesionLoss(**protocol["loss"])
        train_data = SegmentationData(args.data_root, args.identity, "train", protocol["canvas"])
        val_data = SegmentationData(args.data_root, args.identity, "val", protocol["canvas"])
        probe_indices = resolve_probe_indices(train_data, preflight)
        if resume_state is not None:
            model.load_state_dict(resume_state["model"], strict=True)
            optimizer.load_state_dict(resume_state["optimizer"])
            completed = persisted = resume_state["completed"]
            best, history = resume_state["best"], resume_state["validation_history"]
            guard.load_state_dict(resume_state["guard"])
            loader_generator.set_state(resume_state["loader_rng"])
            restore_rng(resume_state["rng"])
            checkpoint_sha = file_sha256(out / "last.pt")
            append_json(out / "control.jsonl", {"event": "resume_PAUSED", "update": completed,
                        "acknowledged_pause_id": args.resume_pause_id,
                        "acknowledged_reasons": resume_state.get("pause_reasons")})
            del resume_state
        else:
            write_json(out / "protocol.json", {"protocol": protocol, "run_identity": run_identity,
                       "source_sha256": digest, "source_files": source,
                       "pretrained": model.backbone.initialization, "identity_sha256": identity_sha,
                       "initialized_from_task_checkpoint": False, "formal_updates": 0})
        sampler = ResumableBatches(len(train_data), completed, protocol["updates"], protocol["seed"])
        loader_args = {"num_workers": args.workers, "pin_memory": True,
                       "worker_init_fn": worker_setup, "generator": loader_generator}
        if args.workers:
            loader_args.update(prefetch_factor=1, persistent_workers=True)
        loader = DataLoader(train_data, batch_sampler=sampler, **loader_args)
        write_json(out / "progress.json", progress_record("TRAINING"))
        write_json(out / "result.json", {"status": "TRAINING", "completed": completed,
                                        "test_accessed": False, "target_achieved": False})
        started = time.monotonic()
        loader_rng_before_iterator = loader_generator.get_state()
        iterator = iter(loader)
        if args.resume:
            # Worker creation consumes a base seed, but augmentations use only the
            # saved per-sample seeds. Preserve the uninterrupted generator state.
            loader_generator.set_state(loader_rng_before_iterator)
        for batch in iterator:
            lr = learning_rate(completed, protocol)
            for group in optimizer.param_groups:
                group["lr"] = lr
            in_flight = completed + 1
            write_json(out / "progress.json", progress_record("TRAINING"))
            values = update_once(model, optimizer, criterion, batch, micro_batch=args.micro_batch,
                                  clip_norm=protocol["optimizer"]["clip_norm"])
            completed += 1
            in_flight = None
            guard.note_supervision(completed, values["supervision_positive_pixels"])
            event = {"update": completed, "executed_updates": completed,
                     "checkpoint_persisted_updates": persisted, "lr": lr,
                     "seconds_this_process": time.monotonic() - started, "metrics": values,
                     "training_sids": list(batch["sid"])}
            append_json(out / "train.jsonl", event)
            write_json(out / "progress.json", progress_record("TRAINING"))
            if completed == 1 or completed % 50 == 0:
                print(json.dumps({"update": completed, "lr": lr, "loss": values["loss"],
                                  "grad_norm_before_clip": values["grad_norm_before_clip"]}), flush=True)
            reasons = []
            if completed % 250 == 0:
                probe = evaluate(model, train_data, probe_indices, training_probe=True)
                probe["update"] = completed
                write_json(out / "probes" / f"train_{completed:05d}.json", probe)
                reasons.extend(guard.observe_probe(probe, completed))
            if completed % 2000 == 0:
                val = evaluate(model, val_data); val["update"] = completed
                write_json(out / f"val_{completed:05d}.json", val)
                reasons.extend(guard.observe_validation(val, completed))
                history.append(val)
                if best is None or val["mIoU_percent"] > best["mIoU_percent"]:
                    best = {"update": completed, "mIoU_percent": val["mIoU_percent"]}
                    save_checkpoint(out / "best_val.pt", {"model": model.state_dict(), **best,
                                    "source_sha256": digest, "secondary_only": True})
            if reasons:
                pause_id = str(uuid.uuid4())
                pause_reasons = list(reasons)
                status = "PAUSED"
                persist(status)
                write_json(out / "result.json", {"status": status, "completed": completed,
                    "executed_updates": completed, "checkpoint_persisted_updates": persisted,
                    "pause_id": pause_id, "reasons": reasons, "target_achieved": False,
                    "test_accessed": False, "same_recipe_resume_only": True,
                    "additional_SGD_permitted": completed < 40000,
                    "endpoint_finalize_only": completed == 40000,
                    "endpoint_validation": history[-1] if completed == 40000 else None})
                return
            if completed % 1000 == 0:
                persist("TRAINING")
        if completed != 40000 or not history or history[-1]["update"] != 40000:
            raise RuntimeError("Missing fixed 40k endpoint or endpoint validation")
        persist("COMPLETE")
        final = history[-1]
        passed = final["mIoU_percent"] >= protocol["val_target_percent"]
        lock = {"status": "VAL_TARGET_PASSED" if passed else "VAL_TARGET_NOT_REACHED",
                "source_sha256": digest, "run_identity": run_identity,
                "checkpoint": str((out / "last.pt").resolve()), "checkpoint_sha256": checkpoint_sha,
                "identity_sha256": identity_sha, "completed": completed, "executed_updates": completed,
                "checkpoint_persisted_updates": persisted, "val": final, "test_permitted": passed,
                "val_target_exceeded": final["mIoU_percent"] > protocol["val_target_percent"],
                "test_accessed": False, "primary": "fixed_40000_raw_weights", "best_val_secondary": best}
        write_json(out / "endpoint_lock.json", lock)
        write_json(out / "result.json", {"status": lock["status"], "completed": completed,
                   "target_achieved": False, "val_target_reached": passed,
                   "val_target_exceeded": lock["val_target_exceeded"], "test_accessed": False})
    except BaseException as error:
        failure = {"status": "FAILED", "completed": completed, "executed_updates": completed,
                   "checkpoint_persisted_updates": persisted, "in_flight_update": in_flight,
                   "additional_executed_updates_uncertain": [0, 1] if in_flight is not None else [0, 0],
                   "error_type": type(error).__name__, "error": str(error),
                   "run_identity": run_identity, "test_accessed": False, "resume_forbidden": True}
        write_json(out / "failure.json", failure)
        write_json(out / "progress.json", {**progress_record("FAILED"), "resume_forbidden": True})
        raise
    finally:
        release_run_lease(lease)


if __name__ == "__main__":
    main()
