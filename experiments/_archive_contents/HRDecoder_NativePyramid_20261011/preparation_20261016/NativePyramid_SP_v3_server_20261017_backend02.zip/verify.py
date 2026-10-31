"""Read-only data audit and discarded CPU/CUDA diagnostics before a fresh run."""
from __future__ import annotations
import argparse
import contextlib
import gc
import io
import json
import os
from pathlib import Path
import time
import unittest

import cv2
import numpy as np
import torch

from data import (SegmentationData, load_identity, normalize, read_label, ResumableBatches,
                  verify_training_files, IDENTITY_SHA)
from losses import SparseLesionLoss
from metrics import accumulate, summarize
from model import NativePyramid, file_sha256
from train import (source_identity, write_json, make_optimizer, update_once, seed_all,
                   runtime_identity, evaluate, configure_runtime)
import tests


def audit_data(root, identity_path, out):
    identity = load_identity(identity_path)
    root = Path(root)
    checks, stats = verify_training_files(root, identity_path), []
    for split in ("train", "val"):
        for row in identity["splits"][split]["samples"]:
            label = read_label(root / "labels" / split / (row["sid"] + ".png"))
            if list(label.shape) != row["hw"]:
                raise ValueError("Native label geometry differs from identity")
            if split == "train":
                counts = np.bincount(label.reshape(-1), minlength=256)[1:5]
                stats.append({"sid": row["sid"], "positive_pixels": counts.tolist(), "pixels": label.size})
    # Two images per class, selected from training GT only. Stable tie-breaking.
    selected = []
    for c in range(4):
        ordered = sorted((r for r in stats if r["positive_pixels"][c] > 0),
                         key=lambda r: (-r["positive_pixels"][c], r["sid"]))
        candidates = [r["sid"] for r in ordered if r["sid"] not in selected]
        if len(candidates) < 2:
            raise RuntimeError(f"Cannot freeze two distinct positive training images for class {c+1}")
        selected.extend(candidates[:2])
    coverage = [sum(int(next(r for r in stats if r["sid"] == sid)["positive_pixels"][c] > 0)
                    for sid in selected) for c in range(4)]
    if len(selected) != 8 or min(coverage) < 2:
        raise RuntimeError("Fixed probe bank must cover each class with at least two images")
    result = {"status": "PASS", "checks": checks, "train_probe_sids": selected,
              "train_statistics": stats, "test_pixels_read": False,
              "test_metadata_only": True, "identity_sha256": file_sha256(identity_path),
              "positive_image_coverage": coverage}
    write_json(out / "data_audit.json", result)
    return result


def run_units(out):
    stream = io.StringIO()
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests)])
    suite.addTests(unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*v3.py"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (out / "unit_tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    summary = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped)}
    write_json(out / "unit_tests.json", summary)
    if not result.wasSuccessful():
        raise RuntimeError(stream.getvalue())
    return summary


def checkpoint_parity(model, out):
    model.train()
    selected = ["backbone.net.conv1.weight", "backbone.net.stage2.0.branches.0.0.conv1.weight",
                "backbone.net.stage4.2.branches.3.3.conv2.weight", "decoder.readout.weight"]
    image = torch.randn(1, 3, 128, 128, device="cuda")
    label = torch.randint(0, 5, (1, 128, 128), device="cuda")
    criterion = SparseLesionLoss(min_negatives=16)
    buffers = {n: v.clone() for n, v in model.named_buffers()}
    records = []
    reference = None
    for enabled in (False, True):
        model.backbone.checkpointing = enabled
        model.decoder.checkpointing = enabled
        model.zero_grad(set_to_none=True)
        output = model(image)
        loss, _ = criterion(output, label); loss.backward()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if n in selected}
        missing = [n for n, p in model.named_parameters() if p.grad is None or not bool(torch.isfinite(p.grad).all())]
        if missing:
            raise RuntimeError(f"Untrained/nonfinite parameter path: {missing[:8]}")
        current = {"logits": output["logits"].detach().cpu(), "gradients": gradients}
        if reference is not None:
            torch.testing.assert_close(current["logits"], reference["logits"], rtol=1e-5, atol=1e-6)
            for name in selected:
                torch.testing.assert_close(gradients[name], reference["gradients"][name], rtol=2e-4, atol=2e-6)
        reference = current
        records.append({"checkpointing": enabled, "loss": float(loss.detach()),
                        "nonzero_gradient_parameters": sum(int(p.grad.count_nonzero() > 0) for p in model.parameters())})
        del output, loss
    if not all(torch.equal(buffers[name], value) for name, value in model.named_buffers()):
        raise RuntimeError("Frozen BN buffers changed across forward/backward")
    model.zero_grad(set_to_none=True)
    result = {"status": "PASS", "scope": "full W48, FP32 CUDA, 128 grid; no SGD", "records": records,
              "selected_gradient_parity": selected, "all_BN_buffers_unchanged": True,
              "initialization": model.backbone.initialization,
              "parameters": sum(p.numel() for p in model.parameters()),
              "decoder_parameters": sum(p.numel() for p in model.decoder.parameters())}
    write_json(out / "backbone_integration.json", result)
    return result


def precision_micro_parity(model, out):
    """W48: complete-batch loss vs accumulated loss on per-image forwards.

    The model always schedules convolution with physical batch one, including
    the B=2 reference. This tests one complete-loss backward vs two micro-loss
    backwards, not native cuDNN B=2 vs B=1 convolution kernels.
    """
    image = torch.randn(2, 3, 128, 128, device="cuda")
    label = torch.zeros(2, 128, 128, dtype=torch.long, device="cuda")
    label[0, :4, :4] = 1; label[0, 4:8, :4] = 2
    label[1, :32, :32] = 3; label[1, 32:48, :32] = 4; label[:, -8:] = 255
    criterion = SparseLesionLoss()
    denominators = criterion.batch_denominators(label)
    buffers = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    records = []
    # BF16 may use different convolution reductions for B=1/B=2. Tolerances
    # are locked before execution and are reported, never widened after failure.
    for fixture in ("zero_residual_initialization", "deterministic_nonzero_residual_connectivity_fixture"):
        if fixture.startswith("deterministic"):
            # This is a synthetic connectivity fixture, never task pretraining.
            # The model is discarded after parity and no optimizer is used.
            with torch.no_grad():
                for p in (model.decoder.auxiliary.weight, model.decoder.readout.weight):
                    p.copy_(torch.linspace(-.01, .01, p.numel(), device=p.device).reshape_as(p))
        for amp in (False, True):
            tolerance = {"rtol": .03, "atol": .0005} if amp else {"rtol": .0003, "atol": .000003}
            reference = None
            for checkpointing, micro in ((False, 2), (True, 2), (True, 1)):
                model.backbone.checkpointing = model.decoder.checkpointing = checkpointing
                model.zero_grad(set_to_none=True)
                pieces, semantic_pieces, total = [], [], 0.
                for j in range(0, 2, micro):
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        output = model(image[j:j+micro])
                    loss, _ = criterion(output, label[j:j+micro], denominators=denominators)
                    loss.backward()
                    pieces.append(output["logits"].detach().cpu())
                    semantic_pieces.append(output["auxiliary_logits"].detach().cpu())
                    total += float(loss.detach())
                    del output, loss
                current, semantic = torch.cat(pieces), torch.cat(semantic_pieces)
                for n, p in model.named_parameters():
                    if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                        raise RuntimeError(f"Missing/nonfinite parity gradient: {n}")
                if reference is None:
                    reference = {"logits": current, "semantic": semantic, "loss": torch.tensor(total),
                                 "gradients": {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}}
                else:
                    torch.testing.assert_close(current, reference["logits"], **tolerance)
                    torch.testing.assert_close(semantic, reference["semantic"], **tolerance)
                    torch.testing.assert_close(torch.tensor(total), reference["loss"], **tolerance)
                    for n, p in model.named_parameters():
                        try:
                            torch.testing.assert_close(p.grad.detach().cpu(), reference["gradients"][n], **tolerance)
                        except AssertionError as error:
                            raise AssertionError(f"{n}, fixture={fixture}, amp={amp}, checkpointing={checkpointing}, micro={micro}: {error}") from error
                records.append({"fixture": fixture, "BF16": amp, "checkpointing": checkpointing, "micro_batch": micro,
                                "physical_forward_batch": 1, "loss_batch": micro,
                                "backward_calls": 2 // micro,
                                "full_parameter_gradient_comparison": True, "S_Z_loss_comparison": True,
                                "tolerance": tolerance, "loss": total})
            del reference
    with torch.no_grad():
        model.decoder.auxiliary.weight.zero_(); model.decoder.readout.weight.zero_()
    if not all(torch.equal(b.detach().cpu(), buffers[n]) for n, b in model.named_buffers()):
        raise RuntimeError("BN buffers changed in precision/microbatch parity")
    model.zero_grad(set_to_none=True)
    result = {"status": "PASS", "canvas": 128, "global_batch": 2, "SGD_updates": 0,
              "model_forward_schedule": "per_image_forward_then_concatenate",
              "native_physical_B2_convolution_parity_claimed": False,
              "all_BN_buffers_unchanged": True, "records": records}
    write_json(out / "precision_micro_parity.json", result)
    return result


def real_patch_bank(dataset):
    # Eight real training crops, two per class. No synthetic target, no test/val.
    records, patches, labels = [], [], []
    used = set()
    for c in range(1, 5):
        found = 0
        for index in range(len(dataset)):
            if index in used:
                continue
            image, label, row = dataset.native(index)
            ys, xs = np.where(label == c)
            if len(ys) < 20:
                continue
            # Densest occupied 128x128 bin; choose a centered 256 crop around it.
            codes = (ys // 128) * (label.shape[1] // 128 + 1) + xs // 128
            code = int(np.bincount(codes).argmax())
            cy, cx = divmod(code, label.shape[1] // 128 + 1)
            y = max(0, min(label.shape[0] - 256, cy * 128 + 64 - 128))
            x = max(0, min(label.shape[1] - 256, cx * 128 + 64 - 128))
            patch_label = label[y:y+256, x:x+256].copy()
            if np.count_nonzero(patch_label == c) < 8:
                continue
            patches.append(normalize(image[y:y+256, x:x+256]))
            labels.append(torch.from_numpy(patch_label.astype(np.int64)))
            records.append({"sid": row["sid"], "target_class": c, "crop_yxhw": [y, x, 256, 256],
                            "class_positive_pixels": [int(np.count_nonzero(patch_label == j)) for j in range(1, 5)]})
            used.add(index); found += 1
            if found == 2:
                break
        if found < 2:
            raise RuntimeError("Could not construct all-class real training diagnostic")
    return torch.stack(patches), torch.stack(labels), records


@torch.no_grad()
def patch_metrics(model, images, labels, criterion):
    model.eval()
    counts = np.zeros((4, 3), np.int64); losses = []
    positive_values = [[] for _ in range(4)]
    for start in range(0, len(images), 2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(images[start:start+2].cuda())
        loss, _ = criterion(output, labels[start:start+2].cuda()); losses.append(float(loss))
        for j in range(len(output["logits"])):
            accumulate(counts, (output["logits"][j].float().sigmoid() > .5).cpu().numpy(), labels[start+j].numpy())
            logits = output["logits"][j].cpu().numpy()
            for c in range(4):
                positive_values[c].append(logits[c][labels[start+j].numpy() == c + 1])
    result = summarize(counts, len(images)); result["loss"] = float(np.mean(losses))
    result["positive_logit_median"] = [float(np.median(np.concatenate(v))) for v in positive_values]
    model.train()
    return result


def fit_probe(pretrained, dataset, protocol, steps, out):
    if steps != 256:
        raise ValueError("The locked crop gate is exactly 256 updates; nodes 224 and 256")
    images, labels, crops = real_patch_bank(dataset)
    write_json(out / "frozen_crop_bank.json", {"crops": crops, "augmentation": False,
               "source_sha256": source_identity()[0], "identity_sha256": IDENTITY_SHA})
    seed_all(42)
    model = NativePyramid(pretrained).cuda().train()
    optimizer = make_optimizer(model, protocol)
    criterion = SparseLesionLoss(**protocol["loss"])
    start = patch_metrics(model, images, labels, criterion)
    trajectory, step_losses = [], []
    for update, pairs in enumerate(ResumableBatches(8, 0, steps)):
        indices = [i for i, _ in pairs]
        write_json(out / "diagnostic_progress.json", {"stage": "crop_learning", "completed_updates": update,
                   "in_flight_update": update+1, "weights_discarded": True, "formal_updates": 0})
        values = update_once(model, optimizer, criterion,
                             {"image": images[indices], "label": labels[indices]}, micro_batch=2)
        step_losses.append(values["loss"])
        write_json(out / "diagnostic_progress.json", {"stage": "crop_learning", "completed_updates": update+1,
                   "in_flight_update": None, "weights_discarded": True, "formal_updates": 0})
        if (update + 1) % 16 == 0:
            print(json.dumps({"discarded_crop_probe_update": update + 1, "loss": values["loss"]}), flush=True)
            trajectory.append({"update": update + 1, **patch_metrics(model, images, labels, criterion)})
    finish = patch_metrics(model, images, labels, criterion)
    gates = [r for r in trajectory if r["update"] in (224, 256)]
    first_mean, last_mean = float(np.mean(step_losses[:16])), float(np.mean(step_losses[-16:]))
    passed = len(gates) == 2 and all(min(r["IoU_percent"]) >= 20 for r in gates) and last_mean < first_mean
    result = {"status": "PASS" if passed else "NOT_PASSED",
              "scope": "256x256 fixed REAL TRAINING crops, learnability diagnostic only",
              "not_validation_accuracy": True, "not_formal_40k": True, "discarded_updates": steps,
              "sampling": "same uniform epoch-shuffled sampler as main training, eight crop indices",
              "initial": start, "final": finish, "trajectory": trajectory, "crops": crops,
              "required_nodes": [224, 256], "required_each_class_iou_percent": 20.,
              "step_losses": step_losses, "first_16_loss_mean": first_mean, "last_16_loss_mean": last_mean,
              "weights_retained": False}
    write_json(out / "real_crop_learning.json", result)
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    return result


def full_resolution_smoke(pretrained, dataset, protocol, out, probe_sids):
    seed_all(42)
    model = NativePyramid(pretrained).cuda().train()
    optimizer = make_optimizer(model, protocol)
    criterion = SparseLesionLoss(**protocol["loss"])
    # Freeze eight augmented real images before SGD. Two images per class
    # must retain that class on the actual 2048 label grid.
    order = {r["sid"]: i for i, r in enumerate(dataset.rows)}
    items, bank = [], []
    for j, sid in enumerate(probe_sids):
        target_class = j // 2 + 1
        for attempt in range(100):
            seed = 42 + j * 100 + attempt
            item = dataset[(order[sid], seed)]
            if int((item["label"] == target_class).sum()) > 0:
                items.append(item)
                bank.append({"sid": sid, "augmentation_seed": seed, "target_class": target_class,
                             "positive_pixels": [int((item["label"] == c).sum()) for c in range(1, 5)]})
                break
        else:
            raise RuntimeError("Cannot retain all four classes in the fixed real-2048 diagnostic")
    if len(items) != 8 or any(sum(r["positive_pixels"][c] > 0 for r in bank) < 2 for c in range(4)):
        raise RuntimeError("Real-2048 bank lacks two positive images for every class")
    write_json(out / "frozen_full_resolution_bank.json", {"samples": bank})
    bn = {n: v.detach().cpu().clone() for n, v in model.named_buffers()}
    selected_names = ("decoder.deep.0.0.weight", "decoder.fusions.2.0.0.weight",
                      "decoder.stem.0.weight", "decoder.detail.0.0.weight",
                      "decoder.auxiliary.weight", "decoder.readout.weight",
                      "backbone.net.bn1.weight", "backbone.net.bn1.bias")
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in selected_names}
    if len(initial) != len(selected_names):
        raise RuntimeError("Gradient/parameter diagnostic keys no longer match v3")
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    updates = []
    try:
        for i in range(8):
            pair = items[(i % 4) * 2:(i % 4) * 2 + 2]
            batch = {key: torch.stack([item[key] for item in pair]) for key in ("image", "label")}
            write_json(out / "diagnostic_progress.json", {"stage": "real_2048", "completed_updates": i,
                       "in_flight_update": i+1, "crop_updates": 256, "weights_discarded": True, "formal_updates": 0})
            values = update_once(model, optimizer, criterion, batch, micro_batch=1)
            torch.cuda.synchronize()
            updates.append(values)
            write_json(out / "diagnostic_progress.json", {"stage": "real_2048", "completed_updates": i+1,
                       "in_flight_update": None, "crop_updates": 256, "weights_discarded": True, "formal_updates": 0})
            print(json.dumps({"discarded_2048_smoke_update": i + 1, "loss": values["loss"]}), flush=True)
        training_seconds = time.monotonic() - started
        training_peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        training_peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        first = updates[0]["gradients_before_clip"]["modules"]
        late = [v["gradients_before_clip"]["modules"] for v in updates[1:]]
        first_required = ("decoder.auxiliary", "decoder.readout", "decoder.direct_classifiers",
                          "backbone.net.conv1", "backbone.net.bn1")
        late_required = ("decoder.deep", "decoder.laterals", "decoder.fusions", "decoder.stem", "decoder.detail")
        if any(first.get(n, 0.) <= 0 for n in first_required):
            raise RuntimeError("C/D last layers or direct backbone path has no first-step task gradient")
        if any(not any(r.get(n, 0.) > 0 for r in late) for n in late_required):
            raise RuntimeError("A residual internal branch remains without task gradient after zero heads update")
        # A live sibling must not hide a dead lateral/fusion/refinement module.
        required_parameter_modules = sorted({n.rsplit(".", 1)[0] for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("decoder.") and
            not n.startswith(("decoder.direct_", "decoder.auxiliary.", "decoder.readout."))})
        late_parameter_modules = [v["gradients_before_clip"]["parameter_modules"] for v in updates[1:]]
        active_internal = {n: any(r.get(n, 0.) > 0 for r in late_parameter_modules)
                           for n in required_parameter_modules}
        if not active_internal or not all(active_internal.values()):
            raise RuntimeError("Residual parameter modules without task gradients: " +
                               str([n for n, active in active_internal.items() if not active]))
        if any(v["gradients_before_clip"]["missing_gradient_parameter_modules"] for v in updates):
            raise RuntimeError("A trainable parameter module has missing task gradients")
        changes = {n: float((p.detach().cpu() - initial[n]).abs().max())
                   for n, p in model.named_parameters() if n in initial}
        if any(value == 0 for value in changes.values()):
            raise RuntimeError("A monitored decoder parameter did not change")
        if not all(torch.equal(v.detach().cpu(), bn[n]) for n, v in model.named_buffers()):
            raise RuntimeError("Frozen backbone BN buffers changed during real-2048 updates")
        response = evaluate(model, dataset, [order[sid] for sid in probe_sids], training_probe=True)
        result = {"status": "PASS", "canvas": 2048, "global_batch": 2, "micro_batch": 1,
                  "discarded_updates": 8, "mean_seconds_per_update": training_seconds / 8,
                  "SGD_seconds": training_seconds, "probe_seconds": response["seconds"],
                  "peak_allocated_GiB": training_peak_allocated,
                  "peak_reserved_GiB": training_peak_reserved,
                  "gpu": torch.cuda.get_device_name(0), "metrics": updates, "frozen_bank": bank,
                  "all_BN_buffers_unchanged": True, "monitored_parameter_max_change": changes,
                  "internal_parameter_module_gradient_gate": {"status": "PASS", "active_modules": active_internal},
                  "per_class_response": response,
                  "weights_retained": False, "not_formal_40k": True}
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "OOM", "completed_discarded_updates": len(updates), "error": str(error),
                  "canvas": 2048, "global_batch": 2, "micro_batch": 1, "gpu": torch.cuda.get_device_name(0),
                  "not_formal_40k": True}
    write_json(out / "full_resolution_smoke.json", result)
    del model, optimizer, items; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root"); parser.add_argument("--identity")
    parser.add_argument("--pretrained"); parser.add_argument("--output", required=True)
    parser.add_argument("--static-only", action="store_true", help="No real-data SGD; cannot grant training PASS")
    parser.add_argument("--crop-steps", type=int, choices=(256,), default=256)
    parser.add_argument("--skip-full-resolution", action="store_true")
    args = parser.parse_args()
    if args.pretrained:
        os.environ["NATIVE_PYRAMID_PRETRAINED"] = args.pretrained
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("A diagnostic attempt needs a fresh output directory; never overwrite old evidence")
    torch.set_num_threads(4); cv2.setNumThreads(1)
    configure_runtime()
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text(encoding="utf-8"))
    digest, files = source_identity()
    record = {"version": 3, "protocol_version": 3, "source_sha256": digest, "source_files": files,
              "formal_updates": 0, "test_accessed": False, "environment": runtime_identity(),
              "mode": "static_only" if args.static_only else "server_preflight"}
    try:
        record["unit_tests"] = run_units(out)
        if args.static_only:
            record.update(status="STATIC_CHECKS_PASSED_SERVER_GATES_PENDING", discarded_real_data_updates=0,
                          architecture="NativePyramid-SP_v3", server_gates_completed=False)
            write_json(out / "static_checks.json", record)
            print(json.dumps({"status": record["status"], "tests": record["unit_tests"]["tests"]}), flush=True)
            return
        if not all((args.data_root, args.identity, args.pretrained)):
            raise ValueError("Server preflight requires --data-root --identity --pretrained")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Server preflight requires CUDA BF16; no silent precision fallback")
        if record["unit_tests"]["skipped"]:
            raise RuntimeError("Server preflight must execute every structural/unit check without skips")
        audit = audit_data(args.data_root, args.identity, out)
        record["data_audit"] = {key: value for key, value in audit.items() if key != "train_statistics"}
        record["train_probe_sids"] = audit["train_probe_sids"]
        print(json.dumps({"stage": "units_and_data_pass", "tests": record["unit_tests"]["tests"]}), flush=True)
        seed_all(42)
        model = NativePyramid(args.pretrained).cuda().train()
        record["backbone_integration"] = checkpoint_parity(model, out)
        record["precision_micro_parity"] = precision_micro_parity(model, out)
        del model; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"stage": "checkpoint_parity_pass"}), flush=True)
        dataset = SegmentationData(args.data_root, args.identity, "train", 2048)
        record["real_crop_learning"] = fit_probe(args.pretrained, dataset, protocol, args.crop_steps, out)
        if record["real_crop_learning"]["status"] == "PASS" and not args.skip_full_resolution:
            record["full_resolution_smoke"] = full_resolution_smoke(args.pretrained, dataset, protocol, out,
                                                                  audit["train_probe_sids"])
        else:
            record["full_resolution_smoke"] = {"status": "NOT_RUN", "reason": "crop_gate_failed_or_explicit_skip"}
        record["status"] = ("PASS" if all(record[name]["status"] == "PASS" for name in
                                           ("unit_tests", "backbone_integration", "precision_micro_parity",
                                            "real_crop_learning", "full_resolution_smoke"))
                            else "INCOMPLETE_OR_FAILED")
        write_json(out / "preflight.json", record)
        print(json.dumps({"status": record["status"], "formal_updates": 0, "test_accessed": False}), flush=True)
    except BaseException as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        if (out / "diagnostic_progress.json").is_file():
            last = json.loads((out / "diagnostic_progress.json").read_text(encoding="utf-8"))
            record["diagnostic_progress_at_failure"] = last
            record["additional_diagnostic_updates_uncertain"] = [0, 1] if last.get("in_flight_update") else [0, 0]
        write_json(out / "preflight.json", record)
        raise


if __name__ == "__main__":
    main()
