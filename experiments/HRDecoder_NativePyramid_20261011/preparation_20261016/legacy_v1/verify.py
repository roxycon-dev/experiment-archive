"""Read-only data audit and discarded CPU/CUDA diagnostics before a fresh run."""
from __future__ import annotations
import argparse
import contextlib
import gc
import io
import json
from pathlib import Path
import time
import unittest

import cv2
import numpy as np
import torch

from data import SegmentationData, load_identity, normalize, read_label, ResumableBatches
from losses import SparseLesionLoss
from metrics import accumulate, summarize
from model import NativePyramid, file_sha256
from train import source_identity, write_json, make_optimizer, update_once, seed_all
import tests


def audit_data(root, identity_path, out):
    identity = load_identity(identity_path)
    root = Path(root)
    checks, stats = {}, []
    for split in ("train", "val"):
        images = {p.stem: p for p in (root / "images" / split).iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        for row in identity["splits"][split]["samples"]:
            label_path = root / "labels" / split / (row["sid"] + ".png")
            if file_sha256(images[row["sid"]]) != row["image_sha256"]:
                raise ValueError(f"Image hash mismatch: {split}/{row['sid']}")
            if file_sha256(label_path) != row["label_sha256"]:
                raise ValueError(f"Mask hash mismatch: {split}/{row['sid']}")
            if split == "train":
                label = read_label(label_path)
                counts = np.bincount(label.reshape(-1), minlength=256)[1:5]
                stats.append({"sid": row["sid"], "positive_pixels": counts.tolist(), "pixels": label.size})
        checks[split] = {"images_and_labels_sha256_verified": len(images)}
    # Two images per class, selected from training GT only. Stable tie-breaking.
    selected = []
    for c in range(4):
        ordered = sorted(stats, key=lambda r: (-r["positive_pixels"][c], r["sid"]))
        selected.extend(next(r["sid"] for r in ordered if r["sid"] not in selected) for _ in range(2))
    result = {"status": "PASS", "checks": checks, "train_probe_sids": selected,
              "train_statistics": stats, "test_pixels_read": False,
              "test_metadata_only": True, "identity_sha256": file_sha256(identity_path)}
    write_json(out / "data_audit.json", result)
    return result


def run_units(out):
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    (out / "unit_tests.txt").write_text(stream.getvalue(), encoding="utf-8")
    summary = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors)}
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
            accumulate(counts, (output["logits"][j] > 0).cpu().numpy(), labels[start+j].numpy())
            logits = output["logits"][j].cpu().numpy()
            for c in range(4):
                positive_values[c].append(logits[c][labels[start+j].numpy() == c + 1])
    result = summarize(counts, len(images)); result["loss"] = float(np.mean(losses))
    result["positive_logit_median"] = [float(np.median(np.concatenate(v))) for v in positive_values]
    model.train()
    return result


def fit_probe(pretrained, dataset, protocol, steps, out):
    images, labels, crops = real_patch_bank(dataset)
    seed_all(42)
    model = NativePyramid(pretrained).cuda().train()
    optimizer = make_optimizer(model, protocol)
    criterion = SparseLesionLoss(**protocol["loss"])
    start = patch_metrics(model, images, labels, criterion)
    trajectory = []
    for update, pairs in enumerate(ResumableBatches(8, 0, steps)):
        indices = [i for i, _ in pairs]
        values = update_once(model, optimizer, criterion,
                             {"image": images[indices], "label": labels[indices]}, micro_batch=2)
        if (update + 1) % 16 == 0:
            print(json.dumps({"discarded_crop_probe_update": update + 1, "loss": values["loss"]}), flush=True)
            trajectory.append({"update": update + 1, **patch_metrics(model, images, labels, criterion)})
    finish = patch_metrics(model, images, labels, criterion)
    result = {"status": "PASS" if finish["loss"] < start["loss"] and min(finish["IoU_percent"]) > 0 else "NOT_PASSED",
              "scope": "256x256 fixed REAL TRAINING crops, learnability diagnostic only",
              "not_validation_accuracy": True, "not_formal_40k": True, "discarded_updates": steps,
              "sampling": "same uniform epoch-shuffled sampler as main training, eight crop indices",
              "initial": start, "final": finish, "trajectory": trajectory, "crops": crops,
              "weights_retained": False}
    write_json(out / "real_crop_learning.json", result)
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    return result


def full_resolution_smoke(pretrained, dataset, protocol, out):
    seed_all(42)
    model = NativePyramid(pretrained).cuda().train()
    optimizer = make_optimizer(model, protocol)
    criterion = SparseLesionLoss(**protocol["loss"])
    # Use first two train samples at deterministic original augmentation seeds.
    items = [dataset[(0, 42)], dataset[(1, 43)]]
    batch = {key: torch.stack([item[key] for item in items]) for key in ("image", "label")}
    sids = [item["sid"] for item in items]
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    updates = []
    try:
        for i in range(2):
            values = update_once(model, optimizer, criterion, batch, micro_batch=1)
            torch.cuda.synchronize()
            updates.append(values)
            print(json.dumps({"discarded_2048_smoke_update": i + 1, "loss": values["loss"]}), flush=True)
        result = {"status": "PASS", "canvas": 2048, "global_batch": 2, "micro_batch": 1,
                  "discarded_updates": 2, "mean_seconds_per_update": (time.monotonic() - started) / 2,
                  "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                  "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                  "gpu": torch.cuda.get_device_name(0), "metrics": updates, "train_sids": sids,
                  "weights_retained": False, "not_formal_40k": True}
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "OOM", "completed_discarded_updates": len(updates), "error": str(error),
                  "canvas": 2048, "global_batch": 2, "micro_batch": 1, "gpu": torch.cuda.get_device_name(0),
                  "not_formal_40k": True}
    write_json(out / "full_resolution_smoke.json", result)
    del model, optimizer, batch; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True); parser.add_argument("--identity", required=True)
    parser.add_argument("--pretrained", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--crop-steps", type=int, default=128)
    parser.add_argument("--skip-full-resolution", action="store_true")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    protocol = json.loads((Path(__file__).parent / "protocol.json").read_text())
    digest, files = source_identity()
    record = {"source_sha256": digest, "source_files": files, "formal_updates": 0,
              "test_accessed": False, "torch": torch.__version__}
    try:
        record["unit_tests"] = run_units(out)
        audit = audit_data(args.data_root, args.identity, out)
        record["data_audit"] = {key: value for key, value in audit.items() if key != "train_statistics"}
        record["train_probe_sids"] = audit["train_probe_sids"]
        print(json.dumps({"stage": "units_and_data_pass", "tests": record["unit_tests"]["tests"]}), flush=True)
        seed_all(42)
        model = NativePyramid(args.pretrained).cuda().train()
        record["backbone_integration"] = checkpoint_parity(model, out)
        del model; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"stage": "checkpoint_parity_pass"}), flush=True)
        dataset = SegmentationData(args.data_root, args.identity, "train", 2048)
        record["real_crop_learning"] = fit_probe(args.pretrained, dataset, protocol, args.crop_steps, out)
        if not args.skip_full_resolution:
            record["full_resolution_smoke"] = full_resolution_smoke(args.pretrained, dataset, protocol, out)
        else:
            record["full_resolution_smoke"] = {"status": "NOT_RUN"}
        record["status"] = ("PASS" if all(record[name]["status"] == "PASS" for name in
                                           ("unit_tests", "backbone_integration", "real_crop_learning", "full_resolution_smoke"))
                            else "INCOMPLETE_OR_FAILED")
        write_json(out / "preflight.json", record)
        print(json.dumps({"status": record["status"], "formal_updates": 0, "test_accessed": False}), flush=True)
    except BaseException as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        write_json(out / "preflight.json", record)
        raise


if __name__ == "__main__":
    main()
