"""Single-4090 v3 execution: audit gate, separate smoke, exact 40k batch endpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from model import V3Student, build_baseline, file_sha, tensor_state_sha, install_legacy_bf16_interpolation
from objectives import learning_rate, losses, routed_backward

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/root/HRDecoder/data/DDR_split_613")
REPO = ROOT/"vendor/champion_repo"
W48 = Path("/root/.cache/torch/hub/checkpoints/hrnetv2_w48-d2186c55.pth")
PROTOCOL = {"design_version": 3, "seed": 42, "private_seed": 20261008,
    "canvas": 2048, "updates": 40000, "global_batch": 2,
    "optimizer": {"type": "SGD", "lr": .005, "momentum": .9, "weight_decay": .0005},
    "lr": {"type": "poly", "power": 1, "warmup": 1500, "warmup_ratio": 1e-6},
    "lambda_k": [[0, 0], [1500, 0], [4000, .5], [32000, .5], [40000, .1]],
    "lambda_d": ".2 * lambda_k", "precision": "bf16", "validation_interval": 4000,
    "classes": ["EX", "HE", "SE", "MA"], "threshold": "strictly > .5",
    "single_GPU_only": True, "student_TTA": False, "student_teacher_or_H4": False,
    "train_probabilities_in_sample": True, "OOF": False}

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    os.replace(temp, path)

def code_identity():
    return {name: file_sha(HERE/name) for name in
        ("model.py", "private_modules.py", "objectives.py", "data.py", "train.py")}

def identity(mode):
    cpu = json.loads((HERE/"CPU_PREPARATION_PASS.json").read_text())
    real = json.loads((HERE/"REAL_CACHE_PIPELINE_PASS.json").read_text())
    assert cpu["status"] == real["status"] == "PASS"
    server = json.loads((ROOT/"reports/SERVER_INITIALIZATION_PASS.json").read_text())
    assert server["status"] == "PASS" and server["torch"] == torch.__version__
    resume = json.loads((ROOT/"reports/SERVER_OPTIMIZER_RESUME_PASS.json").read_text())
    assert resume["status"] == "PASS" and resume["torch"] == torch.__version__
    route = json.loads((ROOT/"reports/SERVER_REAL_HRNET_ROUTE_PASS.json").read_text())
    assert route["status"] == "PASS" and route["torch"] == torch.__version__
    manifest = json.loads((HERE/"execution_manifest.json").read_text())
    for name, digest in manifest.items():
        assert file_sha(HERE/name) == digest, "Training preparation source SHA mismatch: "+name
    ready = json.loads((ROOT/"CACHE_READY.json").read_text())
    assert ready["status"] == "PASS" and ready["images_per_member"] == 454
    assert ready["all_file_and_tensor_SHA_verified"]
    assert all(c["covariance_rank_relative_tolerance_1e_minus6"] == 2 for c in ready["class_statistics"]), "Degenerate train contrasts need review"
    material = {"protocol": PROTOCOL, "mode": mode, "code": code_identity(),
        "cache_identity": ready["member_identity_sha256"],
        "data_identity_SHA": file_sha(ROOT/"metadata/data_identity.json"),
        "baseline_config_SHA": file_sha(REPO/"configs/lesion/exp/baseline_split613.py"),
        "source_manifest_SHA": file_sha(ROOT/"metadata/source_manifest.json"),
        "ImageNet_W48_SHA": file_sha(W48), "torch": torch.__version__}
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return material, digest

class ResumableBatches(Sampler):
    """Deterministic epoch permutations and per-sample seeds, including worker replay."""
    def __init__(self, n, start=0, stop=40000, seed=42):
        assert n % 2 == 0
        self.n, self.start, self.stop, self.seed = n, start, stop, seed
    def __len__(self): return self.stop-self.start
    def __iter__(self):
        previous_epoch = None
        for update in range(self.start, self.stop):
            epoch = (update*2)//self.n
            if epoch != previous_epoch:
                generator = torch.Generator().manual_seed(self.seed+epoch)
                order = torch.randperm(self.n, generator=generator).tolist()
                previous_epoch = epoch
            offset = (update*2) % self.n
            yield [(order[offset+j], self.seed+update*2+j) for j in range(2)]

def move(batch):
    return {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

def one_step(model, optimizer, scaler, batch, update, mode):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=batch["image"].device.type, dtype=torch.bfloat16,
            enabled=batch["image"].is_cuda):
        output = model(batch["image"], batch["label"], batch["valid"])
    task, auxiliary, metrics = losses(output, batch["label"], batch["valid"],
        batch.get("A"), batch.get("U"), batch.get("S"), update, mode=mode)
    assert torch.isfinite(task) and torch.isfinite(auxiliary)
    routed_backward(task, auxiliary, model.private_parameters(), scale=scaler.get_scale(),
        route="ordinary" if mode == "ordinary_aux" else "private")
    if scaler.is_enabled(): scaler.unscale_(optimizer)
    finite = torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
    if not bool(finite):
        raise FloatingPointError("Nonfinite gradient: stop before consuming a formal SGD update")
    scaler.step(optimizer); scaler.update()
    return {key: float(value.detach().item()) if torch.is_tensor(value) else value for key, value in metrics.items()}

@torch.no_grad()
def evaluate(model, data_root=DATA):
    from data import evaluation_image
    from PIL import Image
    rows = json.loads((ROOT/"metadata/data_identity.json").read_text())["splits"]["val"]["samples"]
    assert len(rows) == 76
    was_training = model.training; model.eval()
    counts = np.zeros((4, 3), dtype=np.int64)
    images = {p.stem: p for p in (data_root/"images/val").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    for row in rows:
        image = evaluation_image(images[row["sid"]]).unsqueeze(0).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): output = model(image)
        logits = F.interpolate(output["logits"].float(), size=row["hw"], mode="bilinear", align_corners=False)
        prediction = (logits.sigmoid() > .5).cpu().numpy()[0]
        with Image.open(data_root/"labels/val"/(row["sid"]+".png")) as source: label = np.asarray(source)
        valid = label != 255
        for c in range(4):
            target = label == c+1; p = prediction[c] & valid
            counts[c] += [int((p & target).sum()), int((p & ~target).sum()), int((~p & target & valid).sum())]
    model.train(was_training)
    tp, fp, fn = counts.T
    iou = np.divide(tp, tp+fp+fn, out=np.full(4, np.nan), where=(tp+fp+fn) > 0)
    precision = np.divide(tp, tp+fp, out=np.zeros(4), where=(tp+fp) > 0)
    recall = np.divide(tp, tp+fn, out=np.zeros(4), where=(tp+fn) > 0)
    return {"mIoU_percent": float(np.nanmean(iou)*100), "IoU_percent": (iou*100).tolist(),
        "precision_percent": (precision*100).tolist(), "recall_percent": (recall*100).tolist(), "counts": counts.tolist()}

def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}

def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])

def save_checkpoint(path, payload):
    temporary = path.with_suffix(".writing")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--train", action="store_true")
    parser.add_argument("--mode", choices=("B0", "B1", "v3", "capacity_kd", "ordinary_aux"), default="v3")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "DDP is not validated; this release supports one 4090 only"
    assert torch.__version__ == "2.0.1+cu117", "Use the pinned legacy environment"
    assert torch.cuda.is_available(), "GPU is unavailable; leave the no-GPU instance for preparation"
    install_legacy_bf16_interpolation()
    from data import TrainCacheDataset
    material, digest = identity(args.mode)
    work = ROOT/"training_runs"/args.mode; work.mkdir(parents=True, exist_ok=True)
    if args.train:
        smoke = json.loads((work/"GPU_SMOKE_PASS.json").read_text())
        assert smoke["identity_SHA"] == digest, "Repeat GPU smoke after source or protocol changes"
        assert args.resume or not (work/"last.pt").exists(), "Use --resume for an existing run"
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    baseline, _ = build_baseline(REPO, W48)
    base_sha = tensor_state_sha(baseline)
    expected_init = json.loads((ROOT/"reports/SERVER_INITIALIZATION_PASS.json").read_text())["baseline_initial_state_SHA"]
    assert base_sha == expected_init, "Seed-42 baseline initialization changed after preparation"
    model = V3Student(baseline, base_only=args.mode in ("B0", "B1")).cuda().train()
    smoke_calls = []
    smoke_hook = model.base.backbone.register_forward_hook(lambda *values: smoke_calls.append(1)) if args.smoke else None
    optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # BF16, no FP16 scaling required.
    completed, best = 0, float("-inf")
    if args.resume:
        checkpoint = torch.load(work/"last.pt", map_location="cpu")
        assert checkpoint["identity_SHA"] == digest
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint["scaler"])
        completed, best = checkpoint["completed"], checkpoint["best"]
        restore_rng(checkpoint["rng"])
        del checkpoint
    dataset = TrainCacheDataset(ROOT, DATA, with_teachers=args.mode != "B0")
    sampler = ResumableBatches(len(dataset), completed, completed+1 if args.smoke else 40000)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=2, pin_memory=True,
        persistent_workers=True, prefetch_factor=1, generator=generator)
    write_json(work/"protocol.json", {"identity_SHA": digest, "identity": material,
        "baseline_initial_state_SHA": base_sha, "historical_B0_initial_snapshot_available": False})
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for batch in loader:
        index = completed
        for group in optimizer.param_groups: group["lr"] = learning_rate(index)
        # Smoke enables the nonzero auxiliary loss; its update is discarded.
        metric = one_step(model, optimizer, scaler, move(batch), 4000 if args.smoke else index+1, args.mode)
        completed += 1
        if args.smoke:
            smoke_hook.remove()
            assert len(smoke_calls) == 1
            if model.private is not None:
                assert torch.count_nonzero(model.private.fusion.final_conv.weight) > 0
            write_json(work/"GPU_SMOKE_PASS.json", {"status": "PASS", "identity_SHA": digest,
                "batch": 2, "canvas": 2048, "single_HRNet_forward": True,
                "measured_backbone_calls": len(smoke_calls), "device": torch.cuda.get_device_name(0),
                "losses": metric, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "seconds": time.monotonic()-started,
                "formal_updates": 0, "probe_optimizer_step_discarded": True})
            print("GPU_SMOKE_PASS", flush=True)
            return
        if completed % 50 == 0 or completed == 1:
            print(json.dumps({"completed": completed, "expected": 40000, "losses": metric,
                "lr": learning_rate(index), "seconds": time.monotonic()-started}), flush=True)
        if completed % 1000 == 0:
            validation = None
            if completed % 4000 == 0:
                validation = evaluate(model)
                write_json(work/("val_%05d.json" % completed), validation)
                if validation["mIoU_percent"] > best:
                    best = validation["mIoU_percent"]
                    save_checkpoint(work/"best_val.pt", {"model": model.state_dict(), "completed": completed,
                        "validation": validation, "identity_SHA": digest})
            save_checkpoint(work/"last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "completed": completed, "best": best,
                "rng": rng_state(), "identity_SHA": digest})
            write_json(work/"progress.json", {"completed": completed, "expected": 40000,
                "best_val_mIoU": best if np.isfinite(best) else None, "last_validation": validation})
    assert completed == 40000
    write_json(work/"TRAINING_COMPLETE.json", {"status": "COMPLETE", "completed": 40000,
        "identity_SHA": digest, "primary_checkpoint": "last.pt", "secondary_checkpoint": "best_val.pt",
        "no_test_selection": True})

if __name__ == "__main__": main()
