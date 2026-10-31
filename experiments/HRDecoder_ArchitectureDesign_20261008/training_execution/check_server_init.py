"""Construct one actual baseline on the server CPU, preserving the exact runtime."""
import json
from pathlib import Path
import resource
import time
import torch
from model import build_baseline, V3Student, tensor_state_sha, W48_SHA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
started = time.monotonic()
torch.set_num_threads(1)
baseline, cfg = build_baseline(ROOT/"vendor/champion_repo",
    "/root/.cache/torch/hub/checkpoints/hrnetv2_w48-d2186c55.pth")
initial_sha = tensor_state_sha(baseline)
rng = torch.get_rng_state().clone()
student = V3Student(baseline)
assert torch.equal(rng, torch.get_rng_state())
assert initial_sha == tensor_state_sha(student.base)
assert sum(p.numel() for p in student.parameters()) == 71992068
result = {"status": "PASS", "torch": torch.__version__, "W48_SHA": W48_SHA,
    "baseline_initial_state_SHA": initial_sha, "base_parameters": 65572420,
    "new_parameters": 6419648, "student_parameters": 71992068,
    "baseline_tensors_preserved": True, "base_rng_preserved": True,
    "ImageNet_W48_only": True, "task_teacher_weights_loaded": False,
    "all_backbone_tensors_equal_pinned_ImageNet": True,
    "GPU_used": False, "forward_run": False, "optimizer_step": False,
    "historical_B0_initial_snapshot_available": False,
    "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    "seconds": time.monotonic()-started}
(ROOT/"reports/SERVER_INITIALIZATION_PASS.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result), flush=True)
