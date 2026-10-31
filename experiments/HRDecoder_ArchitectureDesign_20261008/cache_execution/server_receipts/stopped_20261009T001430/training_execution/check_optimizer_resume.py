"""Exercise actual step/checkpoint logic with a tiny CPU graph, no formal updates."""
import copy
import argparse
import json
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from torch import nn
from train import one_step, save_checkpoint, rng_state, restore_rng

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
# This probe intentionally validates the CPU RNGs; CUDA RNG is a GPU-smoke concern.
torch.cuda.get_rng_state_all = lambda: []
torch.cuda.set_rng_state_all = lambda values: None

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Conv2d(3, 4, 1)
        self.local = nn.Conv2d(4, 4, 3, padding=1)
        self.global_path = nn.Conv2d(4, 4, 1)
        self.classifier = nn.Conv2d(8, 4, 1)
    def private_parameters(self): return tuple(self.local.parameters())+tuple(self.global_path.parameters())
    def forward(self, image, label, valid):
        shared = self.shared(image)
        local = self.local(shared); global_value = self.global_path(shared)
        return {"logits": shared+self.classifier(torch.cat((local, global_value), 1)),
            "d_hat_u": local, "d_hat_s": global_value, "hr_loss": shared.sum()*0}

def batch():
    values = {"image": torch.randn(2, 3, 8, 8)*float(np.random.uniform(.8, 1.2)),
        "label": torch.randint(0, 5, (2, 1, 8, 8)), "valid": torch.ones(2, 1, 8, 8)}
    values["valid"][..., :2, :3] = 0
    for member in ("A", "U", "S"): values[member] = torch.rand(2, 4, 8, 8)
    return values

random.seed(42); np.random.seed(42); torch.manual_seed(42)
model = Tiny()
optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
scaler = torch.cuda.amp.GradScaler(enabled=False)
with tempfile.TemporaryDirectory(dir=str(HERE), prefix="resume_probe_") as temporary:
    directory = Path(temporary).resolve()
    directory.relative_to(HERE.resolve())  # Verified workspace-bound cleanup target.
    one_step(model, optimizer, scaler, batch(), 4000, "v3")
    path = directory/"resume.pt"
    save_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "rng": rng_state(), "completed": 1})
    for _ in range(2): one_step(model, optimizer, scaler, batch(), 4000, "v3")
    final_state = copy.deepcopy(model.state_dict())
    final_momentum = [p["momentum_buffer"].clone() for p in optimizer.state.values()]
    next_draw = torch.rand(10)
    resumed = Tiny()
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
    saved = torch.load(path, map_location="cpu")
    resumed.load_state_dict(saved["model"]); resumed_optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"]); restore_rng(saved["rng"])
    for _ in range(2): one_step(resumed, resumed_optimizer, scaler, batch(), 4000, "v3")
    assert all(torch.equal(tensor, resumed.state_dict()[name]) for name, tensor in final_state.items())
    assert all(torch.equal(first, second["momentum_buffer"]) for first, second in zip(final_momentum, resumed_optimizer.state.values()))
    assert torch.equal(next_draw, torch.rand(10))
report = {"status": "PASS", "torch": torch.__version__, "actual_one_step_function_used": True,
    "uninterrupted_vs_resume_parameters_bitwise_equal": True,
    "momentum_bitwise_equal": True, "RNG_replay_bitwise_equal": True,
    "graph": "tiny CPU shared/private/final graph", "real_HRNet": False,
    "formal_updates": 0, "synthetic_optimizer_steps": 5, "GPU_used": False}
parser = argparse.ArgumentParser()
parser.add_argument("--output", default=str(HERE/"OPTIMIZER_RESUME_PASS.json"))
args = parser.parse_args()
Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report), flush=True)
