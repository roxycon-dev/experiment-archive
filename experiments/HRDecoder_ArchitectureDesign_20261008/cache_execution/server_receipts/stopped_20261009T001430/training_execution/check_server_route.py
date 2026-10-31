"""Actual pinned-runtime HRNet gradient routing on small CPU tensors."""
import json
from pathlib import Path
import resource
import time
import torch
from model import build_baseline, V3Student
from objectives import masked_dice, masked_kl, signed_aux, routed_backward

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
started = time.monotonic()
torch.set_num_threads(1)
base, _ = build_baseline(ROOT/"vendor/champion_repo",
    "/root/.cache/torch/hub/checkpoints/hrnetv2_w48-d2186c55.pth", canvas=128)
model = V3Student(base, canvas=128).cpu().train()
with torch.no_grad(): model.private.fusion.final_conv.weight.normal_(0, .02)
image = torch.randn(2, 3, 128, 128)
label = torch.randint(0, 5, (2, 1, 128, 128))
valid = torch.ones(2, 1, 128, 128); valid[..., :11, :7] = 0
output = model(image, label, valid)
logits = output["logits"]-output["base_logits"]+output["base_logits"].detach()
teacher = torch.rand_like(logits)
task = masked_dice(logits, label, valid)+.5*masked_kl(logits, teacher, valid)
aux = .1*signed_aux([output["d_hat_u"], output["d_hat_s"]],
    [torch.rand_like(logits)*2-1, torch.rand_like(logits)*2-1], valid)
shared = model.base.backbone.conv1.weight
classifier = model.private.fusion.project.conv.weight
private_l = model.private.local.p_r.conv.weight
private_g = model.private.global_path.p_f34.conv.weight
selected = (shared, classifier, private_l, private_g)
task_grad = torch.autograd.grad(task, selected, retain_graph=True)
aux_private = torch.autograd.grad(aux, (private_l, private_g), retain_graph=True)
raw_aux_shared = torch.autograd.grad(aux, shared, retain_graph=True)[0]
assert task_grad[0].abs().sum() > 0 and raw_aux_shared.abs().sum() > 0
routed_backward(task, aux, model.private_parameters(), scale=32.)
expected = [task_grad[0], task_grad[1], task_grad[2]+aux_private[0], task_grad[3]+aux_private[1]]
differences = [float((parameter.grad/32-reference).abs().max()) for parameter, reference in zip(selected, expected)]
assert max(differences) < 1e-6
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
report = {"status": "PASS", "torch": torch.__version__, "real_HRNet": True,
    "canvas": 128, "batch": 2, "auxiliary_private_only": True,
    "shared_and_classifier_match_task_only": True, "private_matches_task_plus_auxiliary": True,
    "shared_task_gradient_through_private_nonzero": True,
    "unrouted_auxiliary_shared_gradient_nonzero": True, "scale": 32.,
    "max_abs_difference_by_group": differences,
    "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    "seconds": time.monotonic()-started, "GPU_used": False, "optimizer_step": False,
    "diagnostic_final_layer_modified_only_in_memory": True}
(ROOT/"reports/SERVER_REAL_HRNET_ROUTE_PASS.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report), flush=True)
