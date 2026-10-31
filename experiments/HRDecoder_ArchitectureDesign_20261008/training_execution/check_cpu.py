"""Actual HRNet integration, exact baseline BN sequence, geometry and routing probes."""
import copy
import json
import math
import random
import sys
from pathlib import Path
import time
import types
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"python_deps"))
REPO = HERE.parents[1]/"HRDecoder_Ready_20261002/vendor/champion_repo"
W48 = HERE.parents[1]/"HRDecoder_Ready_20261002/assets/hrnetv2_w48-d2186c55.pth"
sys.path.insert(0, str(REPO))
from model import build_baseline, V3Student, tensor_state_sha, file_sha
from objectives import masked_dice, masked_kl, signed_aux, routed_backward, lambda_k, learning_rate, losses
from data import training_pipeline, augment, SoftResize, SoftRotate
from mmseg.datasets.pipelines.transforms import Resize, RandomCrop, RandomFlip, RandomRotate, Normalize, Pad
from data import MEAN, STD
from train import ResumableBatches

torch.set_num_threads(4)
checks = {}
def check(name, condition, detail=None):
    assert bool(condition), (name, detail)
    checks[name] = True if detail is None else detail
    print(json.dumps({"check": name, "PASS": True}), flush=True)

def geometry_checks():
    rng = np.random.RandomState(7)
    image = rng.uniform(0, 255, (147, 203, 3)).astype(np.float32)
    label = rng.randint(0, 5, (147, 203)).astype(np.uint8)
    label[:13, :21] = 255
    prob = np.stack([image[..., 0]/255, image[..., 1]/255, image[..., 2]/255,
        np.ones(label.shape, np.float32)])
    old = [Resize(img_scale=(128, 128), ratio_range=(.5, 2.), keep_ratio=True),
        RandomCrop(crop_size=(128, 128), cat_max_ratio=.75), RandomFlip(flip_ratio=.5),
        RandomRotate(prob=1., degree=(-45, 45), pad_val=0, seg_pad_val=0, auto_bound=False),
        Normalize(mean=MEAN, std=STD, to_rgb=True), Pad(size=(128, 128), pad_val=0, seg_pad_val=0)]
    maximum = 0.
    for seed in range(20):
        np.random.seed(seed)
        original = {"img": image.copy(), "gt_semantic_seg": label.copy(),
            "seg_fields": ["gt_semantic_seg"], "img_shape": image.shape}
        for transform in old: original = transform(original)
        original_state = np.random.get_state()
        np.random.seed(seed)
        tensors, replay = augment(image.copy(), label.copy(), {m: prob.copy() for m in ("A", "U", "S")}, training_pipeline(128))
        replay_state = np.random.get_state()
        assert np.array_equal(original["img"], replay["img"])
        assert np.array_equal(original["gt_semantic_seg"], replay["gt_semantic_seg"])
        assert np.array_equal(original_state[1], replay_state[1]) and original_state[2:] == replay_state[2:]
        np.testing.assert_array_equal(tensors["A"].numpy(), tensors["U"].numpy())
        np.testing.assert_array_equal(tensors["A"].numpy(), tensors["S"].numpy())
        # Unnormalize red-channel RGB input; compare the replayed float soft map.
        restored_b = (replay["img"][..., 2]*STD[2]+MEAN[2])/255
        geo = replay["v_geo"] >= 1-1e-6
        maximum = max(maximum, float(np.max(np.abs(restored_b[geo]-tensors["A"][0].numpy()[geo]))))
        assert not tensors["valid"].numpy()[0][replay["v_geo"] < 1-1e-6].any()
        np.testing.assert_array_equal(tensors["A"][3].numpy(), replay["v_geo"])
        assert not tensors["valid"].numpy()[0][replay["v_label"] == 0].any()
    check("original_image_label_and_RNG_replay_20_seeds", True)
    check("three_soft_members_same_bilinear_geometry", maximum < 1e-6, {"max_abs": maximum})
    check("void_rotation_and_padding_masks_applied", True)
    black_pipeline = [SoftResize(img_scale=(128, 128), ratio_range=(1., 1.)),
        SoftRotate(prob=1., degree=(0., 0.), pad_val=0, seg_pad_val=0),
        Normalize(mean=MEAN, std=STD, to_rgb=True), Pad(size=(128, 128), pad_val=0, seg_pad_val=0)]
    black, _ = augment(np.zeros((128, 128, 3), np.uint8), np.zeros((128, 128), np.uint8),
        {m: np.zeros((4, 128, 128), np.float32) for m in ("A", "U", "S")}, black_pipeline)
    check("real_black_background_remains_valid", black["valid"].sum() == 128*128)

def integration_checks():
    started = time.monotonic()
    baseline, _ = build_baseline(REPO, W48, canvas=128)
    initial_sha = tensor_state_sha(baseline)
    baseline_rng = torch.get_rng_state().clone()
    model = V3Student(baseline, canvas=128)
    check("private_initialization_preserves_baseline_rng", torch.equal(baseline_rng, torch.get_rng_state()))
    check("private_initialization_preserves_all_base_tensors", tensor_state_sha(model.base) == initial_sha)
    check("reviewed_new_parameters", sum(p.numel() for p in model.private.parameters()) == 6419648)
    reference = copy.deepcopy(baseline)
    captured = {}
    def capture(self, logits, image, semantics):
        captured["logits"] = logits
        return logits
    reference.refine_logits = types.MethodType(capture, reference)
    torch.manual_seed(71)
    image = torch.randn(2, 3, 128, 128)
    label = torch.randint(0, 5, (2, 1, 128, 128))
    valid = torch.ones(2, 1, 128, 128)
    np.random.seed(42)
    reference.train()
    original_losses = reference.forward_train(image, [], label)
    crop_rng_end = np.random.get_state()
    calls = []
    handle = model.base.backbone.register_forward_hook(lambda *args: calls.append(1))
    model.train(); np.random.seed(42)
    output = model(image, label, valid)
    handle.remove()
    check("actual_HRNet_called_once", len(calls) == 1)
    check("zero_residual_exact_original_train_logits", torch.equal(output["logits"], captured["logits"]),
        {"max_abs": float((output["logits"]-captured["logits"]).abs().max())})
    bn_names = [name for name in reference.state_dict() if "running_" in name or "num_batches_tracked" in name]
    check("all_original_BN_buffers_and_decode_order_exact", all(torch.equal(reference.state_dict()[name], model.base.state_dict()[name]) for name in bn_names), {"BN_buffers": len(bn_names)})
    check("HR_crop_random_draws_exact", np.array_equal(crop_rng_end[1], np.random.get_state()[1]) and crop_rng_end[2:] == np.random.get_state()[2:])
    check("original_HR_crop_dice_weight_point1", torch.allclose(.1*output["hr_loss"], original_losses["hr.loss_seg"], atol=1e-7))
    check("original_squared_dice_smooth_matches_valid_domain", torch.allclose(masked_dice(output["logits"], label, valid), original_losses["fuse.loss_seg"], atol=1e-7))
    del reference, original_losses, captured, output
    model.zero_grad(set_to_none=True)
    # A nonzero diagnostic final layer makes task gradients reach private paths.
    with torch.no_grad(): model.private.fusion.final_conv.weight.normal_(0, .02)
    output = model(image, label, valid)
    logits_via_private = output["logits"]-output["base_logits"]+output["base_logits"].detach()
    task = masked_dice(logits_via_private, label, valid)
    targets = [torch.rand_like(output["d_hat_u"])*2-1, torch.rand_like(output["d_hat_s"])*2-1]
    auxiliary = .1*signed_aux([output["d_hat_u"], output["d_hat_s"]], targets, valid)
    shared = model.base.backbone.conv1.weight
    classifier = model.private.fusion.project.conv.weight
    local = model.private.local.p_r.conv.weight
    global_parameter = model.private.global_path.p_f34.conv.weight
    selected = (shared, classifier, local, global_parameter)
    task_grad = torch.autograd.grad(task, selected, retain_graph=True)
    aux_grad = torch.autograd.grad(auxiliary, (local, global_parameter), retain_graph=True)
    raw_shared_aux = torch.autograd.grad(auxiliary, shared, retain_graph=True)[0]
    check("task_via_private_reaches_real_HRNet", task_grad[0].abs().sum() > 0)
    check("unrouted_aux_would_reach_real_HRNet", raw_shared_aux.abs().sum() > 0)
    routed_backward(task, auxiliary, model.private_parameters(), scale=32.)
    expected = [task_grad[0], task_grad[1], task_grad[2]+aux_grad[0], task_grad[3]+aux_grad[1]]
    differences = [float((parameter.grad/32-reference).abs().max()) for parameter, reference in zip(selected, expected)]
    check("scaled_aux_only_private_route_real_model", max(differences) < 1e-6, {"group_max_abs": differences, "scale": 32})
    check("real_model_routed_gradients_finite", all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
    control = V3Student(copy.deepcopy(model.base), canvas=128, base_only=True).eval()
    control_output = control(image)
    check("B0_B1_no_private_modules_or_teacher_network", control.private is None and sum(p.numel() for p in control.parameters()) == 65572420)
    check("B0_B1_output_exact_original_inference", torch.equal(control_output["logits"], control.base.encode_decode(image, [])))
    b0_task, b0_aux, _ = losses(control_output, label, valid, None, None, None, 4000, mode="B0")
    check("B0_only_masked_GT_and_HR", torch.equal(b0_task, masked_dice(control_output["logits"], label, valid)) and b0_aux == 0)
    a, u, s = (torch.rand_like(control_output["logits"]) for _ in range(3))
    b1_task, b1_aux, _ = losses(control_output, label, valid, a, u, s, 4000, mode="B1")
    expected_b1 = b0_task+.5*masked_kl(control_output["logits"], .75*(.75*a+.25*u)+.25*s, valid)
    check("B1_only_GT_HR_and_final_T_KD", torch.equal(b1_task, expected_b1) and b1_aux == 0)
    routed_backward(b1_task, b1_aux, control.private_parameters())
    check("B0_B1_empty_private_route_backward_valid", control.base.backbone.conv1.weight.grad.abs().sum() > 0)
    return {"real_HRNet_source": str(REPO), "ImageNet_W48_SHA": file_sha(W48),
        "baseline_initial_state_SHA": initial_sha, "base_parameters": sum(p.numel() for p in model.base.parameters()),
        "total_parameters": sum(p.numel() for p in model.parameters()), "seconds": time.monotonic()-started}

def loss_and_resume_checks():
    torch.manual_seed(12)
    z = torch.randn(2, 4, 17, 19, requires_grad=True)
    label = torch.randint(0, 5, (2, 1, 17, 19))
    teacher = torch.rand_like(z, requires_grad=True)
    valid = torch.ones(2, 1, 17, 19); valid[..., :5, :] = 0
    d1 = torch.randn_like(z, requires_grad=True); d2 = torch.randn_like(z, requires_grad=True)
    target = [torch.rand_like(z)*2-1 for _ in range(2)]
    value = masked_dice(z, label, valid)+masked_kl(z, teacher, valid)+signed_aux([d1, d2], target, valid)
    zero = masked_dice(z, label, valid*0)+masked_kl(z, teacher, valid*0)+signed_aux([d1, d2], target, valid*0)
    check("empty_valid_domain_differentiable_zero", zero == 0 and all(g.abs().sum() == 0 for g in torch.autograd.grad(zero, (z, d1, d2), retain_graph=True)))
    check("teacher_target_detached", torch.autograd.grad(value, teacher, allow_unused=True)[0] is None)
    check("locked_schedules", lambda_k(1500) == 0 and lambda_k(4000) == .5 and lambda_k(32000) == .5 and abs(lambda_k(40000)-.1) < 1e-12)
    check("original_poly_warmup_including_poly_factor", abs(learning_rate(0)-5e-9) < 1e-15 and abs(learning_rate(1500)-.0048125) < 1e-15)
    batches = list(ResumableBatches(454, 0, 240))
    resumed = list(ResumableBatches(454, 215, 240))
    check("sample_ids_and_augmentation_seeds_resume_exact", batches[215:] == resumed)
    first_epoch_ids = [index for batch in batches[:227] for index, seed in batch]
    check("two_images_per_batch_all_454_once_per_epoch", len(first_epoch_ids) == 454 and len(set(first_epoch_ids)) == 454)

if __name__ == "__main__":
    geometry_checks()
    model_details = integration_checks()
    loss_and_resume_checks()
    report = {"status": "PASS", "checks": checks, "model": model_details, "torch": torch.__version__,
        "scope": {"real_HRNet": True, "canvas": 128, "GPU_used": False, "CUDA_AMP_validated": False,
            "DDP_supported": False, "formal_training_updates": 0, "historical_B0_initial_snapshot_compared": False}}
    (HERE/"CPU_PREPARATION_PASS.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": "PASS", "checks": len(checks), "model": model_details}), flush=True)
