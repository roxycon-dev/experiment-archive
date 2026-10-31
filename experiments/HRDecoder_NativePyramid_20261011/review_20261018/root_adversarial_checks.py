"""CPU design conformance fixtures only; no W48, data, GPU or optimizer steps."""
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "jointcontext_spec_v2.json"
EPS = 1e-5
RESULTS = []
torch.set_num_threads(1)


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append({"name": name, "status": "PASS", "detail": detail})
    except Exception as exc:
        RESULTS.append({"name": name, "status": "FAIL",
                        "error": type(exc).__name__ + ": " + str(exc)})


def amp(enabled):
    return torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False) if enabled else nullcontext()


def dice(logits, labels, pooled=True):
    valid = labels != 255
    if not valid.any():
        raise ValueError("no valid supervision")
    parts = []
    for c in range(4):
        if pooled:
            p = torch.sigmoid(logits[:, c][valid])
            y = (labels[valid] == c + 1).to(p.dtype)
            parts.append(1 - (2 * (p * y).sum() + EPS) /
                         ((p * p).sum() + y.sum() + EPS))
        else:
            each = []
            for i in range(labels.shape[0]):
                vi = valid[i]
                if not vi.any():
                    raise ValueError("filter all-ignore aux instances before the head")
                p = torch.sigmoid(logits[i, c][vi])
                y = (labels[i][vi] == c + 1).to(p.dtype)
                each.append(1 - (2 * (p * y).sum() + EPS) /
                            ((p * p).sum() + y.sum() + EPS))
            parts.append(torch.stack(each).mean())
    return torch.stack(parts).mean()


class BankBN(nn.Module):
    """Small executable reference for shared affine and read-only recompute."""
    def __init__(self, channels, domains):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        for domain in domains:
            self.register_buffer("mean_" + domain, torch.zeros(channels))
            self.register_buffer("var_" + domain, torch.ones(channels))
            self.register_buffer("count_" + domain, torch.zeros((), dtype=torch.long))

    def forward(self, x, domain, write=True):
        mean = getattr(self, "mean_" + domain)
        var = getattr(self, "var_" + domain)
        if write:
            getattr(self, "count_" + domain).add_(1)
        else:
            mean, var = mean.clone(), var.clone()
        with torch.autocast("cpu", enabled=False):
            out = F.batch_norm(x.float(), mean, var, self.weight, self.bias,
                               training=True, momentum=.1, eps=EPS)
        return out.to(x.dtype)


class TinyEncoder(nn.Module):
    """Four output stages exercise intermediate-output VJPs; not an HRNet."""
    def __init__(self):
        super().__init__()
        channels = [3, 2, 3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i + 1], 3,
                      stride=4 if i == 0 else 2, padding=1, bias=False)
            for i in range(4)])
        self.norms = nn.ModuleList([BankBN(c, ["G", "N"]) for c in channels[1:]])

    def forward(self, x, domain, write=True):
        out = []
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x), domain, write))
            out.append(x)
        return tuple(out)


class TinyJointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(14, 4, 1, bias=False)
        self.pbn = BankBN(4, ["G", "N", "aux"])
        self.kc = nn.Conv2d(4, 4, 7, padding=3, bias=False)
        self.kd = nn.Conv2d(4, 4, 7, padding=3, bias=False)
        nn.init.zeros_(self.kd.weight)
        self.hbn = BankBN(4, ["main", "aux"])
        self.cls = nn.Conv2d(4, 4, 1)

    def joined(self, feats):
        return torch.cat([F.interpolate(f, feats[0].shape[-2:], mode="bilinear",
                                        align_corners=False) for f in feats], dim=1)

    def projected(self, x, domain):
        return F.relu(self.pbn(self.proj(x), domain))

    def forward(self, fg, fn, labels, boxes):
        fg, fn = self.joined(fg), self.joined(fn)
        g = F.interpolate(self.projected(fg, "G"), fn.shape[-2:],
                          mode="bilinear", align_corners=False)
        n = self.projected(fn, "N")
        common = (n.float() + g.float()) / math.sqrt(2)
        difference = (n.float() - g.float()) / math.sqrt(2)
        a = self.kc(common).float() + self.kd(difference).float()
        z = self.cls(F.relu(self.hbn(a, "main")))
        with torch.autocast("cpu", enabled=False):
            z = F.interpolate(z.float(), (64, 64), mode="bilinear", align_corners=False)
            main_loss = dice(z, labels)
        crops, masks = [], []
        for y, x in boxes:
            m = labels[:, y:y + 32, x:x + 32]
            f = fg[:, :, y // 8:(y + 32) // 8, x // 8:(x + 32) // 8]
            f = F.interpolate(f, (8, 8), mode="bilinear", align_corners=False)
            keep = (m != 255).flatten(1).any(1)
            crops.extend(f[i:i + 1] for i in range(2) if keep[i])
            masks.extend(m[i:i + 1] for i in range(2) if keep[i])
        aux_loss = main_loss.new_zeros(())
        if crops:
            f = self.projected(torch.cat(crops), "aux")
            az = self.cls(F.relu(self.hbn(self.kc(f), "aux")))
            with torch.autocast("cpu", enabled=False):
                az = F.interpolate(az.float(), (32, 32), mode="bilinear", align_corners=False)
                aux_loss = dice(az, torch.cat(masks), pooled=False)
        return main_loss + .1 * aux_loss, len(crops), z


class TinyFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.head = TinyJointHead()


def gradients_and_state(precision, aux_case):
    torch.manual_seed(17018)
    full = TinyFixture()
    staged = copy.deepcopy(full)
    x = torch.randn(2, 3, 64, 64)
    ginput = F.interpolate(x, (32, 32), mode="bilinear", align_corners=False)
    labels = torch.zeros(2, 64, 64, dtype=torch.long)
    labels[:, 8:15, 9:14] = 1
    labels[:, 24:29, 33:41] = 2
    labels[:, 40:48, 20:27] = 3
    labels[:, 53:55, 54:56] = 4
    boxes = [(0, 0), (0, 32), (32, 0), (32, 32)]
    if aux_case == "one":
        labels.fill_(255)
        labels[1, 4:12, 4:12] = 0
        labels[1, 5:8, 5:8] = 1
    elif aux_case == "none":
        labels.fill_(255)
        labels[:, 56:64, 56:64] = 1
        boxes = [(0, 0), (0, 8), (8, 0), (16, 16)]
    assert (labels != 255).any()
    bf16 = precision == "bf16"
    with amp(bf16):
        fg = full.encoder(ginput, "G")
    with amp(bf16):
        fn = full.encoder(x, "N")
    with amp(bf16):
        loss_full, naux, zfull = full.head(fg, fn, labels, boxes)
    loss_full.backward()
    with torch.no_grad(), amp(bf16):
        cg = staged.encoder(ginput, "G")
    with torch.no_grad(), amp(bf16):
        cn = staged.encoder(x, "N")
    cg = tuple(t.detach().requires_grad_(True) for t in cg)
    cn = tuple(t.detach().requires_grad_(True) for t in cn)
    with amp(bf16):
        loss_staged, snaux, zstaged = staged.head(cg, cn, labels, boxes)
    loss_staged.backward()
    adjoints = [tuple(t.grad.detach().clone() for t in feats) for feats in (cg, cn)]
    for domain, inp, cached, grad in zip(["G", "N"], [ginput, x], [cg, cn], adjoints):
        before = {k: v.clone() for k, v in staged.encoder.named_buffers()}
        with amp(bf16):
            recomputed = staged.encoder(inp, domain, write=False)
        for old, new in zip(cached, recomputed):
            assert torch.equal(old, new), "cache/recompute forward changed"
            assert new.requires_grad
        torch.autograd.backward(recomputed, grad)
        for key, value in staged.encoder.named_buffers():
            assert torch.equal(value, before[key]), "BN recompute wrote persistent state"
    assert snaux == naux == {"all": 8, "one": 1, "none": 0}[aux_case]
    assert torch.equal(zfull, zstaged)
    max_abs, sq_error, sq_ref = 0., 0., 0.
    bad = []
    for (key, p), (skey, sp) in zip(full.named_parameters(), staged.named_parameters()):
        assert key == skey and p.grad is not None and sp.grad is not None, key
        delta = (p.grad - sp.grad).double()
        error = float(delta.abs().max())
        max_abs = max(max_abs, error)
        sq_error += float(delta.square().sum())
        sq_ref += float(p.grad.double().square().sum())
        if not torch.allclose(p.grad, sp.grad, rtol=2e-5, atol=2e-6):
            bad.append(key)
    assert not bad, "gradient disagreement: " + str(bad)
    for (key, buf), (skey, sbuf) in zip(full.named_buffers(), staged.named_buffers()):
        assert key == skey and torch.equal(buf, sbuf), key
        if "count_" in key:
            expected = 0 if key.endswith("aux") and naux == 0 else 1
            assert int(buf) == expected, (key, buf, expected)
    assert staged.head.kd.weight.grad.norm() > 0
    assert len(list(staged.parameters())) == len({id(p) for p in staged.parameters()})
    return {"precision": precision, "valid_aux_instances": naux,
            "max_gradient_absolute_difference": max_abs,
            "global_gradient_relative_l2": math.sqrt(sq_error / max(sq_ref, 1e-30)),
            "exact_output_and_bn_state_match": True,
            "full_model": False, "optimizer_steps": 0}


def autocast_failure():
    torch.manual_seed(2)
    outputs = {}
    for cache in (True, False):
        layer = nn.Linear(4, 3, bias=False)
        x = torch.randn(2, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache):
            with torch.no_grad():
                layer(x)
            y = layer(x)
        outputs[str(cache)] = y.requires_grad
        if not cache:
            y.float().sum().backward()
            assert layer.weight.grad is not None
    assert outputs["False"]
    encoder_cases = {}
    for cache in (True, False):
        encoder = TinyEncoder()
        inp = torch.randn(2, 3, 64, 64)
        with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=cache):
            with torch.no_grad():
                encoder(inp, "N")
            feats = encoder(inp, "N", write=False)
            objective = sum(f.float().square().mean() for f in feats)
        assert objective.requires_grad
        objective.backward()
        missing = [key for key, p in encoder.named_parameters() if p.grad is None]
        encoder_cases[str(cache)] = {
            "output_requires_grad": objective.requires_grad,
            "missing_parameter_gradients": missing}
        if not cache:
            assert not missing
    return {"same_outer_context_cache_on_requires_grad": outputs["True"],
            "cache_off_requires_grad": outputs["False"],
            "BN_can_mask_missing_conv_gradients": encoder_cases,
            "observed_torch": torch.__version__, "device": "cpu"}


def loss_derivatives():
    torch.manual_seed(5)
    max_error = 0.
    for empty in (False, True):
        labels = torch.zeros(2, 3, 4, dtype=torch.long)
        labels[:, 0, 0] = 255
        if not empty:
            labels[0, 1, 1] = 1
            labels[1, 2, 2] = 3
        z = torch.randn(2, 4, 3, 4, dtype=torch.float64, requires_grad=True)
        loss = dice(z, labels)
        grad, = torch.autograd.grad(loss, z)
        valid = labels != 255
        for c in range(4):
            p = z[:, c].sigmoid()
            y = (labels == c + 1).double()
            a = 2 * (p[valid] * y[valid]).sum() + EPS
            b = p[valid].square().sum() + y[valid].sum() + EPS
            expected = (2 * a * p / b.square() - 2 * y / b) * p * (1 - p) / 4
            expected = torch.where(valid, expected, torch.zeros_like(expected))
            max_error = max(max_error, float((expected - grad[:, c]).abs().max().detach()))
            assert torch.allclose(expected, grad[:, c], atol=1e-12, rtol=1e-10)
        for ix in [(0, 0, 1, 1), (1, 2, 2, 2), (0, 1, 0, 0)]:
            plus, minus = z.detach().clone(), z.detach().clone()
            plus[ix] += 1e-5
            minus[ix] -= 1e-5
            fd = float((dice(plus, labels) - dice(minus, labels)) / 2e-5)
            assert abs(fd - float(grad[ix])) < 2e-9
    masked = torch.zeros(1, 4, 2, 2, dtype=torch.float64, requires_grad=True)
    y = torch.zeros(1, 2, 2, dtype=torch.long)
    y[:, 0, 0] = 255
    with torch.no_grad():
        masked[:, :, 0, 0] = float("nan")
    value = dice(masked, y)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(masked.grad).all()
    try:
        dice(torch.zeros(1, 4, 2, 2), torch.full((1, 2, 2), 255))
        raise AssertionError("all-ignore must reject")
    except ValueError:
        pass
    return {"max_analytic_autograd_error": max_error,
            "finite_difference_and_masked_nan": "PASS", "empty_class_uses_same_dice": True}


def optimizer_geometry():
    torch.manual_seed(9)
    n, g, kc, kd, gn, gg, vn, vg = [torch.randn(31, dtype=torch.float64) for _ in range(8)]
    r = math.sqrt(2)
    c, d = (n + g) / r, (n - g) / r
    assert torch.allclose(c.square().sum() + d.square().sum(), n.square().sum() + g.square().sum())
    kn, kg = (kc + kd) / r, (kc - kd) / r
    gc, gd = (gn + gg) / r, (gn - gg) / r
    vc, vd = (vn + vg) / r, (vn - vg) / r
    assert torch.allclose(kc * c + kd * d, kn * n + kg * g)
    clip = min(1., 5. / float(torch.sqrt(gn.square().sum() + gg.square().sum())))
    clip2 = min(1., 5. / float(torch.sqrt(gc.square().sum() + gd.square().sum())))
    assert abs(clip - clip2) < 1e-12
    eta, momentum, wd = .005, .9, .0005
    cn = kc - eta * (momentum * vc + clip2 * gc + wd * kc)
    dn = kd - eta * (momentum * vd + clip2 * gd + wd * kd)
    expected_n = kn - eta * (momentum * vn + clip * gn + wd * kn)
    expected_g = kg - eta * (momentum * vg + clip * gg + wd * kg)
    assert torch.allclose((cn + dn) / r, expected_n, atol=1e-12, rtol=1e-12)
    assert torch.allclose((cn - dn) / r, expected_g, atol=1e-12, rtol=1e-12)
    return {"energy_and_gradient_norm": "preserved",
            "momentum_weight_decay_global_clip_coordinate_identity": "PASS",
            "scope": "vector arithmetic; no optimizer or training loop",
            "bn_note": "common rescaling before BN is mostly normalized away; no convergence-speed claim"}


def counterexamples():
    v, eps, p = 2 * 2048 ** 2, EPS, .5
    def common_bias_grad(positives):
        a, b = 2 * positives * p + eps, v * p * p + positives + eps
        return (2 * a * v * p / (b * b) - 2 * positives / b) * p * (1-p)
    p1, p1000, p0 = common_bias_grad(1), common_bias_grad(1000), common_bias_grad(0)
    # Same proposed Dice can improve while the fixed-threshold IoU declines.
    def scalar_dice(prob):
        return 1 - (2 * prob[0] + EPS) / (sum(a*a for a in prob) + 1 + EPS)
    before, after = [.51, .49], [.49, .01]
    assert scalar_dice(after) < scalar_dice(before)
    return {"old_unit_empty_bce_bias_gradient": p,
            "same_dice_bias_gradient_if_P_1": p1,
            "same_dice_bias_gradient_if_P_1000": p1000,
            "same_dice_bias_gradient_if_P_0": p0,
            "old_empty_bce_to_P1_ratio": p / p1,
            "old_empty_bce_to_P1000_ratio": p / p1000,
            "scope": "constant p=0.5, all-valid illustrative batch; not observed network gradients",
            "loss_not_iou_guarantee": {"before": scalar_dice(before), "after": scalar_dice(after),
                                     "before_iou": 1., "after_iou": 0.}}


def lattice_observation():
    # Does not establish that a trained semantic feature must follow this ramp.
    ramp = (8 * torch.arange(256, dtype=torch.float32) + .5).view(1, 1, 1, 256)
    standard = F.interpolate(ramp, (1, 512), mode="bilinear", align_corners=False)
    native_centers = (4 * torch.arange(512, dtype=torch.float32)).view(1, 1, 1, 512)
    error = float((standard[..., 1:-1] - native_centers[..., 1:-1]).abs().max())
    assert error == 1.5
    return {"stem_center_ramp_offset_input_pixels": error,
            "decision": "retain conventional interpolation; remove any exact-lattice claim",
            "reason": "trained multi-branch feature semantics are not a coordinate ramp; custom correction has no performance evidence"}


def main():
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    assert spec["main_loss"]["empty_class_bce"] is False
    assert spec["precision"]["autocast_cache_enabled"] is False
    assert spec["decoder"]["common_feature"] == "C=(N+G)/sqrt(2)"
    check("review 1: conditional-loss scale and metric counterexamples", counterexamples)
    check("review 1: orthonormal view and optimizer-coordinate algebra", optimizer_geometry)
    check("review 1: interpolation interpretation", lattice_observation)
    check("review 2: AMP cached no_grad failure and cache-off repair", autocast_failure)
    check("review 2: revised Dice analytical/autograd/finite-difference agreement", loss_derivatives)
    for precision in ["fp32", "bf16"]:
        for case in ["all", "one", "none"]:
            check("review 3: full vs staged graphs " + precision + " aux=" + case,
                  lambda precision=precision, case=case: gradients_and_state(precision, case))
    report = {"date": "2026-10-18", "reviewer": "root", "torch": torch.__version__,
              "device": "CPU", "fixture": "small four-stage shared encoder and joint head",
              "full_W48_model_executed": False, "dataset_pixels_loaded": False,
              "gpu_used": False, "optimizer_steps": 0, "new_validation_metrics": False,
              "spec_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": RESULTS, "passed": sum(x["status"] == "PASS" for x in RESULTS),
              "failed": sum(x["status"] == "FAIL" for x in RESULTS)}
    (ROOT / "root_adversarial_checks.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
