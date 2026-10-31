"""CPU design probes only: no HRNet, real images, teacher weights or optimizer steps."""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(42)

def resize(x, hw):
    return F.interpolate(x, size=hw, mode='bilinear', align_corners=False)

class ChannelLN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=1e-5)
    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

def projection(cin, cout, k=1, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k//2, bias=False), ChannelLN(cout), nn.GELU())

class Block(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(64, 64, k, padding=k//2, groups=64, bias=False),
                                  ChannelLN(64), nn.GELU(), nn.Conv2d(64, 64, 1, bias=False))
    def forward(self, x):
        return x + self.body(x)

def zero_output():
    out = nn.Conv2d(64, 4, 1)
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    return out

class Detail(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(projection(3, 16, 3, 2), projection(16, 32, 3, 2))
        self.proj = nn.ModuleList([projection(48, 32), projection(96, 32)])
        self.fuse = projection(96, 64)
        self.blocks = nn.Sequential(Block(3), Block(3))
        self.out = zero_output()
    def forward(self, rgb, feats):
        n = self.stem(rgb)
        x = torch.cat([n] + [resize(p(f), n.shape[-2:]) for p, f in zip(self.proj, feats[:2])], 1)
        return self.out(self.blocks(self.fuse(x))), n

class Context(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleList([projection(c, 64) for c in (48, 96, 192, 384)])
        self.top = nn.ModuleList([Block(7), Block(7), Block(3)])
        self.fuse, self.fine, self.out = projection(96, 64), Block(3), zero_output()
    def forward(self, feats, n):
        ps = [p(f) for p, f in zip(self.proj, feats)]
        x = ps[3]
        for block, lateral in zip(self.top, reversed(ps[:3])):
            x = block(resize(x, lateral.shape[-2:]) + lateral)
        return self.out(self.fine(self.fuse(torch.cat([resize(x, n.shape[-2:]), n], 1))))

class H4(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def features(self, rgb, stages):
        ma = [z[:, 3:4].detach() for z in stages]
        return torch.cat([rgb.detach()] + [z.clamp(-math.log(9999), math.log(9999)) for z in ma]
                         + [ma[2].sigmoid()-ma[0].sigmoid()], 1)
    def forward(self, f7):
        return 6*self.net(f7).tanh()
    def corrected(self, rgb, stages):
        base = stages[2].detach()
        return torch.cat([base[:, :3], base[:, 3:4]+self(self.features(rgb, stages))], 1)
    def tiled(self, f7, tile=16):
        height, width = f7.shape[-2:]
        rows = []
        for y in range(0, height, tile):
            cells = []
            for x in range(0, width, tile):
                y1, x1 = min(y+tile, height), min(x+tile, width)
                sy, sx = max(y-2, 0), max(x-2, 0)
                ey, ex = min(y1+2, height), min(x1+2, width)
                part = self(f7[..., sy:ey, sx:ex])
                cells.append(part[..., y-sy:y1-sy, x-sx:x1-sx])
            rows.append(torch.cat(cells, -1))
        return torch.cat(rows, -2)

class NewModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.detail, self.context, self.h4 = Detail(), Context(), H4()
    def stages(self, rgb, feats, z0):
        r1, n = self.detail(rgb, feats)
        r2 = self.context(feats, n)
        z1 = z0 + resize(r1, z0.shape[-2:])
        return [z0, z1, z1 + resize(r2, z0.shape[-2:])]

def class_average(values, valid):
    active = valid.sum((-2, -1)) > 0
    return torch.where(active, values, 0.).sum()/active.sum().clamp_min(1)

def dice(z, y, valid):
    p, v = z.sigmoid(), valid.to(z.dtype).expand_as(z)
    num = 2*(p*y*v).sum((-2, -1)) + 1e-5
    den = ((p.square()+y.square())*v).sum((-2, -1)) + 1e-5
    return class_average(1-num/den, v)

def kd(z, target, y, valid):
    target, v = target.detach(), valid.to(z.dtype).expand_as(z)
    entropy = -(torch.xlogy(target, target) + torch.xlogy(1-target, 1-target))
    kl = F.binary_cross_entropy_with_logits(z, target, reduction='none') - entropy
    confidence = y*target+(1-y)*(1-target)
    result = sum(.5*(kl*confidence*g).sum((-2, -1))/g.sum((-2, -1)).clamp_min(1)
                 for g in [v*y, v*(1-y)])
    return class_average(result, v)

def h4_loss(zma, yma, valid):
    v, dims = valid.to(zma.dtype), (-3, -2, -1)
    count = v.sum(dims)
    bce = (F.binary_cross_entropy_with_logits(zma, yma, reduction='none')*v).sum(dims)/count.clamp_min(1)
    p = zma.sigmoid()
    d = 1-2*(p*yma*v).sum(dims)/(((p.square()+yma.square())*v).sum(dims)+1)
    d = torch.where((yma*v).sum(dims)>0, d, 0.)
    return torch.where(count>0, .5*bce+.5*d, 0.).sum()/(count>0).sum().clamp_min(1)

def run():
    checks, notes = {}, {}
    def record(name, condition, detail=None):
        checks[name] = bool(condition)
        if detail is not None:
            notes[name] = detail
        assert condition, (name, detail)

    model = NewModules().cpu()
    rgb = torch.randn(2, 3, 64, 64, requires_grad=True)
    feats = [torch.randn(2, c, s, s, requires_grad=True) for c, s in zip((48, 96, 192, 384), (8, 4, 2, 1))]
    z0 = torch.randn(2, 4, 64, 64, requires_grad=True)
    y = (torch.rand_like(z0)>.96).float()
    valid = torch.ones(2, 1, 64, 64)
    valid[..., :5, :] = 0
    targets, weights = [torch.rand_like(z0) for _ in range(3)], (.25, .25, .5)
    parts = dict(detail_including_shared_stem=model.detail, context=model.context, H4=model.h4)
    counts = {k: sum(p.numel() for p in part.parameters()) for k, part in parts.items()}
    record('parameter_counts', list(counts.values())==[26004, 77444, 3361], counts)
    record('total_new_parameters', sum(counts.values())==106809)
    mac, handles = dict.fromkeys(parts, 0), []
    for key, part in parts.items():
        def hook(layer, args, output, key=key):
            mac[key] += output.numel()*(layer.in_channels//layer.groups)*math.prod(layer.kernel_size)
        handles.extend(layer.register_forward_hook(hook) for layer in part.modules() if isinstance(layer, nn.Conv2d))
    stages = model.stages(rgb, feats, z0)
    out = model.h4.corrected(rgb, stages)
    for handle in handles:
        handle.remove()
    mac = {k: v//2*32**2 for k, v in mac.items()}
    record('conv_MAC_2048_batch1', list(mac.values())==[5939134464, 3734241280, 13958643712], mac)
    record('shape_and_initial_identity', all(z.shape==z0.shape and torch.equal(z, z0) for z in stages) and torch.equal(out, z0))
    record('restore_logits_initial_identity', torch.equal(resize(out, (79, 83)).sigmoid(), resize(z0, (79, 83)).sigmoid()))
    gt = sum(w*dice(z, y, valid) for w, z in zip(weights, stages))
    record('initial_GT_value_matches_same_mask_baseline', torch.allclose(gt, dice(z0, y, valid)))
    (gt+.5*sum(w*kd(z, t, y, valid) for w, z, t in zip(weights, stages, targets))).backward()
    record('main_gradients_finite', all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()) and torch.isfinite(z0.grad).all())
    record('first_output_layers_have_gradient', model.detail.out.weight.grad.abs().sum()>0 and model.context.out.weight.grad.abs().sum()>0)
    record('first_hidden_gradient_zero', all(p.grad is None or p.grad.abs().sum()==0 for name, p in model.named_parameters() if not name.startswith('h4') and '.out.' not in name))
    record('main_has_no_H4_gradient', all(p.grad is None for p in model.h4.parameters()))
    record('main_MA_has_gradient', z0.grad[:, 3].abs().sum()>0)
    model.zero_grad(set_to_none=True)
    for tensor in [rgb, z0]+feats:
        tensor.grad = None
    # Direct diagnostic assignment, not an optimizer step or trained weights.
    with torch.no_grad():
        model.detail.out.weight.normal_(0, .01)
        model.context.out.weight.normal_(0, .01)
        model.h4.net[-1].weight.normal_(0, .05)
    stages = model.stages(rgb, feats, z0)
    dice(stages[-1], y, valid).backward()
    record('hidden_gradients_with_nonzero_output_probe', all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for name, p in model.named_parameters() if not name.startswith('h4')))
    model.zero_grad(set_to_none=True)
    for tensor in [rgb, z0]+feats:
        tensor.grad = None
    stages = model.stages(rgb, feats, z0)
    corrected = model.h4.corrected(rgb, stages)
    h4_loss(corrected[:, 3:4], y[:, 3:4], valid).backward()
    record('H4_gradient_isolation', all(p.grad is None for name, p in model.named_parameters() if not name.startswith('h4')) and all(t.grad is None for t in [rgb, z0]+feats))
    record('H4_all_gradients_finite_nonzero', all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.h4.parameters()))
    record('H4_changes_only_MA', torch.equal(corrected[:, :3], stages[-1][:, :3]) and not torch.equal(corrected[:, 3:4], stages[-1][:, 3:4]))
    f7 = torch.randn(1, 7, 43, 51)
    tile_error = (model.h4(f7)-model.h4.tiled(f7, 16)).abs().max().item()
    record('H4_halo2_nonzero_tiles_match_full', tile_error<2e-6, tile_error)
    changed, yd, td = z0.detach().clone(), y.clone(), targets[0].clone()
    changed[..., :5, :], yd[..., :5, :], td[..., :5, :] = 100, 1-yd[..., :5, :], 1-td[..., :5, :]
    record('ignored_prediction_label_target_invariance', torch.allclose(dice(z0, y, valid), dice(changed, yd, valid)) and torch.allclose(kd(z0, targets[0], y, valid), kd(changed, td, yd, valid)) and torch.allclose(h4_loss(z0[:, 3:4], y[:, 3:4], valid), h4_loss(changed[:, 3:4], yd[:, 3:4], valid)))
    ez = torch.randn(2, 4, 5, 7, requires_grad=True)
    ey, ev = torch.zeros_like(ez), torch.zeros(2, 1, 5, 7)
    el = dice(ez, ey, ev)+kd(ez, ey, ey, ev)+h4_loss(ez[:, 3:4], ey[:, 3:4], ev)
    el.backward()
    record('all_ignore_graph_zero', el.item()==0 and ez.grad.abs().sum()==0)
    record('H4_empty_foreground_half_BCE', torch.allclose(h4_loss(ez[:, 3:4], ey[:, 3:4], torch.ones_like(ev)), .5*F.binary_cross_entropy_with_logits(ez[:, 3:4], ey[:, 3:4])))
    extreme = torch.tensor([-1000., 1000., -1000., 1000.]).reshape(1, 1, 2, 2).requires_grad_()
    et = torch.tensor([0., 1., 1., 0.]).reshape_as(extreme)
    ek = kd(extreme, et, et, torch.ones_like(extreme))
    ek.backward()
    record('KL_extreme_logits_endpoint_targets_finite', torch.isfinite(ek) and torch.isfinite(extreme.grad).all())
    ts = [torch.full((1, 1, 1, 1), x, dtype=torch.float64) for x in (.5-1e-6, .5+1e-6)]
    ks = [kd(torch.ones_like(t), t, torch.ones_like(t), torch.ones_like(t)).item() for t in ts]
    record('KD_continuous_at_half', abs(ks[0]-ks[1])<1e-6, ks)
    record('KD_continuity_probe_is_nonzero', min(ks)>0)
    kt = torch.full((1, 1, 2, 3), .2)
    kz = torch.zeros_like(kt)
    expected = .5*.8*(F.binary_cross_entropy_with_logits(kz, kt)+(torch.xlogy(kt, kt)+torch.xlogy(1-kt, 1-kt)).mean())
    record('KD_empty_group_not_renormalized', torch.allclose(kd(kz, kt, kz, torch.ones_like(kt)), expected))
    record('v1_conditional_veto_counterexample', .5625*.05+.1875+.25<.5)
    record('v2_override_negative_base', torch.sigmoid(torch.tensor(-10.+11.)).item()>.5)
    ta, tu, ts = .05, .95, .05
    t1, t2 = .75*ta+.25*tu, .5625*ta+.1875*tu+.25*ts
    record('weak_member_error_not_required_by_targets', t2<.5 and tu>.5, dict(A=ta, P0=t1, T2=t2, weak_U=tu))
    logit = lambda x: math.log(x/(1-x))
    record('teacher_stage_and_logit_telescoping', abs(t2-(.75*t1+.25*ts))<1e-12 and abs(logit(ta)+(logit(t1)-logit(ta))+(logit(t2)-logit(t1))-logit(t2))<1e-12)
    delta = (resize(model.h4(f7), (59, 61))-model.h4(resize(f7, (59, 61)))).abs().max().item()
    record('head_and_resize_not_commutative', delta>1e-3, delta)
    spec = json.loads((Path(__file__).parent/'02_架构规格.json').read_text(encoding='utf-8-sig'))
    record('spec_core_contract', spec['version']==2 and spec['new_parameters_total']==sum(counts.values()) and spec['student_fusion']['domain']=='logit_residual' and spec['training']['global_batches']==40000 and spec['H4']['optimizer_updates']==40000)
    result = dict(scope=__doc__, version=2, torch=torch.__version__, device='cpu', seed=42,
                  passed=len(checks), failed=0, checks=checks, details=notes, new_parameters=counts,
                  new_parameters_total=sum(counts.values()), conv_MAC_2048_batch1=mac,
                  conv_MAC_total=sum(mac.values()),
                  MAC_excludes=['backbone', 'original HRDecoder', 'interpolation', 'normalization', 'activations', 'loss', 'backward', 'optimizer', 'teacher preprocessing'],
                  unverified=['full HRDecoder integration', 'geometric transforms and HR crop mask plumbing', 'ImageNet SHA', '454 train teacher caches', 'GPU memory/speed', 'training stability', 'mIoU', 'multi-seed variation', 'external generalization'])
    (Path(__file__).parent/'evidence'/'design_verification_v2.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    run()
