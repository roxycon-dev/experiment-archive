"""Required coordinate, gradient, RNG, and inherited-model wiring gates on CPU."""
import copy
import gc
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qlib.common import read_json, write_json, utc_now
from qlib.model import build_model
from qlib.loss import masked_loss
from spatial import reference_attention, grid_centers
from final_model import FinalModel, inherited_parameter_map
from random_streams import micro_seed, forward_rng
from schedule import factor, optimizer_for


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(7005)
    evidence = {'status': 'RUNNING', 'started_at': utc_now(), 'tests': {}}
    write_json(ROOT / 'reports/cpu_acceptance.json', evidence)
    centers = grid_centers([(3, 4)], torch.device('cpu'))
    torch.testing.assert_close(centers[0], torch.tensor([.125, 1 / 6]))
    torch.testing.assert_close(centers[4], torch.tensor([.125, .5]))
    value = torch.arange(12, dtype=torch.float32).view(1, 12, 1, 1)
    location = centers[0].view(1, 1, 1, 1, 1, 2).clone()
    weights = torch.ones(1, 1, 1, 1, 1)
    result = reference_attention(value, [(3, 4)], location, weights)
    torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-5, rtol=0)
    location[..., 0] += .25
    torch.testing.assert_close(reference_attention(value, [(3, 4)], location, weights),
                               torch.ones_like(result), atol=1e-5, rtol=0)
    outside = torch.full_like(location, -1)
    torch.testing.assert_close(reference_attention(value, [(3, 4)], outside, weights),
                               torch.zeros_like(result), atol=0, rtol=0)
    evidence['tests']['one_pixel_xy_and_zero_boundary'] = 'PASS'
    value = torch.randn(1, 10, 2, 2, dtype=torch.float64, requires_grad=True)
    location = (torch.rand(1, 2, 2, 2, 2, 2, dtype=torch.float64) * .5 + .2).requires_grad_()
    weights = torch.randn(1, 2, 2, 2, 2, dtype=torch.float64).softmax(-1).requires_grad_()
    assert torch.autograd.gradcheck(lambda v, l, w: reference_attention(v, [(2, 3), (2, 2)], l, w),
                                  (value, location, weights), eps=1e-6, atol=1e-4, rtol=1e-3)
    evidence['tests']['reference_backward_gradcheck'] = 'PASS'
    z = torch.zeros(2, 4, 8, 8, requires_grad=True)
    g, v = torch.zeros_like(z), torch.ones_like(z)
    v[:, :, 0] = 0
    loss = masked_loss(z, g, v)
    loss.backward()
    assert torch.isfinite(loss) and (z.grad[:, :, 1:] > 0).all() and not z.grad[:, :, 0].any()
    evidence['tests']['negative_and_ignore_loss'] = 'PASS'
    before = torch.get_rng_state().clone()
    with forward_rng(42, 'DDR', 1, 0, torch.device('cpu')):
        paired = torch.rand(8)
    assert torch.equal(before, torch.get_rng_state())
    with forward_rng(42, 'TJDR', 1, 0, torch.device('cpu')):
        torch.rand(1000)
    with forward_rng(42, 'DDR', 1, 0, torch.device('cpu')):
        assert torch.equal(paired, torch.rand(8))
    assert micro_seed(42, 'DDR', 1, 0) != micro_seed(42, 'TJDR', 1, 0)
    assert factor(1) == .002 and factor(500) == 1 and abs(factor(32000) - .01) < 1e-15
    evidence['tests']['paired_forward_RNG_and_schedule'] = 'PASS'
    runtime = read_json(ROOT / 'configs/runtime.local.json')
    pretrained = read_json(ROOT / 'reports/pretraining_identity.json')
    parent, loading = build_model(runtime['dinov3_repo'], runtime['pretrained_checkpoint'], pretrained['sha256'])
    protocol = read_json(ROOT / 'configs/protocol.json')
    state = torch.load(protocol['initialization']['seed42']['path'], map_location='cpu', weights_only=False)
    assert state['completed'] == 12000
    parent.load_state_dict(state['model'], strict=True)
    del state
    parent.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = parent(image)
    identity_rows = {}
    for arm in ('C', 'S', 'D', 'SD'):
        clone = copy.deepcopy(parent)
        global_rng = torch.get_rng_state().clone()
        model = FinalModel(clone, arm, 42, backend='reference').eval()
        assert torch.equal(global_rng, torch.get_rng_state())
        mapping = inherited_parameter_map(parent)
        for old, new in mapping.items():
            assert torch.equal(parent.state_dict()[old], model.state_dict()[new])
        with torch.no_grad():
            actual = model(image)
        delta = float((actual - expected).abs().max())
        assert delta <= 1e-5, (arm, delta)
        assert torch.equal(actual.sigmoid() > .5, expected.sigmoid() > .5)
        optimizer, names = optimizer_for(model, fused=False)
        assert sum(len(v) for v in names.values()) == sum(p.requires_grad for p in model.parameters())
        if model.has_source:
            for a, b in zip(model.ddr.parameters(), model.tjdr.parameters()):
                assert a.data_ptr() != b.data_ptr() and torch.equal(a, b)
        identity_rows[arm] = {'max_abs_logit_difference': delta, 'inherited_parameters_exact': True,
                              'initialization_RNG_unchanged': True, 'parameter_names_accounted_for': True}
        if arm == 'S':
            # CPU gate isolates new-branch gradient relay. Full old/new parameter
            # updates and native1024 paths remain mandatory GPU gates.
            model.backbone.requires_grad_(False)
            model.ddr.requires_grad_(False)
            model.ddr.spatial_residual.requires_grad_(True)
            model.train()
            relay_optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            first, second = [], []
            for step in range(2):
                relay_optimizer.zero_grad(set_to_none=True)
                prediction = model(image)
                target = torch.zeros_like(prediction)
                target[:, :, 20:30, 24:36] = 1
                masked_loss(prediction, target, torch.ones_like(target)).backward()
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), name
                if step == 0:
                    for i, block in enumerate(model.spatial.interactions):
                        assert block.injector.output.weight.grad.abs().max() > 0, i
                    for i, projection in enumerate(model.ddr.spatial_residual):
                        assert projection.weight.grad.abs().max() > 0, i
                    first = ['all4injectors', 'all4decoder_spatial_residuals']
                else:
                    for name, p in model.spatial.named_parameters():
                        if p.grad is not None and p.grad.abs().max() > 0:
                            second.append(name)
                    assert any(n.startswith('pyramid.') for n in second)
                    assert any('extractor.' in n for n in second)
                    assert any('injector.value.' in n for n in second)
                relay_optimizer.step()
            evidence['tests']['two_step_new_spatial_gradient_relay'] = {'first': first, 'second_nonzero_parameters': second}
        del model, clone, optimizer
        gc.collect()
        print('CPU', arm, 'identity PASS', delta, flush=True)
        evidence['tests']['zero_step_real_F12_small_window'] = identity_rows
        write_json(ROOT / 'reports/cpu_acceptance.json', evidence)
    evidence.update(status='PASS', completed_at=utc_now(),
                    remaining_GPU_gates=['native1024 fullval parent restoration', 'optimized kernel equivalence',
                                         'full trainable old/new gradient updates', 'exact resume/EMA pairing',
                                         '100complete updates each path and full validation timing'])
    write_json(ROOT / 'reports/cpu_acceptance.json', evidence)
    print('CPU_ACCEPTANCE_PASS', flush=True)


if __name__ == '__main__':
    main()
