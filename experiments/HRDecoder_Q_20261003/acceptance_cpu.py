"""Required coordinate/loss/decision/resume checks; no system inspection or test data."""
from pathlib import Path
import tempfile

import numpy as np
import torch

from qlib.common import digest, read_json, seed_all, utc_now, write_json
from qlib.geometry import Tile, tiles, transform, transform_points, rgb_window, supervision
from qlib.loss import masked_loss
from qlib.metrics import counts, summarize
from qlib.decision import serious_failure, promotion_signals, extension_required, lock_selection
from qlib.optim import lr_for_update, Schedule
from qlib import checkpoint


ROOT = Path(__file__).resolve().parent


def main():
    seed_all(42)
    checks = {}
    # Actual contract dimensions: irregular final tiles and smaller-than-center image.
    for h, w in ((811, 1593), (83, 97), (1536, 2304)):
        seen = np.zeros((h, w), np.uint8)
        target = np.zeros((h, w, 4), np.uint8)
        target[h//2, w//2, :] = 1
        valid = np.ones_like(target)
        restored = np.zeros_like(target)
        for t in tiles(h, w):
            g, v = supervision(target, valid, t)
            assert v.sum() == t.height*t.width*4
            assert g[:, t.height:, :].sum() == 0 and v[:, t.height:, :].sum() == 0
            restored[t.y:t.y+t.height, t.x:t.x+t.width] = g[:, :t.height, :t.width].transpose(1, 2, 0)
            seen[t.y:t.y+t.height, t.x:t.x+t.width] += 1
        assert (seen == 1).all() and np.array_equal(restored, target)
    checks['native_halo_padding_stitch_exact_once'] = True
    h, w = 61, 93
    points = np.array([[2, 4], [30, 50], [60, 92]])
    marked = np.zeros((h, w), np.uint8)
    marked[points[:, 0], points[:, 1]] = np.arange(1, 4)
    for k in range(4):
        for fh in (False, True):
            for fv in (False, True):
                row = dict(rotation=k, flip_h=fh, flip_v=fv)
                transformed = transform(marked, row)
                predicted, hw = transform_points(points, (h, w), row)
                assert tuple(transformed.shape) == hw
                assert np.array_equal(transformed[predicted[:, 0], predicted[:, 1]], [1, 2, 3])
    checks['all_16_rotation_flip_combinations'] = True
    patch = rgb_window(np.zeros((83, 97, 3), np.uint8), Tile(0, 0, 83, 97))
    assert patch.shape == (3, 1024, 1024) and (patch[:, :128] == 0).all()
    assert (patch[:, 128:211, 128:225] != 0).all()  # real black is valid normalized RGB
    checks['normalized_zero_padding_and_real_black_retained'] = True
    z = torch.zeros(2, 4, 23, 31, requires_grad=True)
    g, v = torch.zeros_like(z), torch.ones_like(z)
    v[:, 1, :10] = 0
    base = masked_loss(z, g, v)
    changed_z, changed_g = z.detach().clone(), g.clone()
    changed_z[:, 1, :10] = float('nan')
    changed_g[:, 1, :10] = 1000
    assert torch.equal(base.detach(), masked_loss(changed_z, changed_g, v))
    assert masked_loss(torch.full_like(z, 4.), g, v) > base
    base.backward()
    assert z.grad.abs().sum() > 0 and z.grad[:, 1, :10].abs().sum() == 0
    try:
        masked_loss(z, g, torch.zeros_like(v))
        raise AssertionError('All-invalid loss accepted')
    except ValueError:
        pass
    checks['fp32_masked_loss_empty_positive_ignore_and_background_penalty'] = True
    p = np.array([.5, np.nextafter(np.float32(.5), np.float32(1)), .1, .9], np.float32).reshape(4, 1, 1)
    ct = counts(p, np.array([0, 1, 1, 0], np.uint8).reshape(4, 1, 1), np.ones((4, 1, 1), np.uint8))
    assert ct.tolist() == [[0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
    checks['strict_threshold_and_dataset_aggregation'] = True
    assert lr_for_update('head', 200) == 3e-4 and lr_for_update('backbone', 2000) == 0
    assert lr_for_update('backbone', 2500) == 5e-6
    assert abs(lr_for_update('head', 24000)-3e-6) < 1e-15
    assert abs(lr_for_update('backbone', 24000)-5e-8) < 1e-15
    checks['warmup_unfreeze_and_fixed_24k_lr_horizon'] = True
    def node(s, f):
        return {'single': {'miou_pct': s}, 'fixed_fusion': {'miou_pct': f}}
    assert serious_failure({4000: node(30, 40), 8000: node(30.1, 40)})
    assert not serious_failure({4000: node(30, 40), 8000: node(30.3, 40)})
    assert not serious_failure({4000: node(30, 40)})
    arms = {'Q_DDR': {12000: node(44.1, 43), 16000: node(44.5, 43)}}
    assert extension_required(arms) and len(promotion_signals(arms)) == 1
    mixed_formula = {'Q_DDR': {12000: node(44.2, 43), 16000: node(43, 44.2)}}
    assert not promotion_signals(mixed_formula)
    results = {'Q_DDR': {42: arms['Q_DDR'], 2024: arms['Q_DDR']}}
    assert lock_selection(results, 16000)['selected']['test_eligible']
    checks['stop_extension_promotion_and_two_seed_selection'] = True
    # Check exact RNG, next draw and LR recovery on a small optimizer; large
    # actual model/optimizer recovery remains a separate GPU acceptance gate.
    model = torch.nn.Linear(2, 1)
    model.set_update = lambda u: None
    opt = torch.optim.AdamW([{'params': model.parameters(), 'family': 'head', 'lr': 0.}])
    sched = Schedule(opt)
    model(torch.ones(1, 2)).sum().backward(); opt.step(); sched.step(); opt.zero_grad()
    rows = [{'draw': i} for i in range(24)]
    identity = {'config': 'synthetic_resume_only', 'seed': 42}
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as tmp:
        path = Path(tmp)/'resume.pth'
        checkpoint.save(path, model, opt, sched, None, 1, identity, rows[8:16])
        expected_random = torch.rand(3)
        for p in model.parameters():
            p.data.add_(5)
        completed = checkpoint.load(path, model, opt, sched, None, identity, rows)
        assert completed == 1 and torch.equal(torch.rand(3), expected_random)
        assert opt.param_groups[0]['lr'] == lr_for_update('head', 2)
        try:
            checkpoint.load(path, model, opt, sched, None, {'config': 'wrong'}, rows)
            raise AssertionError('Wrong checkpoint identity accepted')
        except ValueError:
            pass
    checks['strict_checkpoint_rng_next_draw_lr_identity'] = True
    result = {'status': 'PASS', 'checks': checks, 'created_at': utc_now(), 'test_read': False,
              'gpu_pretrained_wiring_validated': False, 'scope': 'CPU protocol acceptance only'}
    write_json(ROOT/'reports/cpu_acceptance.json', result)
    print(result)


if __name__ == '__main__':
    main()
