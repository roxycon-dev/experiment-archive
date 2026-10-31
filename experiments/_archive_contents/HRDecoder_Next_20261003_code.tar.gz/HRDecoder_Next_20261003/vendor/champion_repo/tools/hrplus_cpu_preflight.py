#!/usr/bin/env python
# tools/hrplus_cpu_preflight.py
"""CPU-only preflight for HRPlus smoke pipeline. No GPU, no training, no full forward."""
import argparse, json, os, sys, copy
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', default='hrplus_cpu_preflight.json')
    ap.add_argument('--weights', default='weights/b_weights.pth')
    args = ap.parse_args()

    repo_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_parent)

    from mmcv import Config
    from mmseg.models.builder import build_train_model
    from tools.hrplus_smoke import load_batch_cpu  # import the REAL function

    cfg = Config.fromfile(args.config)
    assert cfg.model.decode_head.num_classes == 4
    assert tuple(cfg.model.hr_settings.hr_scale) == (1024, 1024)

    model = build_train_model(cfg, train_cfg=cfg.get('train_cfg'))
    model.init_weights()
    torch.set_num_threads(4)
    model.cpu()
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            assert not m.training
    # reference params frozen
    for n, p in model.named_parameters():
        if n.startswith('reference.'):
            assert not p.requires_grad, f'reference param {n} requires grad'

    sd = torch.load(args.weights, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    msd = model.state_dict()
    student = {k: v for k, v in sd.items() if not k.startswith('reference.')}
    ref = {k[len('reference.'):]: v for k, v in msd.items() if k.startswith('reference.')}
    msd_student = {k: v for k, v in msd.items() if not k.startswith('reference.')}
    assert set(student.keys()) == set(msd_student.keys()), \
        f'key mismatch: {set(student) ^ set(msd_student)}'
    for k in student:
        assert torch.equal(student[k].float().cpu(), msd_student[k].float().cpu()), f'tensor mismatch {k}'
    assert set(ref) == set(sd)
    for k in sd: assert torch.equal(ref[k], sd[k]), k

    train_batch = load_batch_cpu(cfg, 'train')
    img = train_batch['img'].data[0]
    gt = train_batch['gt_semantic_seg'].data[0]
    assert tuple(img.shape) == (2, 3, 1024, 1024), f'train img {tuple(img.shape)}'
    assert tuple(gt.shape) == (2, 1, 1024, 1024), f'train gt {tuple(gt.shape)}'
    assert all(int(v) in [0,1,2,3,4,255] for v in gt.unique())

    val_batch = load_batch_cpu(cfg, 'val')
    vimg = val_batch['img']
    vimg = vimg[0] if isinstance(vimg, list) else vimg.data[0]
    assert tuple(vimg.shape) == (1, 3, 2048, 2048), f'val img {tuple(vimg.shape)}'
    vm = val_batch['img_metas'][0].data[0] if hasattr(val_batch['img_metas'][0], 'data') else val_batch['img_metas'][0]
    assert len(vm[0]['ori_shape']) >= 2
    assert getattr(cfg.data.val, 'img_dir', '').endswith('images/val'), cfg.data.val.img_dir

    # Synthetic loss test: decode head only, no backbone forward
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.zeros(2, 16, 16, dtype=torch.long)
    for c in range(1, 5):
        target[:, (c - 1) * 4:c * 4, :] = c
    target[:, 0, :] = 255  # ignore stripes
    inp = torch.where(target.unsqueeze(1) == 255, logits.new_tensor(-1e4), logits)
    losses = model.decode_head.losses(inp, target.unsqueeze(1))
    total = sum(v.mean() for k,v in losses.items() if 'loss' in k)
    assert torch.isfinite(total), 'loss not finite'
    total.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert logits.grad[:, :, 0, :].abs().sum() == 0, 'ignore region grad not zero'

    from hrplus_core import multiclass_lovasz_hinge
    lovasz = multiclass_lovasz_hinge(logits.detach(), target)
    assert torch.isfinite(lovasz)

    report = {'status': 'passed', 'gpu_used': False, 'real_training_run': False,
              'note': 'CPU verification only; supervisor patch (persistent_workers=False) verified via real tools.hrplus_smoke.load_batch_cpu import'}
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print('PREFLIGHT PASSED:', args.output)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        with open('hrplus_cpu_preflight.json', 'w') as f:
            json.dump({'status': 'error', 'error': str(e), 'gpu_used': False, 'real_training_run': False}, f)
        print(f'PREFLIGHT ERROR: {e}', file=sys.stderr)
        sys.exit(1)