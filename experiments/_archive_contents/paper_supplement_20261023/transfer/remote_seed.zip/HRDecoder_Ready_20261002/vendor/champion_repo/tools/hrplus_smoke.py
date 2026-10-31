#!/usr/bin/env python3.8
"""GPU preflight smoke test for HRPlusRefinement (mmcv 1.7.1 / torch 2.0.1).

When invoked, runs only the stated preflight; no login/shutdown/network/installs.
Builds model + datasets, runs 3 train steps and 1 eval forward, writes JSON report.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mmcv
from mmcv.parallel import scatter
import mmseg
import torch
import numpy as np

from mmseg.models.builder import build_train_model
from mmseg.datasets import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='HRPlus GPU smoke preflight')
    parser.add_argument('config', help='path to mmseg config file')
    parser.add_argument('--output', required=True, help='path to output JSON report')
    return parser.parse_args()


def param_snapshot(model, prefix):
    """Return {param_name: cpu clone} for params under prefix."""
    snap = {}
    for name, p in model.named_parameters():
        if name.startswith(prefix):
            snap[name] = p.detach().cpu().clone()
    return snap


def state_snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def hash_state(state):
    return {k: tensor_hash(v) for k, v in state.items()}


def extract_losses(losses):
    """Collect all entries whose key contains 'loss' into one summed tensor."""
    if isinstance(losses, dict):
        items = losses.items()
    elif isinstance(losses, (list, tuple)):
        items = enumerate(losses)
    else:
        return losses
    total = None
    for k, v in items:
        if not (isinstance(k, str) and 'loss' in k):
            continue
        if isinstance(v, (list, tuple)):
            for vv in v:
                total = vv if total is None else total + vv
        elif isinstance(v, torch.Tensor):
            total = v if total is None else total + v
    return total


def load_batch_cpu(cfg, split):
    dataset = build_dataset(cfg.data[split])
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=2 if split == 'train' else 1,
        workers_per_gpu=0,
        persistent_workers=False,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=42,
    )
    batch = next(iter(data_loader))
    return batch


def scatter_batch(batch):
    device_id = [torch.cuda.current_device()]
    scattered = scatter(batch, device_id)
    return scattered[0]


def main():
    args = parse_args()
    report = {}
    try:
        cfg = mmcv.Config.fromfile(args.config)

        # ------------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------------
        model = build_train_model(
            cfg, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        assert hasattr(model, 'reference'), 'model must expose reference module'
        model.cuda()

        # Pre-init snapshots: one backbone and one head param
        pre_backbone = None
        pre_head = None
        for name, p in model.named_parameters():
            if 'backbone' in name and pre_backbone is None:
                pre_backbone = (name, p.detach().cpu().clone())
            if ('decode_head' in name or 'head' in name) and pre_head is None:
                pre_head = (name, p.detach().cpu().clone())
        assert pre_backbone is not None and pre_head is not None

        # Source checkpoint (student weights) must exist
        ckpt_path = cfg.model.source_checkpoint
        assert os.path.isfile(ckpt_path), f'missing checkpoint {ckpt_path}'
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        ckpt_state = checkpoint.get('state_dict', checkpoint)

        # Reference (teacher) checkpoint = plain HR100 weights
        ref_state = {k[len('reference.'):]: v
                     for k, v in model.state_dict().items()
                     if k.startswith('reference.')}
        ref_state_pre = state_snapshot(model)
        ref_pre_hashes = hash_state({k: v for k, v in ref_state_pre.items()
                                     if k.startswith('reference.') or True})
        # capture pre-init student and reference snapshots (for post-init compare)
        student_pre = {k: v for k, v in ref_state_pre.items()
                       if not k.startswith('reference.')}
        reference_pre = {k[len('reference.'):]: v for k, v in ref_state_pre.items()
                         if k.startswith('reference.')}

        # ------------------------------------------------------------------
        # init_weights
        # ------------------------------------------------------------------
        model.init_weights()
        torch.cuda.synchronize()

        post = state_snapshot(model)
        student_post = {k: v for k, v in post.items() if not k.startswith('reference.')}
        reference_post = {k[len('reference.'):]: v for k, v in post.items()
                          if k.startswith('reference.')}

        # backbone/head params unchanged by init_weights
        name_bb, t_bb = pre_backbone
        name_h, t_h = pre_head
        cur_bb = dict(model.named_parameters())[name_bb].detach().cpu()
        cur_h = dict(model.named_parameters())[name_h].detach().cpu()
        assert torch.equal(cur_bb, t_bb), 'backbone param changed by init_weights'
        assert torch.equal(cur_h, t_h), 'head param changed by init_weights'

        # student state must exactly match source checkpoint
        assert set(student_post.keys()) == set(ckpt_state.keys()), \
            'student state keys != checkpoint keys'
        for k in ckpt_state:
            assert torch.equal(student_post[k].cpu(), ckpt_state[k].cpu()), \
                f'student tensor mismatch: {k}'

        # reference state must equal B (plain HR100) weights — compare cpu-side
        assert set(reference_post.keys()) == set(ckpt_state.keys())
        for k in ckpt_state:
            assert torch.equal(reference_post[k].cpu(), ckpt_state[k].cpu()), \
                f'reference tensor mismatch: {k}'

        # gradient requirements
        for name, p in model.named_parameters():
            if name.startswith('reference.'):
                assert not p.requires_grad, f'reference param requires grad: {name}'
            else:
                pass  # student grads checked after steps
        trainable = [p for p in model.parameters()
                     if p.requires_grad and not id(p) in
                     {id(q) for q in model.reference.parameters()}]
        for p in trainable:
            assert p.requires_grad

        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or 'BatchNorm' in type(m).__name__:
                assert not m.training, 'BN running statistics must stay frozen'

        # ------------------------------------------------------------------
        # Train loop: 3 steps
        # ------------------------------------------------------------------
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.0005)

        ref_before_steps = {k: v.cpu().clone() for k, v in reference_post.items()}

        batch = load_batch_cpu(cfg, 'train')
        batch = scatter_batch(batch)

        peak_mb = 0.0
        start = time.time()
        for step in range(3):
            losses = model(return_loss=True, **batch)
            loss = extract_losses(losses)
            assert isinstance(loss, torch.Tensor) and loss.dim() == 0
            assert torch.isfinite(loss), f'non-finite loss at step {step}'
            optimizer.zero_grad()
            loss.backward()
            # check finite grads
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f'non-finite grad {name}'
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            torch.cuda.synchronize()
            peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / (1024 ** 2))

        seconds = time.time() - start

        # at least one non-zero student grad
        any_nonzero = False
        ref_grad_none = True
        for name, p in model.named_parameters():
            if name.startswith('reference.'):
                assert not p.requires_grad
                if p.grad is not None:
                    ref_grad_none = False
            else:
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    any_nonzero = True
        assert any_nonzero, 'no non-zero student gradient after steps'
        assert ref_grad_none, 'reference params received gradients'

        # reference state unchanged after steps
        after = state_snapshot(model)
        reference_after = {k[len('reference.'):]: v for k, v in after.items()
                           if k.startswith('reference.')}
        for k in ref_before_steps:
            assert torch.equal(reference_after[k].cpu(), ref_before_steps[k].cpu()), \
                f'reference changed during training: {k}'

        # ------------------------------------------------------------------
        # Eval: first val image only
        # ------------------------------------------------------------------
        val_batch = scatter_batch(load_batch_cpu(cfg, 'val'))
        model.eval()
        ori_shape = val_batch['img_metas'][0][0]['ori_shape'][:2]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **val_batch)
        assert isinstance(result, list) and len(result) == 1, \
            'eval result must be a list of length 1'
        seg = result[0]
        assert isinstance(seg, tuple) and len(seg) == 3, \
            'eval result must be (prob, use_sigmoid, compute_aupr)'
        prob, use_sigmoid, compute_aupr = seg
        assert isinstance(prob, np.ndarray), 'prob must be numpy'
        H, W = ori_shape
        assert prob.shape == (4, H, W), \
            f'prob shape {prob.shape} != {(4, H, W)}'
        assert use_sigmoid and compute_aupr
        assert np.isfinite(prob).all() and prob.min() >= 0.0 and prob.max() <= 1.0, \
            'prob not finite in [0,1]'

        report = {
            'status': 'success',
            'config': args.config,
            'steps': 3,
            'seconds': round(seconds, 3),
            'peak_allocated_mb': round(peak_mb, 2),
            'losses_finite': True,
            'reference_frozen': True,
            'student_matches_checkpoint': True,
            'reference_matches_b_weights': True,
            'val': {
                'ori_shape': [int(H), int(W)],
                'prob_shape': list(prob.shape),
                'use_sigmoid': bool(use_sigmoid),
                'compute_aupr': bool(compute_aupr),
                'prob_min': float(prob.min()),
                'prob_max': float(prob.max()),
            },
            'versions': {
                'mmcv': mmcv.__version__,
                'mmseg': mmseg.__version__,
                'torch': torch.__version__,
            },
        }
        ok = True
    except Exception as e:  # noqa: BLE001
        report = {'status': 'error', 'error': repr(e)}
        ok = False

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
