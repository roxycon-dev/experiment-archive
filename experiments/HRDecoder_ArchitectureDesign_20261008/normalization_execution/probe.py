"""Paired warm diagnostic only. Never writes the original run or reads test labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'residual_execution'))
import train as core
from intervention import apply_intervention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['control', 'gt_stratified'], required=True)
    args = parser.parse_args()
    assert torch.__version__ == '2.0.1+cu117' and torch.cuda.is_available()
    plan = json.loads((HERE / 'probe_protocol.json').read_text())
    assert plan['warm_updates_per_arm'] == 500 and plan['canvas'] == 2048 and plan['batch'] == 2
    parent_path = ROOT / 'training_runs/residual_normalization/parent_08000.pt'
    parent = torch.load(parent_path, map_location='cpu')
    material, old_digest = core.identity()
    assert old_digest == plan['parent_identity_SHA'] == parent['identity_SHA']
    assert parent['completed'] == plan['parent_completed'] == 8000
    assert parent['optimizer']['state'] and 'rng' in parent
    cpu = json.loads((HERE / 'NORMALIZATION_CPU_PASS.json').read_text())
    assert cpu['status'] == 'PASS' and cpu['torch'] == torch.__version__
    identity = {
        'kind': 'paired warm diagnostic; not a fresh40k model', 'arm': args.arm,
        'parent_identity_SHA': old_digest, 'parent_completed': parent['completed'],
        'parent_hardlink_inode': parent_path.stat().st_ino,
        'protocol': plan,
        'code': {name: core.file_sha(HERE / name) for name in
                 ('probe.py', 'intervention.py', 'probe_protocol.json')},
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    work = ROOT / 'training_runs/residual_normalization' / args.arm
    work.mkdir(parents=True, exist_ok=True)
    assert not (work / 'FORMAL_PROBE_STARTED.json').exists(), 'No implicit repeated probe'
    core.WORK = work  # evaluate() writes only this arm's validation progress.
    core.install_legacy_bf16_interpolation()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    baseline, _ = core.build_baseline(core.REPO, core.W48,
                                    verified_weight_sha=material['ImageNet_W48_SHA'])
    model = apply_intervention(core.SACDStudent(baseline), args.arm).cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9,
                               weight_decay=.0005, foreach=True)
    ready = core.install_ram(ROOT)
    from data import TrainCacheDataset
    dataset = TrainCacheDataset(ROOT, core.DATA)

    def reload_parent():
        model.load_state_dict(parent['model'], strict=True)
        optimizer.load_state_dict(parent['optimizer'])
        core.restore_rng(parent['rng'])
        model.train()

    reload_parent()
    smoke_times, calls = [], []
    hook = model.base.backbone.register_forward_hook(lambda *unused: calls.append(1))
    sampler = core.ResumableBatches(len(dataset), 8000, 8003)
    loader = core.CUDAPrefetch(dataset, sampler, workers=12)
    for index, batch in enumerate(loader, 8000):
        assert tuple(batch['image'].shape) == (2, 3, 2048, 2048)
        for group in optimizer.param_groups:
            group['lr'] = core.learning_rate(index)
        started = time.monotonic()
        metrics = core.one_step(model, optimizer, batch, index + 1)
        smoke_times.append(time.monotonic() - started)
        nonzero = {name: any(p.grad is not None and bool(p.grad.ne(0).any()) for p in part.parameters())
                   for name, part in [('local', model.private.local),
                                      ('global', model.private.global_path), ('head', model.private.head)]}
        assert all(nonzero.values())
        print(json.dumps({'arm': args.arm, 'discarded_smoke_step': index - 7999,
                          'losses': metrics, 'seconds': smoke_times[-1]}), flush=True)
    hook.remove()
    assert len(calls) == 3
    del batch
    torch.cuda.synchronize()
    smoke = {'status': 'PASS', 'arm': args.arm, 'identity_SHA': digest,
             'canvas': 2048, 'batch': 2, 'discarded_optimizer_steps': 3, 'formal_updates': 0,
             'finite_losses_and_all_gradients': True, 'backbone_calls': len(calls),
             'private_nonzero_gradients': nonzero, 'step_seconds': smoke_times,
             'device': torch.cuda.get_device_name(0), 'accuracy_validated': False}
    core.write_json(work / 'GPU_SMOKE_PASS.json', smoke)
    # All smoke weights, momentum, BN buffers and RNG are discarded before the measured arm.
    reload_parent()
    core.write_json(work / 'protocol.json', {'identity_SHA': digest, 'identity': identity,
                    'parent_model_optimizer_RNG_loaded': True, 'teacher_cache_bytes': ready['bytes']})
    core.write_json(work / 'FORMAL_PROBE_STARTED.json', {'identity_SHA': digest, 'time_unix': time.time()})
    initial = core.evaluate(model)
    initial.update(parent_completed=8000, probe_updates=0, identity_SHA=digest, arm=args.arm)
    core.write_json(work / 'val_probe_00000.json', initial)
    # Reuse the same persistent worker pool. Forking a second pool after CUDA
    # initialization and CPU validation/OpenCV can inherit locked native mutexes.
    sampler.start, sampler.stop = 8000, 8500
    times = []
    for index, batch in enumerate(loader, 8000):
        for group in optimizer.param_groups:
            group['lr'] = core.learning_rate(index)
        started = time.monotonic()
        metrics = core.one_step(model, optimizer, batch, index + 1)
        times.append(time.monotonic() - started)
        completed = index - 7999
        if completed % 25 == 0:
            progress = {'status': 'DIAGNOSTIC_TRAINING', 'arm': args.arm, 'probe_updates': completed,
                        'parent_completed': 8000, 'expected_probe_updates': 500,
                        'identity_SHA': digest, 'losses': metrics, 'time_unix': time.time(),
                        'mean_compute_seconds': statistics.mean(times[-100:])}
            core.write_json(work / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
        if completed in (250, 500):
            result = core.evaluate(model)
            result.update(parent_completed=8000, probe_updates=completed, arm=args.arm, identity_SHA=digest)
            core.write_json(work / ('val_probe_%05d.json' % completed), result)
            print(json.dumps({'probe_validation': result}), flush=True)
    assert completed == 500
    core.save_checkpoint(work / 'probe_last.pt', {'model': model.state_dict(),
        'optimizer': optimizer.state_dict(), 'rng': core.rng_state(),
        'parent_completed': 8000, 'probe_updates': completed, 'identity_SHA': digest,
        'kind': 'diagnostic only; never a fresh40k checkpoint'})
    core.write_json(work / 'PROBE_COMPLETE.json', {'status': 'COMPLETE', 'arm': args.arm,
        'identity_SHA': digest, 'parent_completed': 8000, 'probe_updates': completed,
        'full_val_points': [0, 250, 500], 'test_evaluated': False, 'time_unix': time.time()})


if __name__ == '__main__':
    main()
