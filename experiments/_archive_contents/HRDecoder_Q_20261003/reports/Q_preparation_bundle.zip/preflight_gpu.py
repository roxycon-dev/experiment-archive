"""The approved pretraining, real-batch wiring, 100-update, learning and full-val gates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from qlib.common import digest, read_json, seed_all, sha, utc_now, write_json
from qlib.data import DDR, WindowDataset
from qlib.model import build_model
from qlib.optim import make_optimizer, Schedule
from qlib.geometry import center_logits
from qlib.loss import masked_loss
from qlib.evaluate import evaluate
from qlib.metrics import counts, summarize
from qlib.cost import estimate_cost
from qlib import checkpoint


ROOT = Path(__file__).resolve().parent


def tensor_digest(parameters):
    h = hashlib.sha256()
    for p in parameters:
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def optimize(model, optimizer, scheduler, batch_iter, micro, accum, dtype, scaler=None):
    optimizer.zero_grad(set_to_none=True)
    waiting, loss_value = 0., 0.
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(accum):
        wait_start = time.perf_counter()
        x, g, v = next(batch_iter)
        waiting += time.perf_counter()-wait_start
        x, g, v = (t.to('cuda', non_blocking=True) for t in (x, g, v))
        if x.shape != (micro, 3, 1024, 1024):
            raise ValueError('True 1024 real micro-batch required')
        if not torch.isfinite(x).all() or float(x.min()) < -3. or float(x.max()) > 3.:
            raise ValueError('RGB normalization range wrong')
        with torch.autocast('cuda', dtype=dtype):
            z = center_logits(model(x))
        value = masked_loss(z, g, v)/accum
        loss_value += float(value.detach())
        (scaler.scale(value) if scaler else value).backward()
    if scaler:
        scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError('Nonfinite true-batch gradient')
    if scaler:
        before = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < before:
            raise FloatingPointError('Skipped optimizer update')
    else:
        optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    return {'seconds': time.perf_counter()-started, 'data_wait_seconds': waiting,
            'loss': loss_value, 'grad_norm': float(grad_norm)}


def loader(dataset, rows, resources, start=0, end=None):
    kwargs = dict(batch_size=resources['micro_batch'], shuffle=False, num_workers=resources['workers'], pin_memory=True,
                  generator=torch.Generator().manual_seed(375))
    if resources['workers']:
        kwargs.update(prefetch_factor=resources['prefetch_factor'], persistent_workers=True)
    return iter(DataLoader(WindowDataset({'DDR': dataset}, rows, start, end), **kwargs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', default='configs/runtime.server.json')
    args = ap.parse_args()
    runtime = read_json(args.runtime)
    resources = runtime['resources']
    if (resources['micro_batch'], resources['accumulation']) not in ((2, 4), (1, 8)):
        raise ValueError('Unregistered micro-batch scheme')
    data = read_json(ROOT/'reports/data_identity.json')
    pretrained = read_json(ROOT/'reports/pretraining_download.json')
    if pretrained['status'] != 'DOWNLOADED_SHA_VERIFIED':
        raise ValueError('Source/license/SHA acceptance incomplete')
    if read_json(ROOT/'reports/cpu_acceptance.json')['status'] != 'PASS' or read_json(ROOT/'reports/p0_acceptance.json')['status'] != 'PASS':
        raise ValueError('Local protocol acceptance incomplete')
    # Every source file is pinned. No host hardware enumeration is performed.
    source = read_json(ROOT/'reports/source_identity.json')
    for rel, expected in source['files'].items():
        if sha(ROOT/rel) != expected:
            raise ValueError('Source changed before GPU preflight: '+rel)
    seed_all(42)
    dataset = DDR(runtime['ddr_root'], 'train', data['splits']['train'])
    validation = DDR(runtime['ddr_root'], 'val', data['splits']['val'])
    # Confirm copied data and fixed P0 cache geometry/metric before training.
    for split in ('train', 'val'):
        for sample in data['splits'][split]['samples']:
            sid = sample['sid']
            if sha(Path(runtime['ddr_root'])/'images'/split/(sid+'.jpg')) != sample['image_sha256'] or sha(Path(runtime['ddr_root'])/'labels'/split/(sid+'.png')) != sample['label_sha256']:
                raise ValueError('Remote DDR identity drift: '+sid)
    stream_path = ROOT/'reports/stream_Q_DDR_seed42.jsonl'
    info = read_json(stream_path.with_suffix('.summary.json'))
    if sha(stream_path) != info['file_sha256']:
        raise ValueError('Transferred stream identity changed')
    rows = [json.loads(line) for line in stream_path.read_text(encoding='utf-8').splitlines()]
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[resources['dtype']]
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 implementation unavailable; lock FP16 before formal comparison')
    model, loading = build_model(runtime['dinov3_repo'], runtime['pretrained_checkpoint'], pretrained['sha256'])
    model = model.to('cuda').train()
    optimizer = make_optimizer(model)
    scheduler = Schedule(optimizer)
    scaler = torch.amp.GradScaler('cuda') if dtype == torch.float16 else None
    micro, accum = resources['micro_batch'], resources['accumulation']
    stage_start = time.perf_counter()
    frozen_digest = tensor_digest(model.backbone.parameters())
    batch_iter = loader(dataset, rows, resources, end=800)
    frozen_times = []
    for step in range(10):
        model.set_update(step)
        frozen_times.append(optimize(model, optimizer, scheduler, batch_iter, micro, accum, dtype, scaler))
    if tensor_digest(model.backbone.parameters()) != frozen_digest:
        raise ValueError('Frozen pretraining backbone changed')
    del batch_iter
    model.set_update(2000)
    scheduler.completed = 2000
    scheduler.prepare_next()
    early_parameters = [p for p in model.backbone.parameters() if not p.requires_grad]
    early_sha = tensor_digest(early_parameters)
    tail_sha = tensor_digest(model.backbone.blocks[8:].parameters())
    head_sha = tensor_digest(model.head.parameters())
    torch.cuda.reset_peak_memory_stats()
    batch_iter = loader(dataset, rows, resources, end=800)
    unfrozen_times = []
    for step in range(100):
        model.set_update(2000+step)
        row = optimize(model, optimizer, scheduler, batch_iter, micro, accum, dtype, scaler)
        unfrozen_times.append(row)
        if step == 0:
            if any(p.grad is not None for p in early_parameters):
                raise ValueError('Early frozen layers received gradient')
            if not all(float(model.head.weight.grad[c].abs().sum()) > 0 for c in range(4)):
                raise ValueError('A lesion head has no gradient')
            for block in model.backbone.blocks[8:]:
                if sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None) == 0:
                    raise ValueError('A last-four block has no gradient')
            if tensor_digest(model.head.parameters()) == head_sha or tensor_digest(model.backbone.blocks[8:].parameters()) == tail_sha:
                raise ValueError('Gradient exists but optimizer did not update required parameters')
        if (step+1) % 20 == 0:
            print(f'Real 1024 last-four backward: {step+1}/100; {row["seconds"]:.3f}s/update', flush=True)
    if tensor_digest(early_parameters) != early_sha:
        raise ValueError('Early-eight/patch parameters changed')
    del batch_iter
    performance = {'real_unfrozen_updates': 100, 'frozen_updates_for_timing': 10,
        'micro_batch': micro, 'accumulation': accum, 'dtype': resources['dtype'],
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'frozen_seconds_per_update_mean': float(np.mean([x['seconds'] for x in frozen_times[2:]])),
        'unfrozen_seconds_per_update_mean': float(np.mean([x['seconds'] for x in unfrozen_times])),
        'unfrozen_seconds_p90': float(np.quantile([x['seconds'] for x in unfrozen_times], .9)),
        'data_wait_seconds_mean': float(np.mean([x['data_wait_seconds'] for x in unfrozen_times])),
        'timing_rows': unfrozen_times, 'attention': 'official_torch_SDPA_no_attention_matrix_logging'}
    # Record the dispatched SDPA implementation, without collecting matrices.
    probe_x = WindowDataset({'DDR': dataset}, rows)[0][0][None].to('cuda')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            model(probe_x)
    performance['sdpa_dispatched_operators'] = sorted({event.key for event in trace.key_averages()
                                                    if '_scaled_dot_product_' in event.key})
    if not performance['sdpa_dispatched_operators']:
        raise ValueError('Official SDPA dispatch not observed')
    del probe_x
    # Actual model+optimizer recovery, next draws/LR/RNG required.
    resume_path = ROOT/'reports/preflight_resume.pth'
    identity = {'phase': 'preflight_only', 'source': source['sha256'], 'pretraining': pretrained['sha256'], 'resources': resources}
    checkpoint.save(resume_path, model, optimizer, scheduler, scaler, 2100, identity, rows[16800:16808])
    before_resume_sha = tensor_digest(model.head.parameters())
    expected_rng = torch.rand(4, device='cuda')
    model.head.weight.data.add_(1.)
    checkpoint.load(resume_path, model, optimizer, scheduler, scaler, identity, rows)
    if tensor_digest(model.head.parameters()) != before_resume_sha or not torch.equal(torch.rand(4, device='cuda'), expected_rng):
        raise ValueError('Real model RNG/weight recovery failed')
    resume_path.unlink()  # disposable preflight file, never a formal checkpoint
    # Fixed eight real train windows; up to 200 frozen-head updates, no val selection.
    del model, optimizer, scheduler
    torch.cuda.empty_cache(); seed_all(42)
    model, _ = build_model(runtime['dinov3_repo'], runtime['pretrained_checkpoint'], pretrained['sha256'])
    model = model.to('cuda').train()
    optimizer, scheduler = make_optimizer(model), None
    scheduler = Schedule(optimizer)
    fixed = [WindowDataset({'DDR': dataset}, rows)[i] for i in range(8)]
    def batches():
        for offset in range(0, 8, micro):
            yield tuple(torch.stack([sample[k] for sample in fixed[offset:offset+micro]]) for k in range(3))
    learning = []
    for step in range(200):
        model.set_update(step)
        record = optimize(model, optimizer, scheduler, iter(batches()), micro, accum, dtype, scaler)
        learning.append(record['loss'])
        if (step+1) % 50 == 0:
            print(f'Fixed real-sample wiring learning: {step+1}/200 loss={record["loss"]:.5f}', flush=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        predicted = center_logits(model(fixed[4][0][None].to('cuda'))).float().sigmoid()
    initial, final = float(np.mean(learning[:10])), float(np.mean(learning[-10:]))
    spatial_std = float(predicted.flatten(2).std(dim=2).max())
    if final >= initial*.9 or spatial_std < 1e-4:
        raise ValueError('Fixed-sample learnability failed: inspect wiring, no blind formal launch')
    write_json(ROOT/'reports/learnability.json', {'status': 'PASS', 'updates': 200, 'initial_loss': initial,
        'final_loss': final, 'probability_std': float(predicted.std()), 'max_class_spatial_std': spatial_std, 'loss_curve': learning,
        'formal_steps_carried_over': 0, 'selection_use': False})
    # Reset model/optimizer/RNG before the full-val timing. All preflight weights are disposable.
    del model, optimizer, scheduler, predicted
    torch.cuda.empty_cache(); seed_all(42)
    model, _ = build_model(runtime['dinov3_repo'], runtime['pretrained_checkpoint'], pretrained['sha256'])
    model = model.to('cuda')
    validation_timing = evaluate(model, validation, runtime, ROOT/'reports/preflight_full_val', 0, False)
    accepted_p0 = read_json(ROOT/'reports/p0_acceptance.json')
    expected_rows = {r['sid']: r['counts'] for r in accepted_p0['per_image']}
    if validation_timing['p0']['counts'] != accepted_p0['counts'] or any(r['p0'] != expected_rows[r['sid']] for r in validation_timing['per_image']):
        raise ValueError('Server P0 cache does not reproduce accepted per-image counts')
    performance['full_76_val_seconds'] = validation_timing['seconds']
    performance['full_76_val_is_initial_model_not_a_candidate'] = True
    costs = estimate_cost(performance['frozen_seconds_per_update_mean'], performance['unfrozen_seconds_per_update_mean'],
                          validation_timing['seconds'], runtime['budget']['hourly_rate_cny'])
    write_json(ROOT/'reports/gpu_preflight.json', {'status': 'PASS', 'loading': loading, 'performance': performance,
        'cost_rows': costs, 'total_preflight_seconds': time.perf_counter()-stage_start, 'created_at': utc_now(),
        'test_read': False, 'formal_training_started': False})
    write_json(ROOT/'reports/pretraining_identity.json', {'status': 'READY', **pretrained, **loading, 'status': 'READY'})
    write_json(ROOT/'reports/formal_readiness.json', {'status': 'READY', 'runtime_sha256': sha(args.runtime),
        'source_sha256': source['sha256'], 'gpu_preflight_sha256': sha(ROOT/'reports/gpu_preflight.json'),
        'budget_authorization': runtime['budget']['approval'], 'formal_training_started': False,
        'micro_batch_locked': micro, 'accumulation_locked': accum, 'dtype_locked': resources['dtype'], 'created_at': utc_now()})
    print('FORMAL_PREPARATION_READY; no formal run was launched', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write_json(ROOT/'reports/gpu_preflight_failure.json', {'status': 'TECHNICAL_PAUSE',
            'reason': f'{type(exc).__name__}: {exc}', 'automatic_restart': False, 'created_at': utc_now()})
        raise
