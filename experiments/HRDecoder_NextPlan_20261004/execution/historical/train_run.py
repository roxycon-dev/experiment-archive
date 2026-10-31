"""One bounded formal run; preflight and campaign state govern all extensions."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from qlib.common import assert_source, digest, read_json, seed_all, sha, utc_now, write_json
from qlib.data import DDR, External, WindowDataset
from qlib.model import build_model
from qlib.optim import make_optimizer, Schedule
from qlib.loss import masked_loss
from qlib.geometry import center_logits
from qlib import checkpoint
from qlib.evaluate import evaluate
from qlib.decision import serious_failure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', default='configs/runtime.server.json')
    ap.add_argument('--arm', choices=['Q_DDR', 'Q_MIX'], required=True)
    ap.add_argument('--seed', type=int, choices=[42, 2024], default=42)
    ap.add_argument('--endpoint', type=int, choices=[16000, 24000], default=16000)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    runtime, protocol = read_json(args.runtime), read_json(root/'configs/protocol.json')
    gate = read_json(root/'reports/formal_readiness.json')
    if gate['status'] != 'READY' or gate['runtime_sha256'] != sha(args.runtime):
        raise ValueError('Runtime changed or GPU/cost acceptance not READY')
    if assert_source(root)['sha256'] != gate['source_sha256']:
        raise ValueError('Code identity differs from accepted preflight')
    if args.seed == 2024 or args.endpoint == 24000:
        campaign = read_json(root/'reports/campaign_state.json')
        if args.seed == 2024 and not campaign.get('review_seed_authorized'):
            raise ValueError('No two-adjacent-node promotion signal')
        if args.endpoint != campaign['common_endpoint']:
            raise ValueError('Endpoint differs from shared campaign decision')
    seed_all(args.seed)
    resources = runtime['resources']
    micro, accum = resources['micro_batch'], resources['accumulation']
    if (micro, accum) not in ((2, 4), (1, 8)):
        raise ValueError('Unregistered micro-batch scheme')
    stream_path = root/'reports'/f'stream_{args.arm}_seed{args.seed}.jsonl'
    stream_info = read_json(stream_path.with_suffix('.summary.json'))
    if sha(stream_path) != stream_info['file_sha256']:
        raise ValueError('Sample stream changed')
    rows = [json.loads(line) for line in stream_path.read_text(encoding='utf-8').splitlines()]
    data = read_json(root/'reports/data_identity.json')
    datasets = {'DDR': DDR(runtime['ddr_root'], 'train', data['splits']['train'])}
    if args.arm == 'Q_MIX':
        external = read_json(runtime['external_manifest'])
        for source in ('IDRiD', 'MAPLES'):
            datasets[source] = External(source, external)
    val = DDR(runtime['ddr_root'], 'val', data['splits']['val'])
    pretraining = read_json(root/'reports/pretraining_identity.json')
    if pretraining['status'] != 'READY':
        raise ValueError('Pretrained source/license/loading acceptance incomplete')
    model, loading = build_model(runtime['dinov3_repo'], runtime['pretrained_checkpoint'], pretraining['sha256'])
    model = model.to('cuda').train()
    optimizer, dtype = make_optimizer(model), {'bfloat16': torch.bfloat16, 'float16': torch.float16}[resources['dtype']]
    scheduler = Schedule(optimizer)
    scaler = torch.amp.GradScaler('cuda') if dtype == torch.float16 else None
    identity = {'arm': args.arm, 'seed': args.seed, 'protocol_sha256': sha(root/'configs/protocol.json'),
        'source_sha256': gate['source_sha256'], 'data_sha256': data['sha256'], 'pretraining_sha256': pretraining['sha256'],
        'stream_sha256': stream_info['file_sha256'], 'resources': resources, 'p0_identity_sha256': sha(root/'configs/p0_identity.json')}
    run = root/'runs'/f'{args.arm}_seed{args.seed}'
    run.mkdir(parents=True, exist_ok=True)
    write_json(run/'initialization.json', loading)
    completed = 0
    if args.resume:
        completed = checkpoint.load(run/'last.pth', model, optimizer, scheduler, scaler, identity, rows)
    elif (run/'last.pth').exists():
        raise ValueError('Existing formal run; explicit identity-checked resume required')
    write_json(run/'identity.json', identity)
    nodes = {int(p.parent.name.removeprefix('val_u')): read_json(p) for p in run.glob('val_u*/metrics.json')}
    formal_nodes = {2000, 4000, 8000, 12000, 16000, 20000, 24000}
    if args.resume and completed in formal_nodes and completed not in nodes:
        nodes[completed] = evaluate(model, val, runtime, run/f'val_u{completed}', completed, keep_probability=completed >= 12000)
        checkpoint.save(run/'last.pth', model, optimizer, scheduler, scaler, completed, identity, rows[completed*8:completed*8+8])
        if completed >= 12000:
            checkpoint.save(run/f'checkpoint_u{completed}.pth', model, optimizer, scheduler, scaler, completed, identity, rows[completed*8:completed*8+8])
    if completed >= 8000 and serious_failure(nodes):
        write_json(run/'status.json', {'stage': 'SERIOUS_FAILURE_STOP', 'completed': completed,
            'all_three_stop_conditions_met': True, 'identity_sha256': digest(identity), 'updated_at': utc_now()})
        return
    total_started = time.perf_counter()
    loader_args = {'batch_size': micro, 'shuffle': False, 'num_workers': resources['workers'], 'pin_memory': True,
                   'generator': torch.Generator().manual_seed(args.seed+333)}
    if resources['workers']:
        loader_args.update(prefetch_factor=resources['prefetch_factor'], persistent_workers=True)
    loader = iter(DataLoader(WindowDataset(datasets, rows, completed*8, args.endpoint*8), **loader_args))
    pending_update = None
    try:
        while completed < args.endpoint:
            pending_update = completed+1
            model.set_update(completed)
            optimizer.zero_grad(set_to_none=True)
            waiting, loss_value = 0., 0.
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(accum):
                wait_start = time.perf_counter()
                x, g, v = next(loader)
                waiting += time.perf_counter()-wait_start
                x, g, v = (t.to('cuda', non_blocking=True) for t in (x, g, v))
                with torch.autocast('cuda', dtype=dtype):
                    z = center_logits(model(x))
                loss = masked_loss(z, g, v)/accum
                loss_value += float(loss.detach())
                (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient; no blind update/restart')
            if scaler:
                before = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < before:
                    raise FloatingPointError('Skipped FP16 update; stop at last consistent checkpoint')
            else:
                optimizer.step()
            completed += 1
            scheduler.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-started
            if completed % 20 == 0:
                with (run/'train.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'update': completed, 'loss': loss_value, 'seconds_per_update': seconds,
                        'data_wait_seconds': waiting, 'grad_norm': float(norm), 'next_lr': [g['lr'] for g in optimizer.param_groups],
                        'time': utc_now()})+'\n')
                write_json(run/'status.json', {'stage': 'TRAINING', 'completed': completed, 'endpoint': args.endpoint,
                    'cursor': completed*8, 'identity_sha256': digest(identity), 'updated_at': utc_now()})
                print(f'{args.arm} seed{args.seed} update{completed}: loss={loss_value:.5f} seconds={seconds:.3f}', flush=True)
            if completed % 200 == 0 or completed in formal_nodes:
                checkpoint.save(run/'last.pth', model, optimizer, scheduler, scaler, completed, identity, rows[completed*8:completed*8+8])
            if completed in formal_nodes:
                # Only registered late candidates get probability caches.
                nodes[completed] = evaluate(model, val, runtime, run/f'val_u{completed}', completed, keep_probability=completed >= 12000)
                checkpoint.save(run/'last.pth', model, optimizer, scheduler, scaler, completed, identity, rows[completed*8:completed*8+8])
                if completed >= 12000:
                    path = run/f'checkpoint_u{completed}.pth'
                    checkpoint.save(path, model, optimizer, scheduler, scaler, completed, identity, rows[completed*8:completed*8+8])
                if completed == 8000 and serious_failure(nodes):
                    write_json(run/'status.json', {'stage': 'SERIOUS_FAILURE_STOP', 'completed': completed,
                        'all_three_stop_conditions_met': True, 'identity_sha256': digest(identity), 'updated_at': utc_now()})
                    return
        write_json(run/'status.json', {'stage': 'COMPLETE', 'completed': completed, 'endpoint': args.endpoint,
            'total_seconds_this_invocation': time.perf_counter()-total_started, 'identity_sha256': digest(identity), 'updated_at': utc_now()})
    except BaseException as exc:
        # Last accepted optimizer boundary remains untouched after partial/nonfinite update.
        write_json(run/'status.json', {'stage': 'TECHNICAL_PAUSE', 'completed_in_memory': completed,
            'pending_update': pending_update, 'reason': f'{type(exc).__name__}: {exc}', 'automatic_restart': False, 'updated_at': utc_now()})
        raise


if __name__ == '__main__':
    main()
