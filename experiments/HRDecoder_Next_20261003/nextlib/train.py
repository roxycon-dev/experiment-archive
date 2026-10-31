"""Paired 0/1k/2k/4k/8k training, persistent exact stream cursor and optimizer."""
import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

from trainer.loss import masked_bce_dice_loss
from .common import write_json, read_json, digest, sha, trusted_load
from .data import replay
from .geometry import patch_tensor, global_view, align_global, supervision, center_logits, canvas
from .model import build_fusion
from .evaluate import evaluate, fusion_cache
from .decision import paired_stop, seed_gate, best_candidate
from .global_cache import exact_global, cache_stats


class ReplayDataset(Dataset):
    def __init__(self, raw, stream, first_draw):
        self.raw, self.stream, self.first = raw, stream, first_draw

    def __len__(self):
        return len(self.stream['rows'])-self.first

    def __getitem__(self, ix):
        row = self.stream['rows'][self.first+ix]
        rgb, lab, tile = replay(self.raw, row, self.stream['geometry'],
                                self.stream['center'], self.stream['halo'])
        return rgb, lab, tile, row


def passthrough(batch):
    return batch


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def save_checkpoint(path, value):
    tmp = Path(path).with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def local_state(model):
    # Frozen D is separately SHA-identified; avoid repeating its 263MB in every checkpoint.
    return {k: v for k, v in model.state_dict().items() if not k.startswith('global_branch.')}


def load_local_state(model, state):
    own = model.state_dict()
    required = {k for k in own if not k.startswith('global_branch.')}
    if set(state) != required:
        raise ValueError('Fusion checkpoint key set changed')
    own.update(state)
    model.load_state_dict(own, strict=True)


def prune_unselected_probabilities(run_dir, retained_update):
    """Prune only this new run's derived npz arrays; all node metrics/counts remain."""
    run_dir=Path(run_dir).resolve()
    if (Path(__file__).resolve().parents[1]/'work').resolve() not in run_dir.parents:
        # Synthetic acceptance directories are intentionally never pruned here.
        return
    for cache in run_dir.glob('val_u*'):
        if not cache.is_dir() or cache.name==f'val_u{retained_update}':continue
        resolved=cache.resolve()
        if resolved.parent!=run_dir:raise ValueError('Cache path escaped this new run')
        index_path=cache/'index.json'
        if not index_path.exists():continue
        index=read_json(index_path)
        for npz in cache.glob('*.npz'):
            if npz.resolve().parent!=resolved:raise ValueError('Probability path escaped this new run')
            npz.unlink()
        index['probabilities_removed_after_not_selected']=True
        write_json(index_path,index)


def gradient_evidence(model, after_projection=False):
    groups = {'decoder': [], 'stage2': [], 'stage3': [], 'global_adapter_last': []}
    if model.has_detail:
        groups.update(detail_projection=[], detail_stem=[])
    for name, p in model.named_parameters():
        if name.startswith('global_branch.') and (p.requires_grad or p.grad is not None):
            raise ValueError('Frozen D received a gradient')
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            raise ValueError(f'Nonfinite gradient: {name}')
        norm = float(p.grad.float().norm())
        if name.startswith('local.decoder.'): groups['decoder'].append(norm)
        if name.startswith('local.encoder.stages.2.'): groups['stage2'].append(norm)
        if name.startswith('local.encoder.stages.3.'): groups['stage3'].append(norm)
        if name.startswith('global_adapter.3.'): groups['global_adapter_last'].append(norm)
        if name.startswith('detail_project'): groups['detail_projection'].append(norm)
        if name.startswith('detail_stem'): groups['detail_stem'].append(norm)
    report = {k: max(v, default=0) for k, v in groups.items()}
    required = [k for k in groups if k != 'detail_stem' or after_projection]
    if not all(report[k] > 0 for k in required):
        raise ValueError(f'Disconnected trainable path: {report}')
    report['D_no_gradient'] = True
    return report


def train_to_node(settings, protocol, stream, train_ds, val_ds, lock, arm, seed, node, run_dir, baseline_cache):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = dict(protocol_sha256=digest(protocol), stream_sha256=stream['rows_sha256'],
        lock_sha256=digest(lock), arm=arm, seed=seed, micro_batch=protocol['micro_batch'],
        grad_accum=protocol['grad_accum'], data_sha256=lock['data_sha256'], source_sha256=lock['source_sha256'])
    history_path = run_dir/'history.json'
    history = read_json(history_path) if history_path.exists() else []
    if any(r['update'] == node for r in history):
        return history
    model, report = build_fusion(settings, lock['start_asset'], arm, seed)
    model.local.gradient_checkpointing = protocol.get('gradient_checkpointing', False)
    model.to('cuda')
    if protocol.get('optimizer','AdamW')!='AdamW':raise ValueError('Undeclared optimizer change')
    optimizer = torch.optim.AdamW(model.parameter_groups(protocol.get('pretrained_lr',1e-5),
            protocol.get('new_lr',1e-4)), weight_decay=protocol.get('weight_decay',.01))
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    update, cursor, amp_skips = 0, 0, 0
    checkpoint = run_dir/'last.pth'
    if checkpoint.exists():
        ck = trusted_load(checkpoint)
        if ck['identity'] != identity:
            raise ValueError('Resume identity mismatch')
        load_local_state(model, ck['model_state'])
        optimizer.load_state_dict(ck['optimizer'])
        scaler.load_state_dict(ck['scaler'])
        update, cursor, amp_skips = ck['update'], ck['stream_cursor'], ck['amp_skips']
        restore_rng(ck['rng'])
    else:
        torch.manual_seed(seed+2)
        torch.cuda.manual_seed_all(seed+2)
        np.random.seed(seed)
        random.seed(seed)
        write_json(run_dir/'initialization.json', report)
        write_json(run_dir/'resolved_protocol.json', protocol)
        write_json(run_dir/'identity.json', identity)
    if update > node:
        raise ValueError('Missing history for an already passed node; recover artifact before continuing')
    effective = protocol['micro_batch']*protocol['grad_accum']
    if effective != 8 or stream['effective_batch'] != effective:
        raise ValueError('Both arms must use the locked effective batch eight')
    workers = protocol.get('workers', 4)
    loader = DataLoader(ReplayDataset(train_ds, stream, cursor), batch_size=effective,
        shuffle=False, num_workers=workers, collate_fn=passthrough,
        persistent_workers=workers > 0, prefetch_factor=1 if workers else None,
        generator=torch.Generator().manual_seed(seed+100))
    iterator = iter(loader)
    foreground = np.zeros(4, np.int64)
    used = 0
    started = time.time()
    segment_start_update = update
    while update < node:
        try:
            batch = next(iterator)
        except StopIteration:
            raise RuntimeError('Paired stream exhausted; no unrecorded replacement samples allowed')
        if len(batch) != effective:
            raise ValueError('Incomplete effective batch')
        model.train()
        # Index LR by successful paired update, not wall time or discarded data.
        warmup = protocol['warmup_updates']
        if update < warmup:
            lr_scale = (update+1)/warmup
        else:
            phase = min(1., (update-warmup)/(protocol['updates']-warmup))
            lr_scale = .01+.99*(1+math.cos(math.pi*phase))/2
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*lr_scale
        optimizer.zero_grad(set_to_none=True)
        losses = []
        batch_foreground = np.zeros(4,np.int64)
        for offset in range(0, effective, protocol['micro_batch']):
            local, aligned, targets, valid = [], [], [], []
            for rgb, label, tile, row in batch[offset:offset+protocol['micro_batch']]:
                x = patch_tensor(rgb, tile, 'cuda')
                # The branch sees this actual transformed whole image, never a rotated old feature.
                with torch.no_grad():
                    if protocol.get('frozen_global_cache', False):
                        feature, ledger = exact_global(model.global_branch, rgb,
                            protocol['global_long_side'], (settings['assets']['D']['sha256'],
                            lock['data_sha256'], lock['source_sha256']),
                            protocol.get('frozen_global_cache_GiB', 48))
                    else:
                        view, ledger = global_view(rgb, protocol['global_long_side'], 'cuda')
                        feature = model.global_branch(view)
                    g = align_global(feature, ledger, tile)
                target, v = supervision(label, tile, 'cuda')
                local.append(x); aligned.append(g); targets.append(target); valid.append(v)
                batch_foreground += target.flatten(2).any(dim=2).cpu().numpy()[0].astype(np.int64)
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(torch.cat(local), torch.cat(aligned))
                logits = center_logits(logits, batch[0][2])
            loss, parts = masked_bce_dice_loss(logits, torch.cat(targets), torch.cat(valid))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; checkpoint remains at previous complete update')
            scaler.scale(loss/protocol['grad_accum']).backward()
            losses.append(float(loss.detach()))
        scaler.unscale_(optimizer)
        if update < 2:
            evidence = gradient_evidence(model, after_projection=update > 0)
            write_json(run_dir/f'gradient_update_{update+1}.json', evidence)
        torch.nn.utils.clip_grad_norm_(list(p for p in model.parameters() if p.requires_grad), protocol.get('grad_clip',1.))
        before = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < before:
            # Replay this SAME batch; unequal AMP skips cannot shift M0/M1 data exposure.
            amp_skips += 1
            if amp_skips > 50:
                raise FloatingPointError('Repeated AMP overflow; do not silently change one arm')
            iterator = iter(DataLoader(ReplayDataset(train_ds, stream, cursor), batch_size=effective,
                shuffle=False, num_workers=0, collate_fn=passthrough))
            continue
        update += 1
        cursor += effective
        foreground += batch_foreground
        used += effective
        if update % 100 == 0:
            print(f'{arm} seed={seed} u={update}/{node} loss={np.mean(losses):.5f} '
                  f'sec/update={(time.time()-started)/max(1,update-segment_start_update):.3f}', flush=True)
        if update % 250 == 0 or update == node:
            save_checkpoint(checkpoint, dict(identity=identity, update=update, stream_cursor=cursor,
                model_state=local_state(model), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                rng=rng_state(), amp_skips=amp_skips, history=history,
                resolved_protocol=protocol, initialization=report))
            write_json(run_dir/'progress.json', dict(update=update, target=node, stream_cursor=cursor,
                amp_skips=amp_skips, center_foreground_draws=foreground.tolist(), draws_counted=used,
                frozen_global_cache=cache_stats(),
                seconds_this_segment=time.time()-started))
    def predict(rgb):
        work, _ = canvas(rgb, None, lock['canvas'])
        logits = model.predict(work, protocol['center'], protocol['halo'], protocol['global_long_side'])
        if work.shape[:2] != rgb.shape[:2]:
            logits = F.interpolate(logits, rgb.shape[:2], mode='bilinear', align_corners=False)
        return logits
    cache = run_dir/f'val_u{node}'
    solo = evaluate(val_ds, predict, cache, dict(**identity, update=node), baseline_cache)
    fused = fusion_cache(val_ds, baseline_cache, cache, run_dir/f'half_u{node}', .5)
    result = dict(arm=arm, seed=seed, update=node, solo=solo, fixed_half=fused)
    before_best = None if not history else best_candidate(history)
    history.append(result)
    write_json(history_path, history)
    now_best = best_candidate(history)
    if before_best is None or now_best != before_best:
        save_checkpoint(run_dir/'best.pth', dict(identity=identity, update=node,
            model_state=local_state(model), resolved_protocol=protocol, initialization=report,
            selected=now_best, stream_cursor=cursor))
    if protocol.get('cache_policy')=='G_all_M_best_per_arm':
        prune_unselected_probabilities(run_dir,now_best['update'])
    # Persist history even at the zero-update node for robust process restarts.
    save_checkpoint(checkpoint, dict(identity=identity, update=update, stream_cursor=cursor,
        model_state=local_state(model), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
        rng=rng_state(), amp_skips=amp_skips, history=history, resolved_protocol=protocol, initialization=report))
    del model, optimizer, scaler, iterator, loader
    torch.cuda.empty_cache()
    return history


def run_pair(settings, protocol, stream, train_ds, val_ds, lock, seed, root, baseline_cache):
    selection_path=Path(root)/'selection.json'
    pair_identity=dict(protocol_sha256=digest(protocol),stream_sha256=stream['rows_sha256'],
                       geometry_sha256=digest(lock),seed=seed)
    if selection_path.exists():
        saved=read_json(selection_path)
        if saved.get('identity')!=pair_identity:raise ValueError('Completed paired run identity differs')
        return saved['second_seed_gate'],saved['winner']
    histories = {}
    for node in protocol['nodes']:
        for arm in ('M0', 'M1'):
            histories[arm] = train_to_node(settings, protocol, stream, train_ds, val_ds, lock,
                                           arm, seed, node, Path(root)/arm, baseline_cache)
        stopped, reason = paired_stop(histories, lock['baseline_val_miou'])
        write_json(Path(root)/'paired_state.json', dict(node=node, stop=stopped and node < protocol['updates'],
            reason=reason, best={arm:best_candidate(h) for arm,h in histories.items()}))
        if stopped and node < protocol['updates']:
            break
    gate, winner = seed_gate(histories, lock['baseline_val_miou'])
    write_json(Path(root)/'selection.json', dict(identity=pair_identity,winner=winner, second_seed_gate=gate,
        engineering_gain_threshold=.005, histories=[str(Path(root)/arm/'history.json') for arm in histories]))
    return gate, winner
