import os
import random
from pathlib import Path

import numpy as np
import torch

from .common import digest


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save(path, model, optimizer, scheduler, scaler, completed, identity, next_rows):
    if completed != scheduler.completed:
        raise ValueError('Scheduler/optimizer checkpoint position mismatch')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict() if scaler else None,
                'rng': rng_state(), 'completed': completed, 'cursor': completed*8,
                'identity': identity, 'identity_sha256': digest(identity), 'next_rows_sha256': digest(next_rows)}, temporary)
    os.replace(temporary, path)


def load(path, model, optimizer, scheduler, scaler, identity, rows):
    # Only our own hash-verified checkpoint, never an arbitrary downloaded pickle.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['identity'] != identity or state['identity_sha256'] != digest(identity):
        raise ValueError('Checkpoint identity differs from frozen run')
    completed = state['completed']
    if state['cursor'] != completed*8 or state['next_rows_sha256'] != digest(rows[completed*8:completed*8+8]):
        raise ValueError('Checkpoint sample cursor/next update differs')
    model.load_state_dict(state['model'], strict=True)
    model.set_update(completed)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    if scaler is not None:
        if state['scaler'] is None:
            raise ValueError('Missing FP16 scaler')
        scaler.load_state_dict(state['scaler'])
    if scheduler.completed != completed:
        raise ValueError('Resume scheduler update differs')
    restore_rng(state['rng'])
    return completed
