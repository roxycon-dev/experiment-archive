"""Full optimizer/scheduler/EMA/RNG and two independent stream cursors."""
import copy
import os
from pathlib import Path
import torch
from qlib.common import digest
from qlib.checkpoint import rng_state, restore_rng


class EMA:
    def __init__(self, model):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.completed = 0

    @torch.no_grad()
    def update(self, model):
        current = model.state_dict()
        floating_old, floating_new = [], []
        for name, value in self.model.state_dict().items():
            if value.is_floating_point():
                floating_old.append(value)
                floating_new.append(current[name])
            else:
                value.copy_(current[name])
        torch._foreach_mul_(floating_old, .999)
        torch._foreach_add_(floating_old, floating_new, alpha=.001)
        self.completed += 1


def cursor(completed, arm):
    return {'DDR': completed * 8, 'TJDR': min(completed, 20000) * 4 if arm in ('D', 'SD') else 0}


def save(path, model, optimizer, scheduler, ema, completed, identity, next_rows):
    assert completed == scheduler.completed == ema.completed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(),
                'ema_completed': ema.completed, 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'rng': rng_state(),
                'completed': completed, 'stream_cursor': cursor(completed, identity['arm']),
                'identity': identity, 'identity_sha256': digest(identity),
                'next_rows_sha256': digest(next_rows)}, temporary)
    os.replace(temporary, path)


def load(path, model, optimizer, scheduler, ema, identity, next_rows_at):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['identity'] == identity and saved['identity_sha256'] == digest(identity)
    completed = saved['completed']
    assert saved['stream_cursor'] == cursor(completed, identity['arm'])
    assert saved['next_rows_sha256'] == digest(next_rows_at(completed))
    model.load_state_dict(saved['model'], strict=True)
    ema.model.load_state_dict(saved['ema'], strict=True)
    ema.completed = saved['ema_completed']
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    assert completed == scheduler.completed == ema.completed
    restore_rng(saved['rng'])
    return completed
