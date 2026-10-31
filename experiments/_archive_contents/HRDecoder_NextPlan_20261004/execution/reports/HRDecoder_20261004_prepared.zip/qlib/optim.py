import math

import torch


def make_optimizer(model):
    planned = {id(p) for block in model.backbone.blocks[8:] for p in block.parameters()}
    planned |= {id(p) for p in model.backbone.norm.parameters()}
    all_backbone = {id(p) for p in model.backbone.parameters()}
    groups = {('head', True): [], ('head', False): [], ('backbone', True): [], ('backbone', False): []}
    for name, p in model.named_parameters():
        if id(p) in all_backbone and id(p) not in planned:
            continue
        family = 'backbone' if id(p) in planned else 'head'
        decay = p.ndim > 1 and not name.endswith('.bias')
        groups[family, decay].append(p)
    return torch.optim.AdamW([{'params': ps, 'family': family, 'lr': 0., 'weight_decay': .01 if decay else 0.}
                             for (family, decay), ps in groups.items() if ps])


def lr_for_update(family, update):
    if not 1 <= update <= 24000:
        raise ValueError('Optimizer update outside registered horizon')
    if family == 'head':
        base, first, peak = 3e-4, 1, 200
    else:
        base, first, peak = 5e-6, 2001, 2500
    if update < first:
        return 0.
    if update <= peak:
        return base * (update-first+1)/(peak-first+1)
    factor = .01 + .99 * .5 * (1 + math.cos(math.pi*(update-peak)/(24000-peak)))
    return base * factor


class Schedule:
    def __init__(self, optimizer):
        self.optimizer, self.completed = optimizer, 0
        self.prepare_next()

    def prepare_next(self):
        if self.completed < 24000:
            for group in self.optimizer.param_groups:
                group['lr'] = lr_for_update(group['family'], self.completed+1)

    def step(self):
        self.completed += 1
        self.prepare_next()

    def state_dict(self):
        return {'completed': self.completed, 'horizon': 24000}

    def load_state_dict(self, state):
        if state['horizon'] != 24000:
            raise ValueError('Scheduler horizon drift')
        self.completed = state['completed']
        self.prepare_next()
