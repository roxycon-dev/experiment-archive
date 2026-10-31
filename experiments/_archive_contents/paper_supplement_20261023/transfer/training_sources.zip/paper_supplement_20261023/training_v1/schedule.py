"""Exact registered parameter families, optimizer, and 32k schedule."""
import math
import torch
from torch import nn


def optimizer_for(model, fused=True):
    families = {}
    for i, block in enumerate(model.backbone.blocks):
        for p in block.parameters():
            families[id(p)] = (f'block{i}', 1e-5 * .8**(11 - i))
    for p in model.backbone.norm.parameters():
        families[id(p)] = ('final_norm', 1e-5)
    for p in model.backbone.parameters():
        families.setdefault(id(p), ('embedding_cls_register', 1e-5 * .8**12))
    for domain in ('ddr', 'tjdr'):
        decoder = getattr(model, domain, None)
        if decoder is None:
            continue
        for p in decoder.parameters():
            families[id(p)] = (domain + '_decoder', 5e-5)
        if hasattr(decoder, 'spatial_residual'):
            for p in decoder.spatial_residual.parameters():
                families[id(p)] = (domain + '_spatial_residual', 1e-4)
    if model.has_spatial:
        for p in model.spatial.parameters():
            families[id(p)] = ('spatial_interaction', 1e-4)
    no_decay = set()
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            no_decay.update(id(p) for p in module.parameters(recurse=False))
        if getattr(module, 'bias', None) is not None:
            no_decay.add(id(module.bias))
    groups, names = {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        assert id(parameter) in families, 'Unregistered parameter: ' + name
        family, peak = families[id(parameter)]
        decay = id(parameter) not in no_decay
        key = family, decay
        groups.setdefault(key, {'params': [], 'family': family, 'peak_lr': peak,
                               'lr': peak / 500, 'weight_decay': .01 if decay else 0.})['params'].append(parameter)
        names.setdefault(f'{family}:decay={decay}', []).append(name)
    optimizer = torch.optim.AdamW(list(groups.values()), betas=(.9, .999), eps=1e-8,
                                  fused=fused)
    return optimizer, names


def factor(update):
    assert 1 <= update <= 32000
    if update <= 500:
        return update / 500
    return .01 + .99 * .5 * (1 + math.cos(math.pi * (update - 500) / 31500))


class Schedule:
    def __init__(self, optimizer):
        self.optimizer, self.completed = optimizer, 0
        self.prepare_next()

    def prepare_next(self):
        if self.completed < 32000:
            for group in self.optimizer.param_groups:
                group['lr'] = group['peak_lr'] * factor(self.completed + 1)

    def step(self):
        self.completed += 1
        self.prepare_next()

    def state_dict(self):
        return {'completed': self.completed, 'horizon': 32000}

    def load_state_dict(self, state):
        assert state['horizon'] == 32000
        self.completed = state['completed']
        self.prepare_next()
