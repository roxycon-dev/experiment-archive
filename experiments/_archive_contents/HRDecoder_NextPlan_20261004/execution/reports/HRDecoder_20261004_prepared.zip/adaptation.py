"""Only trainability and registered per-layer learning rates differ across F arms."""
import math
import re
import torch


def configure(model, arm):
    if arm == 'E_NEG':
        return
    if arm not in ('F4', 'F12'):
        raise ValueError('Unregistered adaptation arm')
    model.backbone.requires_grad_(arm == 'F12')
    if arm == 'F4':
        model.backbone.blocks[8:].requires_grad_(True)
        model.backbone.norm.requires_grad_(True)
    # Unmasked intermediate-layer inference uses mask_token only multiplied by zero.
    # cls_norm acts on discarded output tokens, while backbone.norm acts on patches.
    for name, p in model.backbone.named_parameters():
        if name == 'mask_token' or name.startswith(('cls_norm.', 'local_cls_norm.')):
            p.requires_grad_(False)
    # Historical checkpoint.load calls set_update; preserve the registered F policy.
    model.set_update = lambda completed: setattr(model, 'completed', completed)


def optimizer_for(model):
    groups = {}
    names = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not name.startswith('backbone.'):
            family, peak = 'head', 5e-5
        elif (match := re.match(r'backbone\.blocks\.(\d+)\.', name)):
            index = int(match[1])
            family, peak = f'block{index}', 1e-5 * .8**(11-index)
        elif name.startswith('backbone.norm.'):
            family, peak = 'norm', 1e-5
        else:
            family, peak = 'embedding', 1e-5 * .8**12
        decay = p.ndim > 1 and not name.endswith('.bias')
        key = family, decay
        groups.setdefault(key, {'params': [], 'family': family, 'peak_lr': peak,
                               'lr': 0., 'weight_decay': .01 if decay else 0.})['params'].append(p)
        names.setdefault(f'{family}:{decay}', []).append(name)
    optimizer = torch.optim.AdamW(list(groups.values()))
    return optimizer, names


def factor(update):
    if not 1 <= update <= 12000:
        raise ValueError('Outside registered fine-tuning horizon')
    if update <= 200:
        return update / 200
    return .01 + .99*.5*(1 + math.cos(math.pi*(update-200)/11800))


class Schedule:
    def __init__(self, optimizer):
        self.optimizer, self.completed = optimizer, 0
        self.prepare_next()

    def prepare_next(self):
        if self.completed < 12000:
            for group in self.optimizer.param_groups:
                group['lr'] = group['peak_lr'] * factor(self.completed+1)

    def step(self):
        self.completed += 1
        self.prepare_next()

    def state_dict(self):
        return {'completed': self.completed, 'horizon': 12000}

    def load_state_dict(self, state):
        if state['horizon'] != 12000:
            raise ValueError('Fine-tuning horizon drift')
        self.completed = state['completed']
        self.prepare_next()
