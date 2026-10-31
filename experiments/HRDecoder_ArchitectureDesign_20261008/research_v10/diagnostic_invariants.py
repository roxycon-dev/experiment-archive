"""Frozen-provider warm diagnostics only; formal fresh40k training is unchanged."""
import copy
import torch
from model import SACDStudent


class FrozenProviderStudent(SACDStudent):
    """Keep provider parameters and mode frozen across train/eval transitions."""
    def __init__(self, baseline, **kwargs):
        super().__init__(baseline, **kwargs)
        self.base.requires_grad_(False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, image, label=None, valid=None):
        # A frozen provider has no training-crop supervision. Both diagnostic
        # arms use image-only provider inference; GT remains in the loss only.
        return super().forward(image, label=None, valid=None)

    def diagnostic_parameters(self):
        return tuple(self.private.parameters())


def named_private_optimizer_state(full_state, source_parameter_names, model):
    """Filter an explicitly named SGD parent; never guess unnamed slot ownership."""
    names=list(source_parameter_names)
    assert names==[name for name,_ in model.named_parameters()], 'Parent parameter order lacks matching evidence'
    groups=full_state['param_groups']
    assert len(groups)==1, 'This helper supports the audited single SGD group only'
    slots=groups[0]['params']
    assert len(slots)==len(names) and len(set(slots))==len(slots)
    chosen=[(slot,name) for slot,name in zip(slots,names) if name.startswith('private.')]
    assert [name for _,name in chosen]==['private.'+name for name,_ in model.private.named_parameters()]
    parameters=dict(model.named_parameters())
    for slot,name in chosen:
        assert slot in full_state['state'], 'Warm parent lacks private SGD state: '+name
        value=full_state['state'][slot]
        assert set(value)=={'momentum_buffer'}, 'Expected audited momentum SGD state: '+name
        momentum=value['momentum_buffer']
        assert momentum.shape==parameters[name].shape, 'SGD slot shape does not match parameter: '+name
        assert momentum.dtype==parameters[name].dtype, 'SGD slot dtype does not match parameter: '+name
    state={i:copy.deepcopy(full_state['state'][slot]) for i,(slot,_) in enumerate(chosen)}
    group=copy.deepcopy(groups[0]);group['params']=list(range(len(chosen)))
    if 'param_names' in group:group['param_names']=[name for _,name in chosen]
    return {'state':state,'param_groups':[group]}


def restore_immutable_parent(model, optimizer, model_state, optimizer_state):
    """Restore model/SGD only; the caller must separately restore parent RNG.

    Optimizer tensor aliases cannot mutate the retained parent in CPU replays.
    """
    model.load_state_dict(model_state,strict=True)
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))


def assert_frozen_provider(model, optimizer):
    assert not model.base.training
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert all(not module.training for module in model.base.modules())
    provider_ids={id(parameter) for parameter in model.base.parameters()}
    optimizer_ids={id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    assert not (provider_ids&optimizer_ids)
    assert optimizer_ids=={id(parameter) for parameter in model.private.parameters()}


def provider_snapshot(model):
    return {name:value.detach().cpu().clone() for name,value in model.base.state_dict().items()}


def assert_provider_unchanged(model, snapshot):
    state=model.base.state_dict()
    assert set(state)==set(snapshot)
    for name,value in state.items():
        assert torch.equal(value.detach().cpu(),snapshot[name]), 'Provider state changed: '+name
