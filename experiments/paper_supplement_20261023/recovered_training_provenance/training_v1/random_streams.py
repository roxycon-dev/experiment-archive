"""Domain forward RNG pairing specified by the frozen interface."""
from contextlib import contextmanager
import hashlib
import torch


def micro_seed(seed, source, update, microindex):
    assert source in ('DDR', 'TJDR') and update >= 1
    value = f'campaign20261005|{seed}|{source}|{update}|{microindex}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little') % 2**63


@contextmanager
def forward_rng(seed, source, update, microindex, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        value = micro_seed(seed, source, update, microindex)
        torch.random.default_generator.manual_seed(value)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(value)
        yield
