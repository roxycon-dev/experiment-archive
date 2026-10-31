"""Bounded RAM reuse of exact FP32 forwards of the frozen global branch.

The key hashes the actual normalized, padded augmented full-image input. No
feature is rotated/flipped to approximate a network forward. CPU storage is
lossless and shared by paired arms/nodes in this process, never a model weight.
"""
from collections import OrderedDict
import hashlib
import torch

from .geometry import global_view

_CACHE = OrderedDict()
_BYTES = 0
_HITS = 0
_MISSES = 0


@torch.no_grad()
def exact_global(branch, rgb, long_side, identity, limit_gib=48):
    global _BYTES, _HITS, _MISSES
    view, ledger = global_view(rgb, long_side, 'cpu')
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    key = (identity, tuple(view.shape), str(flags),
           hashlib.sha256(view.numpy().tobytes()).hexdigest())
    device = next(branch.parameters()).device
    if key in _CACHE:
        _HITS += 1
        _CACHE.move_to_end(key)
        return _CACHE[key].to(device), ledger
    _MISSES += 1
    feature = branch(view.to(device)).float()
    stored = feature.detach().cpu()
    size = stored.numel()*stored.element_size()
    limit = int(limit_gib*1024**3)
    while _CACHE and _BYTES+size > limit:
        _, old = _CACHE.popitem(last=False)
        _BYTES -= old.numel()*old.element_size()
    if size <= limit:
        _CACHE[key] = stored
        _BYTES += size
    return feature, ledger


def cache_stats():
    return dict(hits=_HITS, misses=_MISSES, entries=len(_CACHE), bytes=_BYTES,
                storage='bounded_RAM_exact_float32',
                input_identity='SHA256 of actual augmented normalized full input')
