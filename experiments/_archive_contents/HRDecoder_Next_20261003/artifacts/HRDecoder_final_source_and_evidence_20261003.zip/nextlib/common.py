"""Immutable identities, atomic artifacts and strict trusted-project loads."""
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('EX', 'HE', 'SE', 'MA')
LOCAL_MEAN = (123.675, 116.28, 103.53)
LOCAL_STD = (58.395, 57.12, 57.375)
GLOBAL_MEAN = (81.205, 50.636, 21.216)
GLOBAL_STD = (76.252, 48.798, 21.625)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def trusted_load(path):
    # These are user-owned historical checkpoints, identified by SHA first.
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def state_from(ck):
    for key in ('model_state', 'state_dict', 'model'):
        if isinstance(ck, dict) and isinstance(ck.get(key), dict):
            ck = ck[key]
            break
    if not isinstance(ck, dict) or not ck or not all(torch.is_tensor(v) for v in ck.values()):
        raise ValueError('Expected a complete tensor state dictionary')
    return {k[7:] if k.startswith('module.') else k: v for k, v in ck.items()}


def checked_asset(settings, name):
    spec = settings['assets'][name]
    path = Path(spec['path'])
    actual = sha(path)
    if actual != spec['sha256']:
        raise ValueError(f'Asset SHA mismatch: {name}: {path}')
    return path


def strict_load(module, path):
    state = state_from(trusted_load(path))
    module.load_state_dict(state, strict=True)
    # Record every tensor actually installed, not just the loader return value.
    target = module.state_dict()
    if not all(torch.equal(target[k].cpu(), v.cpu()) for k, v in state.items()):
        raise ValueError('Post-load tensor equality failed')
    return {'checkpoint_sha256': sha(path), 'loaded_keys': sorted(state),
            'tensor_shapes': {k: list(v.shape) for k, v in state.items()},
            'missing': [], 'unexpected': [], 'post_load_equal': True}


def source_identity():
    files = {}
    for sub in ('nextlib', 'trainer', 'legacy', 'tools', 'vendor/champion_repo', 'vendor/HRDecoder'):
        for path in sorted((ROOT / sub).rglob('*.py')):
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in sorted(ROOT.glob('*.py')):
        files[path.name] = sha(path)
    return {'files': files, 'sha256': digest(files)}
