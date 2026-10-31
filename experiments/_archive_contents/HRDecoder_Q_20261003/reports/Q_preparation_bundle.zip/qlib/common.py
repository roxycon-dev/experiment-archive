import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CLASSES = ('EX', 'HE', 'SE', 'MA')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_source(root):
    root = Path(root)
    source = read_json(root/'reports/source_identity.json')
    for rel, expected in source['files'].items():
        if sha(root/rel) != expected:
            raise ValueError('Frozen source changed: '+rel)
    return source
