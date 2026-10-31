"""Fixed, val-only fusion comparison during existing full-image evaluation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.probability_cache import load_probability


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_counts(prob, target, valid):
    if prob.shape != target.shape or prob.shape != valid.shape or prob.shape[0] != 4:
        raise ValueError('Fixed fusion geometry mismatch')
    pred = prob > .5
    gt = target > .5
    use = valid > .5
    return np.array([[np.count_nonzero(pred[c] & gt[c] & use[c]),
                      np.count_nonzero(pred[c] & ~gt[c] & use[c]),
                      np.count_nonzero(~pred[c] & gt[c] & use[c])]
                     for c in range(4)], dtype=np.int64)


def miou(counts):
    counts = np.asarray(counts, dtype=np.float64)
    return float(np.mean(counts[:, 0] / counts.sum(axis=1)))


class FixedFusionProbe:
    def __init__(self, config):
        self.base = Path(config['champion_cache'])
        self.u0 = Path(config['baseline_u0_cache'])
        lock = Path(config['baseline_lock'])
        if sha(lock) != config['baseline_lock_sha256']:
            raise ValueError('Completed baseline lock changed')
        locked = json.loads(lock.read_text())
        if any(d != {'family': 'convnext_unet', 'alpha': .25} for d in locked['deploy']):
            raise ValueError('Expected fixed .25 U0 baseline')
        indexes = [json.loads((p/'index.json').read_text()) for p in (self.base, self.u0)]
        ids = []
        for idx in indexes:
            if (idx['split'] != 'val' or idx['dtype'] != 'float32' or
                idx['coordinates'] != 'original_image' or
                idx['channel_order'] != ['EX', 'HE', 'SE', 'MA']):
                raise ValueError('Only original-coordinate val caches allowed')
            ids.append(sorted(s['sample_id'] for s in idx['samples']))
        if ids[0] != ids[1] or len(ids[0]) != 76:
            raise ValueError('Expected same 76-image validation split')
        if indexes[1]['checkpoint_sha256'] != locked['candidates']['convnext_unet']['members'][0]['checkpoint_sha256']:
            raise ValueError('Historical U0 cache identity changed')
        self.ids = ids[0]
        self.expected = locked['source_fusion']['locked_val_miou']
        self.identity = hashlib.sha256(''.join(sha(p/'index.json') for p in (self.base, self.u0)).encode()).hexdigest()
        self.path = Path(config['baseline_counts_cache'])
        self.baseline = {}
        if self.path.exists():
            stored = json.loads(self.path.read_text())
            if stored['identity'] != self.identity:
                raise ValueError('Baseline count cache identity changed')
            self.baseline = stored['counts']
        self.reset()

    def reset(self):
        self.candidate = {}

    def update(self, stem, prob, target, valid):
        if stem not in self.ids or stem in self.candidate:
            raise ValueError('Unknown or repeated val image')
        base = load_probability(self.base, stem)
        if base.shape != prob.shape:
            raise ValueError('Champion and candidate coordinates differ')
        self.candidate[stem] = pixel_counts(.75*base + .25*prob, target, valid).tolist()
        if stem not in self.baseline:
            original = load_probability(self.u0, stem)
            self.baseline[stem] = pixel_counts(.75*base + .25*original, target, valid).tolist()

    def compute(self):
        if sorted(self.candidate) != self.ids or sorted(self.baseline) != self.ids:
            raise ValueError('Fixed fusion requires complete 76-image validation')
        baseline = np.array(list(self.baseline.values()), dtype=np.int64).sum(axis=0)
        candidate = np.array(list(self.candidate.values()), dtype=np.int64).sum(axis=0)
        baseline_score = miou(baseline)
        if abs(baseline_score - self.expected) > .0005:
            raise ValueError('Completed baseline counts do not reproduce locked val')
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps({'identity': self.identity, 'counts': self.baseline}, indent=2))
            temp.replace(self.path)
        return {'fixed_fusion_miou': miou(candidate),
                'fixed_fusion_counts': candidate.tolist(),
                'fixed_fusion_baseline_miou': baseline_score,
                'fixed_fusion_gain': miou(candidate) - baseline_score,
                'fixed_fusion_alpha': .25, 'fixed_fusion_images': 76,
                'fixed_fusion_cache_identity': self.identity}
