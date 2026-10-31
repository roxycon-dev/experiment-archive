import numpy as np

from .common import CLASSES


def counts(probability, target, valid):
    if probability.dtype != np.float32 or probability.shape != target.shape or valid.shape != target.shape:
        raise ValueError('Metric requires aligned FP32 probabilities and four-class supervision')
    if not np.isfinite(probability).all():
        raise FloatingPointError('Nonfinite probabilities')
    p, g, v = probability > .5, target.astype(bool), valid.astype(bool)
    axes = tuple(range(1, p.ndim))
    return np.stack(((p & g & v).sum(axes), (p & ~g & v).sum(axes), (~p & g & v).sum(axes)), axis=1).astype(np.int64)


def summarize(ct):
    ct = np.asarray(ct, dtype=np.int64)
    tp, fp, fn = ct.T
    iou = tp / np.maximum(tp+fp+fn, 1)
    return {'counts': ct.tolist(), 'miou_pct': float(iou.mean()*100), 'per_class': {
        c: {'tp': int(tp[i]), 'fp': int(fp[i]), 'fn': int(fn[i]), 'iou_pct': float(iou[i]*100),
            'precision': float(tp[i]/max(tp[i]+fp[i], 1)), 'recall': float(tp[i]/max(tp[i]+fn[i], 1))}
        for i, c in enumerate(CLASSES)}}


def paired_bootstrap(old, new, seed=100303, draws=2000):
    old, new = np.asarray(old), np.asarray(new)
    if old.shape != new.shape or old.ndim != 3 or old.shape[1:] != (4, 3):
        raise ValueError('Pair by image identity before bootstrap')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(old), (draws, len(old)))
    def scores(a):
        a = a[indices].sum(axis=1)
        return (a[:, :, 0]/np.maximum(a.sum(axis=2), 1)).mean(axis=1)*100
    delta = scores(new)-scores(old)
    return {'paired_image_percentile95_pp': np.quantile(delta, [.025, .975]).tolist(),
            'resamples': draws, 'seed': seed, 'unit': 'image', 'includes_seed_variance': False}
