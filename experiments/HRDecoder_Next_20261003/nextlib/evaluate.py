"""Original-coordinate global counts, fixed threshold, lesion and calibration diagnostics."""
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import shutil
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from trainer.eval import pad_normalized_image, sliding_window_logits
from tools.probability_cache import save_probability, load_probability
from .common import CLASSES, LOCAL_MEAN, LOCAL_STD, GLOBAL_MEAN, GLOBAL_STD, write_json, digest, read_json
from .geometry import normalized, canvas

# Four independent image tasks use the allocated CPU cores; avoid each CCL
# task launching 128 host threads while the GPU waits for serial compression.
cv2.setNumThreads(1)


def counts(prob, label, threshold=.5):
    out = []
    for c in range(4):
        gt, pred = label == c+1, prob[c] > threshold
        tp = int(np.count_nonzero(gt & pred))
        out.append([tp, int(np.count_nonzero(pred & ~gt)), int(np.count_nonzero(gt & ~pred))])
    return np.asarray(out, np.int64)


def metrics(total):
    total = np.asarray(total, np.int64)
    tp, fp, fn = total.T
    iou = tp / np.maximum(tp+fp+fn, 1)
    return dict(mIoU_lesion4=float(iou.mean()), counts=total.tolist(),
                per_class={name: dict(tp=int(tp[c]), fp=int(fp[c]), fn=int(fn[c]),
                    iou=float(iou[c]), precision=float(tp[c]/max(tp[c]+fp[c], 1)),
                    recall=float(tp[c]/max(tp[c]+fn[c], 1))) for c, name in enumerate(CLASSES)})


class Diagnostics:
    def __init__(self, bins=512):
        self.bins = bins
        self.pos = np.zeros((4, bins), np.int64)
        self.neg = np.zeros_like(self.pos)
        self.lesions = np.zeros((4, 5, 5), np.int64)
        self.images = []

    def add(self, sid, prob, label, baseline=None):
        if prob.shape != (4, *label.shape) or prob.dtype != np.float32:
            raise ValueError('Probabilities must be fp32 and in original GT coordinates')
        if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
            raise ValueError('Nonfinite/out-of-range probabilities')
        ct = counts(prob, label)
        row = dict(sid=sid, counts=ct.tolist(), hw=list(label.shape))
        row['probability_histogram'] = {'pos': [], 'neg': []}
        if baseline is not None:
            bc = counts(baseline, label)
            row['baseline_counts'] = bc.tolist()
            row['delta_counts'] = (ct-bc).tolist()
            row['shared_errors'] = []
        h, w = label.shape
        for c in range(4):
            gt = label == c+1
            pred = prob[c] > .5
            ix = np.minimum((prob[c]*self.bins).astype(np.int32), self.bins-1)
            hp = np.bincount(ix[gt], minlength=self.bins)
            hn = np.bincount(ix[~gt], minlength=self.bins)
            self.pos[c] += hp
            self.neg[c] += hn
            row['probability_histogram']['pos'].append(hp.tolist())
            row['probability_histogram']['neg'].append(hn.tolist())
            n, cc, stat, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
            # Columns: lesions, completely missed lesions, GT pixels, FN pixels, boundary FN pixels.
            for k in range(1, n):
                x, y, bw, bh, area = stat[k]
                bucket = min(int(np.searchsorted([16, 64, 256, 1024], area, side='left')), 4)
                lesion = cc[y:y+bh, x:x+bw] == k
                detected = pred[y:y+bh, x:x+bw] & lesion
                eroded = cv2.erode(lesion.astype(np.uint8), np.ones((3, 3), np.uint8),
                                   borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                boundary = lesion & ~eroded
                missed = lesion & ~pred[y:y+bh, x:x+bw]
                self.lesions[c, bucket] += [1, int(not detected.any()), int(area),
                                            int(missed.sum()), int((missed & boundary).sum())]
            if baseline is not None:
                old = baseline[c] > .5
                row['shared_errors'].append(dict(shared_fn=int((gt & ~pred & ~old).sum()),
                    shared_fp=int((~gt & pred & old).sum()), rescued_fn=int((gt & pred & ~old).sum()),
                    new_fp=int((~gt & pred & ~old).sum()), removed_fp=int((~gt & ~pred & old).sum())))
        self.images.append(row)
        return ct

    def report(self):
        total = np.asarray([r['counts'] for r in self.images], np.int64).sum(axis=0)
        curves = {}
        for c, name in enumerate(CLASSES):
            tp = np.cumsum(self.pos[c, ::-1])[::-1]
            fp = np.cumsum(self.neg[c, ::-1])[::-1]
            fn = self.pos[c].sum()-tp
            iou = tp/np.maximum(tp+fp+fn, 1)
            curves[name] = dict(threshold_lower_bound=(np.arange(self.bins)/self.bins).tolist(),
                precision=(tp/np.maximum(tp+fp, 1)).tolist(), recall=(tp/max(self.pos[c].sum(), 1)).tolist(),
                iou=iou.tolist(), oracle_histogram_best=float(iou.max()),
                oracle_note='512-bin diagnostic only; not a deployable or fixed-0.5 score')
        return dict(**metrics(total), n_images=len(self.images), threshold=.5, strict_greater=True,
            coordinates='original_image', channel_order=list(CLASSES), per_image=self.images,
            lesion_area_upper_bounds=[16,64,256,1024,None], lesion_columns=['lesions','completely_missed',
                'gt_pixels','fn_pixels','boundary_fn_pixels'], lesion_buckets=self.lesions.tolist(), curves=curves)


@torch.no_grad()
def legacy_logits(model, rgb, geometry, mode, family='U', device='cuda', ordinary=False):
    view, _ = canvas(rgb, None, geometry)
    mean, std = (LOCAL_MEAN, LOCAL_STD) if family == 'U' else (GLOBAL_MEAN, GLOBAL_STD)
    padded, (h, w) = pad_normalized_image(normalized(view, mean, std))
    x = torch.from_numpy(padded[None]).to(device)
    if mode == 'whole':
        output = model(x) if ordinary else model.eval_forward(x)
    elif mode == 'slide':
        output = sliding_window_logits(model, x, 1024, 768)
    else:
        raise ValueError(mode)
    output = output[..., :h, :w].float()
    return F.interpolate(output, rgb.shape[:2], mode='bilinear', align_corners=False).cpu()


def evaluate(dataset, predict, out, identity, baseline=None, save_probs=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out/'index.json'
    if index_path.exists():
        index = read_json(index_path)
        if index['identity'] != identity:
            raise ValueError(f'Completed cache identity conflict: {out}')
        # Reuse only a complete original-coordinate exact-identity cache.
        for sid in dataset.stems:
            if save_probs and not (out/(sid+'.npz')).is_file():
                raise ValueError('Completed cache is missing a probability file')
        return read_json(out/'metrics.json')
    if save_probs:
        # Reserve one image's uncompressed fp32 output and room for checkpoint writes.
        if hasattr(dataset,'images'):
            from PIL import Image
            sizes=[]
            for path in dataset.images.values():
                with Image.open(path) as im:sizes.append(im.width*im.height)
            largest=max(sizes)
        else:
            largest=max(dataset.load(sid)[1].size for sid in dataset.stems)
        reserve=largest*4*4+1024**3
        if shutil.disk_usage(out).free < reserve:
            raise RuntimeError('Insufficient cache disk space; mirror this NEW run before continuing')
    diagnostics = Diagnostics()
    partial_path=out/'partial.json'
    partial={'identity':identity,'images':{}}
    if partial_path.exists():
        partial=read_json(partial_path)
        if partial['identity'] != identity:
            raise ValueError('Incomplete evaluation belongs to a different identity')
    pending = deque()
    def finish_one():
        row, pos, neg, lesions = pending.popleft().result()
        diagnostics.images.append(row)
        diagnostics.pos += pos
        diagnostics.neg += neg
        diagnostics.lesions += lesions
        partial['images'][row['sid']] = row
        write_json(partial_path,partial)
        print(f'eval {out.name}: {len(diagnostics.images)}/{len(dataset)} {row["sid"]}', flush=True)

    def process_image(sid, p, label):
        one = Diagnostics()
        b = None if baseline is None else load_probability(baseline, sid)
        one.add(sid, p, label, b)
        one.images[0]['lesion_buckets'] = one.lesions.tolist()
        if save_probs:
            save_probability(out/sid, p)
        return one.images[0], one.pos, one.neg, one.lesions

    # Predictions remain serial in exactly the original order. Independent CPU
    # diagnostics/compression overlap the next forward, then commit in order.
    # Counts are persisted only AFTER the matching lossless file is installed.
    with ThreadPoolExecutor(max_workers=4) as pool:
        try:
            for i, sid in enumerate(dataset.stems):
                if sid in partial['images']:
                    if save_probs and not (out/(sid+'.npz')).exists():
                        raise ValueError('Partial counts have no saved probability file')
                    old=partial['images'][sid]
                    diagnostics.images.append(old)
                    diagnostics.pos+=np.asarray(old['probability_histogram']['pos'],np.int64)
                    diagnostics.neg+=np.asarray(old['probability_histogram']['neg'],np.int64)
                    diagnostics.lesions+=np.asarray(old['lesion_buckets'],np.int64)
                    continue
                rgb, label = dataset.load(sid)
                if save_probs and shutil.disk_usage(out).free < label.size*16*(len(pending)+1)+1024**3:
                    while pending:
                        finish_one()
                if save_probs and shutil.disk_usage(out).free < label.size*16+1024**3:
                    raise RuntimeError('Cache disk reserve reached; completed image predictions remain resumable')
                p = predict(rgb)
                if torch.is_tensor(p):
                    p = p.sigmoid().numpy()[0].astype(np.float32)
                pending.append(pool.submit(process_image,sid,p,label))
                if len(pending) >= 4:
                    finish_one()
        finally:
            # Preserve every successfully saved earlier image even if a later
            # GPU forward raises/OOMs; a restart repeats only unfinished work.
            while pending:
                finish_one()
    report = diagnostics.report()
    report['identity'] = identity
    write_json(out/'metrics.json', report)
    write_json(index_path, dict(identity=identity, n_images=len(dataset), split='val',
                                samples=dataset.stems, storage='npz_deflate1_lossless',
                                dtype='float32', coordinates='original_image', channel_order=list(CLASSES)))
    return report


def fusion_cache(dataset, left, right, out, weight=.5):
    identity = dict(operation='fixed_probability_fusion', left=digest(read_json(Path(left)/'index.json')),
                    right=digest(read_json(Path(right)/'index.json')), weight=weight)
    diagnostics = Diagnostics()
    def process(sid,label):
        a=load_probability(left,sid)
        p=(weight*a+(1-weight)*load_probability(right,sid)).astype(np.float32)
        one=Diagnostics();one.add(sid,p,label,a)
        return one
    pending=deque()
    def merge():
        one=pending.popleft().result()
        diagnostics.images.extend(one.images)
        diagnostics.pos+=one.pos;diagnostics.neg+=one.neg;diagnostics.lesions+=one.lesions
    with ThreadPoolExecutor(max_workers=4) as pool:
        for sid in dataset.stems:
            pending.append(pool.submit(process,sid,dataset.load(sid)[1]))
            if len(pending)>=4:merge()
        while pending:merge()
    report = diagnostics.report()
    report['identity'] = identity
    write_json(Path(out)/'metrics.json', report)
    # Lazy cache avoids keeping duplicate large full-val probabilities.
    write_json(Path(out)/'index.json', dict(identity=identity, storage='lazy_weighted_lossless_float32',
        components_cache=[str(Path(left).resolve()),str(Path(right).resolve())], weight=weight,
        split='val', n_images=len(dataset), channel_order=list(CLASSES), coordinates='original_image'))
    return report
