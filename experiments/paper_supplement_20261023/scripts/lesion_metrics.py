"""Preregistered one-to-one connected-component lesion metrics."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def components(mask):
    if mask.ndim != 2:
        raise ValueError('Expected a two-dimensional binary mask')
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)


def match_lesions(gt, prediction, valid=None, minimum_iou=0.10):
    if gt.shape != prediction.shape or gt.ndim != 2:
        raise ValueError('Masks must align in original coordinates')
    valid = np.ones_like(gt, bool) if valid is None else valid.astype(bool)
    if valid.shape != gt.shape:
        raise ValueError('Valid mask shape mismatch')
    ng, gl, gs, _ = components(gt.astype(bool) & valid)
    npred, pl, ps, _ = components(prediction.astype(bool) & valid)
    ga = gs[1:, cv2.CC_STAT_AREA].astype(np.int64)
    pa = ps[1:, cv2.CC_STAT_AREA].astype(np.int64)
    matched = []
    if len(ga) and len(pa):
        overlap = (gl > 0) & (pl > 0)
        pairs, intersections = np.unique(gl[overlap].astype(np.int64) * npred + pl[overlap], return_counts=True)
        gidx, pidx = pairs // npred - 1, pairs % npred - 1
        ious = intersections / (ga[gidx] + pa[pidx] - intersections)
        eligible = ious >= minimum_iou
        # Work only on components participating in eligible overlaps. This avoids
        # allocating a square matrix for thousands of isolated false positives.
        gkeys, pkeys = np.unique(gidx[eligible]), np.unique(pidx[eligible])
        if len(gkeys) and len(pkeys):
            score = np.zeros((len(gkeys), len(pkeys)), np.float64)
            cardinality_bonus = min(len(ga), len(pa)) + 1
            for g, p, iou in zip(gidx[eligible], pidx[eligible], ious[eligible]):
                score[np.searchsorted(gkeys, g), np.searchsorted(pkeys, p)] = cardinality_bonus + iou
            gi, pi = linear_sum_assignment(score, maximize=True)
            for g, p in zip(gi, pi):
                if score[g, p] > 0:
                    matched.append({'gt_id': int(gkeys[g] + 1), 'pred_id': int(pkeys[p] + 1),
                                    'iou': float(score[g, p] - cardinality_bonus)})
    mg = {r['gt_id'] for r in matched}
    mp = {r['pred_id'] for r in matched}
    return {'gt_count': ng - 1, 'pred_count': npred - 1, 'tp': len(matched),
            'fp': npred - 1 - len(matched), 'fn': ng - 1 - len(matched),
            'gt_areas': ga.tolist(), 'pred_areas': pa.tolist(), 'matches': matched,
            'missed_gt_ids': [i for i in range(1, ng) if i not in mg],
            'false_pred_ids': [i for i in range(1, npred) if i not in mp]}


def grouped_counts(result, bins):
    hit = {r['gt_id'] for r in result['matches']}
    rows = {name: {'gt': 0, 'detected': 0} for name in ['small', 'medium', 'large']}
    for i, area in enumerate(result['gt_areas'], 1):
        name = 'small' if area <= bins[0] else ('medium' if area <= bins[1] else 'large')
        rows[name]['gt'] += 1
        rows[name]['detected'] += int(i in hit)
    return rows
