"""Exploration decisions in fraction units (0.003 = 0.3 percentage points)."""
def decide_stop(reports, total, policy):
    if not policy.get('enabled') or len(reports) < 2:
        return {'stop': False, 'reason': 'insufficient_evidence'}
    current, previous = reports[-1], reports[-2]
    if current['update'] < total*policy.get('min_progress', .5):
        return {'stop': False, 'reason': 'learning_grace_period'}
    if policy.get('mode') == 'external_dev_plateau':
        if len(reports) < 3:
            return {'stop': False, 'reason': 'external_requires_three_nodes'}
        recent_best = max(r['mIoU_lesion4'] for r in reports[-2:])
        older_best = max(r['mIoU_lesion4'] for r in reports[:-2])
        if recent_best < older_best + policy.get('minimum_gain', .001):
            return {'stop': True, 'reason': 'external_dev_plateau_select_best_then_transfer'}
        return {'stop': False, 'reason': 'external_dev_improving'}
    reference = policy['reference_miou']
    best = max(r['mIoU_lesion4'] for r in reports)
    gain = current['mIoU_lesion4']-previous['mIoU_lesion4']
    class_ref = policy.get('reference_class_iou') or {}
    complementary = any(max(r['per_class'][c]['iou'] for r in reports) >=
                        value+policy.get('class_margin', .005)
                        for c,value in class_ref.items())
    if best >= reference-policy.get('gap', .015) or complementary:
        return {'stop': False, 'reason': 'competitive_or_class_complementary'}
    if gain >= policy.get('minimum_gain', .003):
        return {'stop': False, 'reason': 'still_learning'}
    # Two adjacent full-val nodes, at least half a schedule. This is an
    # exploration prioritization rule, not a proof of architectural futility.
    return {'stop': True, 'reason': f'low_priority: best={best:.6f}, recent_gain={gain:.6f}, no_class_advantage'}
