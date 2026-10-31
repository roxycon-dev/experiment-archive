"""Recompute saved validation counts for reporting; never evaluate or select models."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parent / 'execution'
RESULTS = ROOT / 'server_results'
CLASSES = ('EX', 'HE', 'SE', 'MA')
FORMULAS = ('single', 'fixed_fusion', 'half_fusion', 'p0')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_metrics(path, label, arm=None):
    x = json.loads(path.read_text(encoding='utf-8'))
    assert x['split'] == 'val' and x['n_images'] == len(x['per_image']) == 76
    assert x['threshold'] == .5 and x['strict_greater'] is True
    assert x['coordinates'] == 'original_image'
    sids = [p['sid'] for p in x['per_image']]
    assert len(set(sids)) == 76
    outputs = {}
    for formula in FORMULAS:
        counts = [[sum(p[formula][i][j] for p in x['per_image'])
                   for j in range(3)] for i in range(4)]
        assert counts == x[formula]['counts'], (path, formula, 'count disagreement')
        categories = {}
        for name, (tp, fp, fn) in zip(CLASSES, counts):
            assert min(tp, fp, fn) >= 0 and tp + fp + fn > 0
            categories[name] = {
                'tp': tp, 'fp': fp, 'fn': fn,
                'iou_pct': 100 * tp / (tp + fp + fn),
                'dice_pct': 200 * tp / (2 * tp + fp + fn),
                'precision': tp / (tp + fp) if tp + fp else None,
                'recall': tp / (tp + fn) if tp + fn else None,
            }
        miou = sum(v['iou_pct'] for v in categories.values()) / 4
        assert abs(miou - x[formula]['miou_pct']) < 1e-10, (path, formula)
        outputs[formula] = {'miou_pct': miou, 'per_class': categories}
    return {
        'label': label, 'arm': arm, 'seed': 42, 'update': x['update'],
        'split': 'val', 'n_images': 76,
        'selection_node': bool(arm and x['update'] in (20000, 32000)),
        'source_path': str(path.resolve()), 'source_sha256': digest(path),
        'sid_set_sha256': hashlib.sha256('\n'.join(sorted(sids)).encode()).hexdigest(),
        'threshold': .5, 'strict_greater': True, 'coordinates': 'original_image',
        'scores_recomputed_from_saved_per_image_counts': True,
        'outputs': outputs,
    }


def main():
    rows = []
    for arm in ('C', 'S', 'D', 'SD'):
        for p in sorted((RESULTS / f'runs/{arm}_seed42').glob('val_u*/metrics.json')):
            rows.append(read_metrics(p, f'{arm}_seed42', arm))
    for label, rel in (
        ('F12_parent42', 'reports/parent_restore_val42/metrics.json'),
        ('Q_DDR16k_R42', 'reports/R_reconstruction_seed42/metrics.json'),
    ):
        rows.append(read_metrics(RESULTS / rel, label))
    assert len({x['sid_set_sha256'] for x in rows}) == 1, 'validation image sets differ'
    p0 = [x['outputs']['p0'] for x in rows]
    assert all(v == p0[0] for v in p0), 'frozen P0 reference counts differ'
    paired = []
    for u in (20000, 32000):
        by_arm = {x['arm']: x for x in rows if x['arm'] and x['update'] == u}
        for formula in FORMULAS[:3]:
            score = {a: v['outputs'][formula]['miou_pct'] for a, v in by_arm.items()}
            contrasts = {f'{a}_minus_C_pp': score[a] - score['C']
                         for a in ('S', 'D', 'SD') if a in score and 'C' in score}
            if set(score) == {'C', 'S', 'D', 'SD'}:
                contrasts['SD_minus_S_minus_D_plus_C_pp'] = score['SD'] - score['S'] - score['D'] + score['C']
            paired.append({'update': u, 'formula': formula, 'scores_pct': score,
                           'missing_arms': sorted({'C', 'S', 'D', 'SD'} - set(score)),
                           'contrasts': contrasts})
    result = {
        'stage': 'COMPLETE_SAVED_VALIDATION_EVIDENCE' if len(rows) == 34 else 'PROVISIONAL_SAVED_VALIDATION_EVIDENCE',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Read-only recomputation of actual saved validation metrics; no model selection or test inference',
        'not_final_outcome': False if len(rows) == 34 else True,
        'reference_formulas': {'fixed_fusion': '.75*P0+.25*current', 'half_fusion': '.5*P0+.5*current'},
        'matched_contrast_interpretation': 'D/SD jointly add external supervision, domain decoder and compute; these contrasts do not isolate a pure data effect. Scores describe the saved matched arms and are not a statistical significance claim.',
        'records': rows, 'paired_registered_endpoint_contrasts': paired,
    }
    out = REPORT / 'evidence/four_arm_saved_validation.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'records': len(rows),
                      'all_saved_scores_and_counts_exactly_recomputed': True,
                      'paired_contrasts': paired}, ensure_ascii=True))


if __name__ == '__main__':
    main()
