"""Recompute fixed S20/C20 system replacement from archived per-image counts."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent


def summarize(a):
    totals = np.asarray(a, dtype=np.int64).sum(axis=0)
    tp, fp, fn = totals.T
    iou = tp / np.maximum(tp + fp + fn, 1)
    return {'miou_pct': float(iou.mean() * 100), 'per_class_iou_pct': (iou * 100).tolist(),
            'MA_precision': float(tp[3] / max(tp[3] + fp[3], 1)),
            'MA_recall': float(tp[3] / max(tp[3] + fn[3], 1)), 'counts': totals.tolist()}


def main():
    records, arrays, rows = {}, {}, []
    for arm in ['C', 'S']:
        p = EXP / f'HRDecoder_FinalDecision_20261005/execution/server_results/runs/{arm}_seed42/val_u20000/metrics.json'
        j = json.loads(p.read_text(encoding='utf-8'))
        assert j['split'] == 'val' and j['update'] == 20000 and j['n_images'] == 76
        assert j['strict_greater'] and j['probability_dtype'] == 'float32' and j['coordinates'] == 'original_image'
        ids = [r['sid'] for r in j['per_image']]
        a = np.asarray([r['fixed_fusion'] for r in j['per_image']], dtype=np.int64)
        result = summarize(a)
        assert abs(result['miou_pct'] - j['fixed_fusion']['miou_pct']) < 1e-12
        records[arm] = {'source': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'ids': ids, 'result': result}
        arrays[arm] = a
        rows.append({'model': 'Full S20 EMA system' if arm == 'S' else 'Replace S20 with matched C20 EMA',
                     'split': 'validation', 'n_images': 76, **{k:v for k,v in result.items() if k not in ['per_class_iou_pct', 'counts']},
                     **dict(zip(['EX_IoU_pct','HE_IoU_pct','SE_IoU_pct','MA_IoU_pct'], result['per_class_iou_pct']))})
    assert records['S']['ids'] == records['C']['ids']
    for formula in ['p0']:
        sources = [json.loads(Path(records[x]['source']).read_text(encoding='utf-8')) for x in ['C','S']]
        assert [r[formula] for r in sources[0]['per_image']] == [r[formula] for r in sources[1]['per_image']]
    rng = np.random.default_rng(20261023)
    idx = rng.integers(0, 76, (2000, 76))
    def score(a):
        t = a[idx].sum(axis=1)
        return (t[:, :, 0] / np.maximum(t.sum(axis=2), 1)).mean(axis=1) * 100
    delta = score(arrays['S']) - score(arrays['C'])
    result = {'status': 'RECOMPUTED_FROM_ARCHIVED_COUNTS_NO_NEW_INFERENCE', 'sources': records,
              'S_minus_C_miou_pp': records['S']['result']['miou_pct'] - records['C']['result']['miou_pct'],
              'paired_image_bootstrap95_pp': np.quantile(delta, [.025,.975]).tolist(),
              'bootstrap_draws': 2000, 'bootstrap_seed': 20261023,
              'limitations': ['validation used for historical selection', 'single seed', 'conditional image bootstrap, no patient identity',
                             'not the pending four-row DDR test experiment', 'MA lesion metrics cannot be recovered from pixel counts']}
    out = ROOT / 'results/matched_system_validation.json'
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    with (ROOT / 'results/matched_system_validation.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},ensure_ascii=False))


if __name__ == '__main__':
    main()
