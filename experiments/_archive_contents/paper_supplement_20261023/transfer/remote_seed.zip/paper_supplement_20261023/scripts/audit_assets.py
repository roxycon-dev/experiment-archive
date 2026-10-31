"""Read-only provenance inventory. Never load checkpoints or evaluate test masks."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('D:/HRDecoder/data/DDR_split_613'))
    args = parser.parse_args()
    records = list(csv.DictReader((EXP / 'paper_writing_20261023/evidence/workspace_file_metadata.csv').open(encoding='utf-8-sig')))
    weights = []
    caches = defaultdict(lambda: {'count': 0, 'bytes': 0})
    for row in records:
        rel = row['path']
        if row['extension'].lower() in ('.pth', '.pt', '.ckpt', '.safetensors'):
            p = EXP / rel
            weights.append({'path': str(p), 'exists_now': p.is_file(), 'bytes': p.stat().st_size if p.is_file() else None})
        if row['extension'] == '.npz' and '/tests/' not in rel:
            key = str(Path(rel).parent)
            caches[key]['count'] += 1
            caches[key]['bytes'] += int(row['bytes'])
    identity = json.loads((EXP / 'HRDecoder_FinalDecision_20261005/execution/reports/data_identity.json').read_text(encoding='utf-8'))
    splits = {}
    for split, info in identity['splits'].items():
        rows = []
        for entry in info['samples']:
            sid = entry['sid']
            image = args.data_root / 'images' / split / (sid + '.jpg')
            label = args.data_root / 'labels' / split / (sid + '.png')
            actual = {'image_sha256': sha(image), 'label_sha256': sha(label)}
            rows.append({'sid': sid, **actual, 'matches_historical': all(actual[k] == entry[k] for k in actual)})
        assert all(r['matches_historical'] for r in rows), split
        splits[split] = {'count': len(rows), 'all_hashes_match': True, 'samples': rows}
    result = {'status': 'LOCAL_ASSET_AUDIT_COMPLETE', 'checkpoints': weights,
              'npz_directories': dict(caches), 'data': splits,
              'inference_performed': False, 'training_performed': False,
              'protocol_sha256': sha(ROOT / 'plan/protocol.md'),
              'baseline_32000_found': False,
              'baseline_note': 'Existing D:/HRDecoder work_dirs checkpoint is seed14, original DDR split, validation225; not split613 seed42 baseline.'}
    out = ROOT / 'evidence/local_asset_audit.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'weights': len(weights), 'data': {s: x['count'] for s, x in splits.items()}, 'output': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
