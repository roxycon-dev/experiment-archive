import argparse
from pathlib import Path
import json

from qlib.common import digest, read_json, sha, write_json
from qlib.data import DDR, External, catalog, make_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', default='configs/runtime.local.json')
    ap.add_argument('--arm', choices=['Q_DDR', 'Q_MIX'], default='Q_DDR')
    ap.add_argument('--seed', type=int, choices=[42, 2024], default=42)
    args = ap.parse_args()
    runtime = read_json(args.runtime)
    output = Path('reports')/f'stream_{args.arm}_seed{args.seed}.jsonl'
    if output.exists():
        info = read_json(output.with_suffix('.summary.json'))
        if sha(output) != info['file_sha256']:
            raise ValueError('Existing stream identity mismatch')
        print('Existing verified stream retained; no resampling')
        return
    identity = read_json('reports/data_identity.json')
    datasets = {'DDR': DDR(runtime['ddr_root'], 'train', identity['splits']['train'])}
    if args.arm == 'Q_MIX':
        external = read_json(runtime['external_manifest'])
        for source in ('IDRiD', 'MAPLES'):
            datasets[source] = External(source, external)
    catalogs = {}
    for source, dataset in datasets.items():
        catalogs[source] = catalog(dataset)
        print(f'Training-only component catalog ready: {source}', flush=True)
    rows = make_stream(datasets, catalogs, args.seed, args.arm)
    with output.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n')
    exposures = {}
    for endpoint in (16000, 24000):
        exposures[str(endpoint)] = {source: sum(r['source'] == source for r in rows[:endpoint*8]) for source in datasets}
    write_json(output.with_suffix('.summary.json'), {'status': 'PASS', 'arm': args.arm, 'seed': args.seed,
        'updates': 24000, 'windows': len(rows), 'rows_sha256': digest(rows), 'file_sha256': sha(output),
        'data_identity_sha256': identity['sha256'], 'exposures': exposures,
        'foreground_positive_images': {s: {c: len(p) for c, p in zip(('EX', 'HE', 'SE', 'MA'), catalog_data[1])} for s, catalog_data in catalogs.items()}})
    print('Frozen stream ready:', output, exposures, flush=True)


if __name__ == '__main__':
    main()
