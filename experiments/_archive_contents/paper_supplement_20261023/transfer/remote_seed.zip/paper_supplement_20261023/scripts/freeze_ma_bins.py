"""Freeze MA lesion size bins using only verified training masks."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('D:/HRDecoder/data/DDR_split_613'))
    args = parser.parse_args()
    destination = ROOT / 'evidence/ma_size_bins.json'
    if destination.exists():
        raise FileExistsError('Frozen bins already exist; do not silently replace')
    cv2.setNumThreads(2)
    manifest = json.loads((ROOT / 'evidence/local_asset_audit.json').read_text(encoding='utf-8'))
    rows, all_areas = [], []
    for i, row in enumerate(manifest['data']['train']['samples']):
        p = args.data_root / 'labels/train' / (row['sid'] + '.png')
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['label_sha256']
        label = np.asarray(Image.open(p))
        n, _, stats, _ = cv2.connectedComponentsWithStats((label == 4).astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()
        rows.append({'sid': row['sid'], 'lesions': n - 1, 'areas': areas})
        all_areas.extend(areas)
        if (i + 1) % 100 == 0:
            print(json.dumps({'training_images': i + 1, 'MA_lesions': len(all_areas)}), flush=True)
    assert all_areas
    bins = np.quantile(all_areas, [1/3, 2/3], method='nearest').astype(int).tolist()
    result = {'status': 'FROZEN_FROM_TRAINING_GT_ONLY', 'connectivity': 8, 'class_id': 4,
              'min_component_area': 1, 'quantile_method': 'nearest', 'quantiles': [1/3, 2/3],
              'boundaries_pixels': bins, 'groups': ['area <= q1', 'q1 < area <= q2', 'area > q2'],
              'training_images': len(rows), 'training_MA_lesions': len(all_areas),
              'protocol_sha256': hashlib.sha256((ROOT / 'plan/protocol.md').read_bytes()).hexdigest(),
              'training_samples': rows, 'test_predictions_accessed': False}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'training_samples'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
