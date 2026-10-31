"""CPU-only identity/annotation audit of the user-confirmed 27-image holdout."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'HRDecoder_Q_20261003'
DERIVED = OLD / 'assets/external_derived/IDRiD_best_segmentation_pixel_masks'
NAMES = ['EX', 'HE', 'SE', 'MA']
MAP = {0: 'MA', 1: 'HE', 2: 'EX', 3: 'SE'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    cv2.setNumThreads(1)
    sys.path.insert(0, str(OLD / 'reports'))
    from prepare_external_manifest import fingerprint
    old_manifest = json.loads((DERIVED / 'manifest.json').read_text(encoding='utf-8'))
    samples = [r for r in old_manifest['samples'] if r['source_split'] == 'testing']
    ids = [f'IDRiD_{i:02d}' for i in range(55, 82)]
    assert [r['sid'] for r in samples] == ids
    old_fp_path = OLD / 'reports/external_acceptance/image_fingerprints.json'
    old_fp = json.loads(old_fp_path.read_text(encoding='utf-8'))
    all_old = [r for group in old_fp.values() for r in group]
    # Reuse the prior scene audit only after checking EVERY image's decoded pixels.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        current = list(pool.map(fingerprint, all_old))
    for prior, now in zip(all_old, current):
        for key in ['pixel_sha256', 'full_hash', 'black_bbox_hash', 'dimensions']:
            assert prior[key] == now[key], (prior['image'], key)
    dedup_path = OLD / 'reports/external_acceptance/image_dedup.final.json'
    dedup = json.loads(dedup_path.read_text(encoding='utf-8'))
    assert dedup['status'] == 'PASS' and dedup['confirmed_duplicates'] == 0
    candidates = [r for r in dedup['candidates'] if any(r[k]['source'] == 'IDRiD' and r[k]['sid'] in ids for k in ['a', 'b'])]
    by_id = {(r['source'], r['sid']): r for r in current}
    rows = []
    canvas = Image.new('RGB', (1200, 960), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, old in enumerate(samples):
        image = Path(old['image_path'])
        txt = Path('D:/dataset/best_segmentation/labels/testing') / (old['sid'] + '.txt')
        assert sha(image) == old['source_image_sha256']
        assert sha(txt) == old['source_polygon_sha256']
        rgb = np.asarray(Image.open(image).convert('RGB'))
        h, w = rgb.shape[:2]
        assert [w, h] == old['size_wh']
        polygons = {c: [] for c in NAMES}
        for line in txt.read_text(encoding='utf-8').splitlines():
            v = line.split()
            if not v: continue
            assert float(v[0]) == int(v[0]) and int(v[0]) in MAP
            xy = np.asarray(v[1:], np.float64).reshape(-1, 2)
            assert len(xy) >= 3 and np.isfinite(xy).all() and ((xy >= 0) & (xy <= 1)).all()
            p = np.rint(xy * [w, h]).astype(np.int32)
            p[:, 0] = p[:, 0].clip(0, w-1); p[:, 1] = p[:, 1].clip(0, h-1)
            polygons[MAP[int(v[0])]].append(p)
        masks = {}; stack = []
        for c in NAMES:
            p = DERIVED / old['masks'][c]['path']
            assert sha(p) == old['masks'][c]['sha256']
            a = np.asarray(Image.open(p))
            assert a.shape == (h, w) and set(np.unique(a)).issubset({0, 255})
            b = np.zeros((h, w), np.uint8)
            if polygons[c]: cv2.fillPoly(b, polygons[c], 255)
            assert np.array_equal(a, b), (old['sid'], c)
            assert np.count_nonzero(a) == old['masks'][c]['positive_pixels']
            masks[c] = {'path': str(p), 'sha256': sha(p), 'positive_pixels': int(np.count_nonzero(a)),
                        'valid_entire_image': True, 'empty_is_negative': not bool(a.any()),
                        'completeness_authority': 'historical user confirmation of official-GT lossless conversion; not independently compared to original GT'}
            stack.append(a > 0)
        stack = np.stack(stack)
        pixel_sha = by_id['IDRiD', old['sid']]['pixel_sha256']
        exact = [r for r in current if r['pixel_sha256'] == pixel_sha and not (r['source'] == 'IDRiD' and r['sid'] == old['sid'])]
        assert not exact, (old['sid'], exact)
        rows.append({'sid': old['sid'], 'image_path': str(image), 'image_sha256': sha(image),
                     'polygon_path': str(txt), 'polygon_sha256': sha(txt), 'hw': [h, w],
                     'decoded_rgb_sha256': pixel_sha, 'masks': masks,
                     'overlapping_GT_pixels': int((stack.sum(0) > 1).sum())})
        if i < 3:
            raw = Image.fromarray(rgb); raw.thumbnail((570, 280))
            overlay = rgb.copy()
            for j, color in enumerate([[230, 160, 0], [0, 114, 178], [204, 121, 167], [0, 158, 115]]):
                overlay[stack[j]] = (.45 * overlay[stack[j]] + .55 * np.asarray(color)).astype(np.uint8)
            ov = Image.fromarray(overlay); ov.thumbnail((570, 280))
            canvas.paste(raw, (10, 320*i+30)); canvas.paste(ov, (610, 320*i+30))
            draw.text((10, 320*i+5), f'{old["sid"]} stored {w} x {h} | EX orange, HE blue, SE purple, MA green', fill='black')
        print(json.dumps({'audited_external': i+1, 'sid': old['sid'], 'hw': [h, w]}), flush=True)
    manifest = {'status': 'FROZEN_BEFORE_EXTERNAL_PREDICTIONS', 'date': '2026-10-23', 'expected_images': 27,
                'usage_history': 'Direct user confirmation in current chat: never trained, selected, tuned or used to compare schemes',
                'history_authority': 'user report, consistent with archived quarantine; no external server audit of all historic runs',
                'dataset_training_exposure': 'Other project Q_MIX/E_NEG explorations used IDRiD training54; selected S20 and matched C20 follow DDR-only Q_DDR->F12->S/C lineage, not those mixed-data parents; these27 testing images isolated by direct user confirmation',
                'lineage_evidence': {str(p): sha(p) for p in [ROOT.parent/'paper_writing_20261023/evidence/Q_parent_identity.json',
                    ROOT.parent/'HRDecoder_NextPlan_20261004/execution/configs/protocol.json',
                    ROOT.parent/'HRDecoder_FinalDecision_20261005/execution/server_results/runs/S_seed42/identity.json',
                    ROOT.parent/'HRDecoder_FinalDecision_20261005/execution/server_results/runs/C_seed42/identity.json']},
                'patient_identity_available': False, 'official_native_geometry_claimed': False,
                'annotation_authority': 'historical user-confirmed official-GT lossless conversion, stored coordinate masks reverified',
                'protocol_sha256': sha(ROOT / 'plan/external_protocol.md'),
                'source_manifest_sha256': sha(DERIVED / 'manifest.json'),
                'dedup_audit': {'status': 'PASS_REUSED_AFTER_ALL_RGB_FINGERPRINTS_REVERIFIED',
                                'images_reverified': len(current), 'candidate_pairs_involving_holdout': len(candidates),
                                'confirmed_duplicates': 0, 'patient_independence_proven': False,
                                'historical_audit_path': str(dedup_path), 'historical_audit_sha256': sha(dedup_path),
                                'historical_fingerprints_sha256': sha(old_fp_path),
                                'holdout_candidates': candidates},
                'samples': rows}
    write(ROOT / 'evidence/external_holdout_manifest.json', manifest)
    canvas.save(ROOT / 'evidence/external_GT_first_three.png')
    print(json.dumps({'status': manifest['status'], 'images': 27, 'fingerprints': len(current)}), flush=True)


if __name__ == '__main__': main()
