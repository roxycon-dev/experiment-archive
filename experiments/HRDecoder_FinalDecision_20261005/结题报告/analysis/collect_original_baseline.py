"""Extract the original DDR baseline from its actual saved training log/report."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('D:/dataset/lesion_segmentation')


def source(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


def main():
    log = ORIGINAL / '实验日志_json/01_板块一_baseline.json'
    document = ORIGINAL / '01_板块一_划分统计报告_baseline_report.md'
    config = ORIGINAL / '01_板块一_baseline实验配置.py'
    split_script = ORIGINAL / '01_板块一_数据集划分脚本_ddr_split.py'
    records = [json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    header = records[0]
    training_step = 0
    curve = []
    for line_number, record in enumerate(records, 1):
        if record.get('mode') == 'train':
            training_step = record['iter']
        elif record.get('mode') == 'val':
            # The validation record's iter=76 is the number of validation images.
            assert record['iter'] == 76
            curve.append({'completed_training_updates': training_step,
                          'n_images': record['iter'], 'miou_pct': 100 * record['mIoU'],
                          'source_line': line_number, 'saved_record': record})
    assert [r['completed_training_updates'] for r in curve] == list(range(4000, 40001, 4000))
    best = max(curve, key=lambda r: r['miou_pct'])
    assert best['completed_training_updates'] == 32000 and abs(best['miou_pct'] - 39.39) < 1e-10
    text = document.read_text(encoding='utf-8')
    assert '42.20' in text and 'best_mIoU_iter_32000.pth' in text
    per_class_reported = {'EX': 51.56, 'HE': 43.13, 'SE': 49.98, 'MA': 24.11}
    data = {
        'stage': 'ORIGINAL_BASELINE_SOURCE_AUDIT',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'experiment_name': header['exp_name'], 'seed': header['seed'],
        'code_version': header['mmseg_version'],
        'reported_split_sizes': {'train': 454, 'val': 76, 'test': 227},
        'source_files': [source(p) for p in (log, document, config, split_script)],
        'expanded_training_configuration_from_original_log': header['config'],
        'validation_curve': curve,
        'best_validation': best,
        'historical_test': {
            'n_images_reported': 227, 'miou_pct_reported': 42.20,
            'per_class_iou_pct_reported': per_class_reported,
            'inference_reported': 'No TTA; validation-selected best_mIoU_iter_32000.pth',
            'source': source(document),
            'precision': 'Historical document values rounded to two decimals; use approximate baseline differences, not fabricated extra digits.',
            'per_image_counts_found_in_this_audit': False,
        },
        'baseline_checkpoint_found_in_this_audit': False,
        'split_identity_and_evaluation_code_match_to_current': 'Audit pending; matching counts alone do not prove identical sample/label identity.',
        'no_baseline_retraining_or_reinference': True,
    }
    out = REPORT / 'evidence/original_HRDecoder_baseline_evidence.json'
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'best_validation': {'update': best['completed_training_updates'], 'miou_pct': best['miou_pct']},
                      'reported_test_miou_pct': 42.20, 'original_log_sha256': source(log)['sha256'],
                      'historical_test_value_is_report_precision': True}, ensure_ascii=True))


if __name__ == '__main__':
    main()
