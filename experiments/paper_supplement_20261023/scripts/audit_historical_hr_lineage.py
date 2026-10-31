"""Extract documentary lineage from real retained checkpoints and original JSON log.

This does not infer a historical parent from a newly retrained comparison model.
"""
import datetime
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT.parent / 'HRDecoder_Ready_20261002/best_pipeline/weights'
LOG = Path('C:/Users/GuanXingRen/AppData/experment/fundus-seg-2026/'
           'hrdecoder-experiments/week2/logs/e4_detail_20260817_034605.log.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def init_lines(config):
    return [line for line in config.splitlines()
            if any(token in line for token in ('pretrained=', 'load_from',
                   'resume_from', 'source_checkpoint', 'reference_checkpoint'))]


def main():
    target = ROOT / 'evidence/historical_hr_lineage_sources'
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ['E4_weights.pth', 'B_weights.pth', 'D_student_2000.pth', 'E_1500.pth']:
        path = WEIGHTS / name
        ckpt = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        meta = ckpt['meta']
        config = meta['config']
        saved = target / (path.stem + '_checkpoint_config.txt')
        saved.write_text(config, encoding='utf-8')
        rows.append({'checkpoint': str(path), 'sha256': sha(path),
                     'config_export': str(saved), 'config_export_sha256': sha(saved),
                     'metadata': {key: meta.get(key) for key in
                         ['exp_name', 'seed', 'iter', 'time', 'hook_msgs']},
                     'initialization_lines': init_lines(config)})
    original = json.loads(LOG.read_text(encoding='utf-8').splitlines()[0])
    config = original['config']
    saved = target / 'E4_original_log_config.txt'
    saved.write_text(config, encoding='utf-8')
    log = {'path': str(LOG), 'sha256': sha(LOG), 'config_export': str(saved),
           'config_export_sha256': sha(saved), 'initialization_lines': init_lines(config)}
    assert 'load_from = None' in config and 'resume_from = None' in config
    assert "pretrained='open-mmlab://msra/hrnetv2_w48'" in config
    assert log['initialization_lines'] == rows[0]['initialization_lines']
    assert any('best_mIoU_iter_36000.pth' in s for s in rows[1]['initialization_lines'])
    for row in rows[2:]:
        assert any("source_checkpoint='weights/b_weights.pth'" in s
                   for s in row['initialization_lines'])
    result = {'status': 'PASS_DOCUMENTARY_LINEAGE_EXTRACTION',
              'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'checkpoints': rows, 'original_E4_log': log,
              'supported_history': [
                  'Retained E4 checkpoint and original training JSON both record ImageNet HRNet initialization, load_from=None, resume_from=None.',
                  'B checkpoint records loading E4 best at36000.',
                  'D and E checkpoint configs record source_checkpoint=weights/b_weights.pth.',
                  'New C/S adaptation runs use an extant F12 raw-model parent, independently verified by training_parent_export.json; it is not the newly retrained HRDecoder baseline.'
              ],
              'limits': [
                  'Documentary metadata is not a byte-level record of every historical initialization operation.',
                  'No recovered evidence in this audit establishes E4 initialization from historical split613 HRDecoder best32000.',
                  'Absence of that original baseline file now does not imply it never existed or that descendant checkpoints are absent.',
                  'A newly trained comparison baseline cannot retroactively establish the parent of an existing improved checkpoint.'
              ],
              'historical_42_20_parent_recovered': False,
              'new_training_or_evaluation_performed_by_this_script': False}
    (ROOT / 'evidence/historical_hr_lineage_audit.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checkpoints': len(rows),
                      'original_E4_log_verified': True,
                      'historical_42_20_parent_recovered': False}))


if __name__ == '__main__':
    main()
