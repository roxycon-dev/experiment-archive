"""Prepare a reviewed, explicit file-only cleanup plan; this script cannot delete."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r'D:\dataset\experiment').resolve()
OUT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8-sig'))


inventory = read('_maintenance/checkpoint_cleanup_20261004/inventory.json')['files']
reasons = {}


def allow(relative, reason, evidence, expected_sha256=None):
    reasons[str(Path(relative))] = dict(reason=reason, evidence=evidence,
                                      expected_sha256=expected_sha256)


backups = read('HRDecoder_Ready_20261002/reports/verified_backups.json')
campaign = read('HRDecoder_Ready_20261002/server_results/work_dirs/campaign_state.json')
assert campaign['status'] == 'main_training_done' and campaign['active_job'] is None
for tag in ('U1_external_pretrain', 'U1_ddr_finetune', 'H1_external_pretrain',
            'H1_ddr_finetune', 'A0_ddr_finetune'):
    assert campaign['jobs'][tag]['status'] in ('complete', 'stopped')
    rec = backups[f'{tag}/ckpt_best.pth']
    allow(f'HRDecoder_Ready_20261002/server_results/work_dirs/{tag}/ckpt_best.pth',
          'Closed, unadopted Ready experiment: remove its obsolete full training checkpoint. '
          'Keep Next/assets/U1.pth and H1.pth as the existing model-only historical copies.',
          ['HRDecoder_Ready_20261002/server_results/work_dirs/campaign_state.json',
           '_全实验档案_20260920/04_本轮实验补充_20261002.md',
           'HRDecoder_Ready_20261002/best_pipeline/local_weight_manifest.json'], rec['sha256'])

completion = read('HRDecoder_Next_20261003/reports/completion_status.json')
selection = read('HRDecoder_Next_20261003/server_results/work/M_seed42/selection.json')
assert completion['experiment_and_review_complete']
assert not selection['second_seed_gate']
assert selection['adaptive_stop']['status'] == 'PAIRED_COST_STOP_AT_2000'
for arm in ('M0', 'M1'):
    rec = selection['adaptive_stop']['checkpoints'][arm]
    allow(f'HRDecoder_Next_20261003/server_results/work/M_seed42/{arm}/last.pth',
          'Closed M experiment stopped for paired negative results; no continuation authorized. '
          'Remove obsolete optimizer/RNG recovery checkpoint, keep this arm best.pth and all evidence.',
          ['HRDecoder_Next_20261003/reports/completion_status.json',
           'HRDecoder_Next_20261003/server_results/work/M_seed42/selection.json'], rec['last_sha256'])

r6 = read('HRDecoder_FusionAware_20260917/pulled/final_20260918T184943/r6_summary.json')
r6b = read('HRDecoder_FusionAware_20260917/pulled/final_20260918T184943/r6b_summary.json')
assert r6['h6']['HE']['decision'] == 'reject' and r6['p2']['decision'] == 'reject'
assert r6['p3']['adopted_heads'] == [] and r6b['status'] == 'rejected'
for name, digest, report in (
    ('head_HE.pth', r6['h6']['HE']['sha256'], 'r6_summary.json'),
    ('ma_expert_r6.pth', r6['p2']['sha256'], 'r6_summary.json'),
    ('ma_expert_r6b.pth', r6b['train']['sha256'], 'r6b_summary.json'),
):
    allow(f'HRDecoder_FusionAware_20260917/pulled/final_20260918T184943/{name}',
          'Explicitly rejected R6/R6b head/expert; no adopted head or deployment dependency.',
          [f'HRDecoder_FusionAware_20260917/pulled/final_20260918T184943/{report}'], digest)

for item in inventory:
    rel = item['relative_path']
    parts = Path(rel).parts
    if len(parts) >= 4 and parts[0] in ('HRDecoder_FullCompetition_20260920', 'HRDecoder_Ready_20261002'):
        if parts[1:3] in (('tests', '_tmp_training'), ('tests', '_tmp')):
            allow(rel, 'Regenerable synthetic test fixture; never a real experiment or pretrained model.',
                  [f'{parts[0]}/tests/run_training_smoke.py',
                   f'{parts[0]}/tests/run_contract_smoke.py'])

delete, keep = [], []
for item in inventory:
    rel = item['relative_path']
    if rel in reasons:
        assert not rel.startswith('HRDecoder_Q_20261003')
        delete.append({**item, **reasons[rel]})
    else:
        reason = 'Retain: potentially useful historical, selected, pretrained, or diagnostic checkpoint.'
        if rel.startswith('HRDecoder_Q_20261003'):
            reason = 'Protected: active Q campaign and official DINOv3 source weights.'
        elif '\\best_pipeline\\' in rel or Path(rel).name.lower() in (
            'b_weights.pth', 'e4_weights.pth', 'd_student_2000.pth', 'e_1500.pth',
            'h4_best.pth', 'u0_historical_best.pth', 'e4_reference_weights.pth',
            'backup_student_2000.pth'):
            reason = 'Protected: historical champion/P0 constituent or its source copy.'
        elif '\\M_seed42\\' in rel:
            reason = 'Retain best model for M architecture diagnosis; only obsolete last is deleted.'
        elif Path(rel).name in ('H1.pth', 'U1.pth'):
            reason = 'Retain compact historical model copy; obsolete full training checkpoint is deleted.'
        elif 'U0_highres' in rel or 'U0_continuation2048' in rel:
            reason = 'Retain selected high-resolution model and paired control for reproduction.'
        elif Path(rel).name in ('C_step_2000.pth', 'S_step_4000.pth'):
            reason = 'Retain historically selected/complementary model referenced by class shortlist.'
        keep.append({**item, 'reason': reason})

assert len(delete) == len(reasons), 'Allowlisted file missing from original inventory'
critical = {}
for name, spec in read('HRDecoder_Ready_20261002/best_pipeline/local_weight_manifest.json')['weights'].items():
    critical[spec['path']] = spec['sha256']
for spec in read('HRDecoder_Next_20261003/configs/assets.json').values():
    critical[str(ROOT / 'HRDecoder_Next_20261003/assets' / spec['file'])] = spec['sha256']
for spec in read('HRDecoder_Q_20261003/reports/run_backup_Q_DDR_seed42.json')['files']:
    if spec['path'].endswith('.pth'):
        critical[str(ROOT / 'HRDecoder_Q_20261003/server_results' / spec['path'])] = spec['sha256']
critical[str(ROOT / 'HRDecoder_Q_20261003/assets/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')] = (
    '73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c')
for arm in ('M0', 'M1'):
    critical[str(ROOT / f'HRDecoder_Next_20261003/server_results/work/M_seed42/{arm}/best.pth')] = (
        selection['adaptive_stop']['checkpoints'][arm]['best_sha256'])
assert not {x['path'] for x in delete}.intersection(critical)
plan = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'root': str(ROOT),
    'authorization': 'User explicitly requested deletion of confirmed useless failed local checkpoints on 2026-10-04.',
    'scope': 'Explicit individual files only; local workspace only; preserve uncertain files and all logs/data/configs/archives.',
    'delete': delete, 'keep': keep,
    'critical_expected_sha256': critical,
    'delete_count': len(delete), 'delete_bytes': sum(x['bytes'] for x in delete),
}
(OUT / 'cleanup_plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'delete_count': len(delete), 'delete_GiB': plan['delete_bytes']/2**30,
                  'keep_count': len(keep), 'plan': str(OUT/'cleanup_plan.json')}, ensure_ascii=False))
