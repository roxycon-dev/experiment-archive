"""Record the latest no-idle instruction and verify the completed stage's local receipts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now

def read(path):return read_json(ROOT/path)
now=utc_now()
for seed in (42,2024):
    b=read(f'reports/run_backup_Q_DDR_seed{seed}.json')
    assert b['complete'] and len(b['files'])==169
    assert sum(x['bytes'] for x in b['files'])==b['total_bytes']
    for x in b['files']:
        p=ROOT/'server_results'/x['path']
        assert p.is_file() and p.stat().st_size==x['bytes'],x['path']
    # Each file was SHA verified on download; retain those verified receipts.
final=read('reports/final_evidence_backup_receipt.json')
assert final['local_backup_state']=='SHA_VERIFIED'
for x in final['files']:
    assert sha(ROOT/'server_results'/x['path'])==x['sha256'],x['path']
assert sha(ROOT/'reports/final_selection.json')==final['lock_sha256']
export=read('reports/idrid_polygon_conversion_receipt.json')
assert export['mask_encode_decode_exact'] and export['source_txt_sha_unchanged']
assert sha(Path(export['output_root'])/'manifest.json')==export['manifest_sha256']

override=read('reports/execution_scope_override.json')
override.update(updated_at_utc=now,latest_instruction_verbatim='我要去睡觉了，今晚不要让服务器空转哈',
    temporary_stop_authorized_by_latest_no_idle_instruction=True,
    temporary_stop_condition='No runnable authorized GPU task; required official external data/access missing; completed results recovered and SHA verified',
    temporary_stop_is_original_plan_completion=False,shutdown_allowed_now=True,
    temporary_stop_reason='Q_DDR two seeds and locked final test finished. Q_MIX cannot start without official data/access. No filler GPU training.',
    resume_condition='Official external native images and masks accepted, prospective continuation protocol frozen, user-assigned instance reachable',
    retain_original_plan_todos=True,retain_all_useful_assets=True)
write_json(ROOT/'reports/execution_scope_override.json',override)
scope=read('reports/original_plan_completion.json')
scope.update(updated_at_utc=now,original_plan_complete=False,shutdown_allowed=True,
    temporary_stop_for_no_idle=True)
scope['tasks']['server_shutdown_and_OFF_verification'].update(
    state='TEMPORARY_STOP_PENDING_OFF_VERIFICATION',reason='Latest user no-idle instruction; original plan remains pending official data')
write_json(ROOT/'reports/original_plan_completion.json',scope)
status=read('reports/preparation_status.json')
status.update(updated_at=now,stage='TEMPORARY_STOP_PREPARED_PENDING_OFFICIAL_EXTERNAL_DATA',
    shutdown_allowed_now=True,original_plan_complete=False,
    temporary_stop_authorization='latest_explicit_no_idle_instruction',
    temporary_stop_reason='Completed authorized GPU work; official external data inaccessible without user input',
    next_action='WAIT_FOR_OFFICIAL_EXTERNAL_ACCESS_THEN_ACCEPT_DATA_AND_CONTINUE_Q_MIX',
    suppress_repeated_ssh_poll_while_stopped=True)
write_json(ROOT/'reports/preparation_status.json',status)
paths=['reports/run_backup_Q_DDR_seed42.json','reports/run_backup_Q_DDR_seed2024.json',
 'reports/final_evidence_backup_receipt.json','reports/final_selection.json',
 'reports/Q_DDR_两种子阶段结果_20261004.md','reports/Q_DDR_stage_analysis.json',
 'reports/Q_DDR_stage_report_receipt.json','reports/idrid_polygon_conversion_receipt.json',
 'reports/external_readiness.json','reports/execution_scope_override.json',
 'reports/original_plan_completion.json']
handoff={'created_at':now,'original_plan_complete':False,'temporary_stop_for_no_idle':True,
    'local_results_sha_verified':True,'useful_assets_preserved':True,'Q_MIX_ready':False,
    'final_selection_sha256':final['lock_sha256'],
    'receipts_and_reports':{p:sha(ROOT/p) for p in paths},
    'blocking_inputs':['Official IDRiD native originals/pixel GT or verified raw archive','MESSIDOR-1 personal download link for MAPLES pairing'],
    'no_fake_GPU_work':True,'platform_OFF_verified':False}
write_json(ROOT/'reports/temporary_stop_handoff.json',handoff)
print({'handoff_saved':True,'original_plan_complete':False,'temporary_stop_prepared':True,'local_results_SHA_receipts_verified':True})
