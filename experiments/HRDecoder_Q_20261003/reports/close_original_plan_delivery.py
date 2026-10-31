"""Record complete scientific delivery and cleanup; platform OFF remains separate."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now

def main():
    target=ROOT/'reports/original_plan_delivery_manifest.json'
    assert not target.exists(), 'Review an existing delivery manifest instead of replacing it'
    now=utc_now()
    final=read_json(ROOT/'reports/continuation_final_evidence_backup_receipt.json')
    report=read_json(ROOT/'reports/original_plan_report_receipt.json')
    immutable=read_json(ROOT/'reports/immutable_campaign_asset_backup_receipt.json')
    cleanup_path=ROOT.parent/'_maintenance/next_retired_U1_cleanup_result_20261004.json'
    cleanup=read_json(cleanup_path)
    assert final['stage']=='COMPLETE_SHA_VERIFIED' and final['final_process_exited']
    assert report['stage']=='COMPLETE_LOCAL_SHA_VERIFIED' and report['lock_sha256']==final['lock_sha256']
    assert immutable['stage']=='COMPLETE_SHA_VERIFIED'
    assert cleanup['stage']=='COMPLETE_EXACT_ALLOWLIST_DELETION_PROTECTED_ASSETS_VERIFIED'
    assert cleanup['protected_metadata_SHA_verified'] and cleanup['protected_baseline_caches_verified']
    assert not cleanup['checkpoint_deletion_performed'] and not cleanup['recursive_delete_performed']
    evidence=[]
    for prefix,receipt in [(ROOT,report),(ROOT/'server_results/continuation_evidence',final),(ROOT/'server_results',immutable)]:
        for row in receipt['files']:
            p=prefix/row['path']
            assert p.is_file() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
            evidence.append(dict(path=str(p.resolve()),bytes=row['bytes'],sha256=row['sha256']))
    assert len(immutable['frozen_source_present_locally'])==251
    for row in immutable['frozen_source_present_locally']:
        assert (ROOT/row['path']).is_file() and sha(ROOT/row['path'])==row['sha256']
    run_receipts=[]
    for arm in ('Q_DDR','Q_MIX'):
        for seed in (42,2024):
            rel='reports/run_backup_%s_seed%s_24k.json'%(arm,seed)
            backup=read_json(ROOT/rel)
            assert backup['complete'] and backup['endpoint']==24000 and len(backup['files'])==327
            for row in backup['files']:
                p=ROOT/'server_results'/row['path']
                assert p.is_file() and p.stat().st_size==row['bytes'], 'Recovered run file missing or changed'
            status=read_json(ROOT/'server_results/runs'/('%s_seed%s'%(arm,seed))/'status.json')
            assert status['stage']=='COMPLETE' and status['completed']==24000
            run_receipts.append(dict(path=str((ROOT/rel).resolve()),sha256=sha(ROOT/rel),files=327,bytes=backup['total_bytes'],verified_at=backup['verified_at']))
    lock_path=ROOT/'server_results/continuation_evidence/reports/continuation_final_selection.json'
    lock=read_json(lock_path)
    assert sha(lock_path)==final['lock_sha256']
    assert sha(ROOT/'reports/final_selection.json')=='26e5852194aca66bb02062ac79b2f0f8026e33652a95133140a3824bffb38905'
    assert sha(ROOT/'reports/final_evaluation.py')==lock['driver_sha256']
    selected_models=[]
    for seed,spec in lock['checkpoints'].items():
        p=ROOT/'server_results/runs'/('Q_DDR_seed'+seed)/'checkpoint_u16000.pth'
        assert p.is_file() and sha(p)==spec['sha256']
        selected_models.append(dict(seed=int(seed),path=str(p.resolve()),sha256=spec['sha256']))
    assert final['summary']['no_new_test_inference'] and not final['summary']['selection_changed_after_test']
    p0=read_json(ROOT/'configs/p0_original_lock.json')
    manifest=dict(stage='COMPLETE_SCIENTIFIC_DELIVERY_LOCAL_SHA_AND_CLEANUP',created_at=now,
        selected=lock['selection']['selected'],selected_lock_path=str(lock_path.resolve()),selected_lock_sha256=final['lock_sha256'],
        selected_checkpoints=selected_models,original_P0_lock=str((ROOT/'configs/p0_original_lock.json').resolve()),
        full_system_formula='0.75 P0 + 0.25 Q_DDR(seed42, update16000); P0 and its seven-model expanded inventory retained in the full report',
        final_test_action=final['test_action'],test_scores_pct=final['summary']['seed_scores_pct'],mean_test_pct=final['summary']['mean_test_pct'],
        report=str((ROOT/'reports/HRDecoder_原方案完整实验报告_20261004.md').resolve()),
        run_backups=run_receipts,total_run_files=1308,verified_evidence=evidence,
        frozen_source_sha256=immutable['source_sha256'],frozen_source_files=251,
        cleanup_receipt=str(cleanup_path.resolve()),cleanup_receipt_sha256=sha(cleanup_path),cleanup_deleted_bytes=cleanup['deleted_bytes'],
        previous_test_and_all_model_evidence_preserved=True,original_plan_scientific_scope_complete=True,
        shutdown_authorized=True,shutdown_dispatched=False,platform_OFF_verified=False,
        actual_billing_receipt_available=False,automatic_additional_experiments=False)
    write_json(target,manifest)
    completion=read_json(ROOT/'reports/original_plan_completion.json')
    completion.update(updated_at=now,updated_at_utc=now,scientific_scope_complete=True,delivery_complete=True,
        original_plan_complete=False,shutdown_allowed=True,remaining_required_action='REQUEST_FINAL_SHUTDOWN_AND_VERIFY_PLATFORM_OFF',delivery_manifest='reports/original_plan_delivery_manifest.json')
    tasks=completion['tasks']
    for name in ('Q_MIX_paired_experiment','Q_MIX_review_common_24k','Q_DDR_common_24k_continuation'):
        tasks[name].update(state='COMPLETE_PAIRED_24K_LOCAL_SHA_VERIFIED',endpoint=24000,review_complete=True,actual_training=False)
    tasks['Q_MIX_paired_experiment'].update(current_run=None,next_handoff=None)
    tasks['Q_MIX_review_common_24k'].update(completed_update=24000,worker_pid=None,backup_receipt='reports/run_backup_Q_MIX_seed2024_24k.json')
    tasks['continuation_final_selection_and_evaluation'].update(state='COMPLETE_LOCAL_SHA_VERIFIED_EXACT_PRIOR_TEST_REUSE',selection_frozen=True,
        test_executed=False,test_evidence_complete=True,test_action=final['test_action'],lock_sha256=final['lock_sha256'],evidence_receipt='reports/continuation_final_evidence_backup_receipt.json',dependency_worker_pid=None)
    tasks['original_plan_final_reports_and_local_SHA_backup'].update(state='COMPLETE_LOCAL_SHA_VERIFIED',report=manifest['report'],receipt='reports/original_plan_report_receipt.json',delivery_manifest='reports/original_plan_delivery_manifest.json')
    tasks['dependency_audit_and_deferred_cleanup'].update(state=cleanup['stage'],receipt=str(cleanup_path),deleted_bytes=cleanup['deleted_bytes'])
    tasks['server_shutdown_and_OFF_verification'].update(state='AUTHORIZED_PENDING_FINAL_SHUTDOWN',reason='All applicable scientific work, necessary local backup, dependency audit and agreed cleanup complete; OFF not yet observed')
    write_json(ROOT/'reports/original_plan_completion.json',completion)
    preparation=read_json(ROOT/'reports/preparation_status.json')
    preparation.update(stage=manifest['stage'],updated_at=now,original_plan_scientific_scope_complete=True,original_plan_complete=False,
        shutdown_allowed_now=True,continuation_final_test_complete=True,continuation_final_evaluation_phase='COMPLETE',formal_process_alive=False,formal_pid=None,
        continuation_final_selection_sha256=final['lock_sha256'],next_action=completion['remaining_required_action'],delivery_manifest='reports/original_plan_delivery_manifest.json')
    write_json(ROOT/'reports/preparation_status.json',preparation)
    scope=read_json(ROOT/'reports/execution_scope_override.json')
    scope.update(updated_at_utc=now,shutdown_allowed_now=True,scientific_scope_complete=True,remaining_data_dependency=None,
        remaining_required_action=completion['remaining_required_action'],delivery_manifest='reports/original_plan_delivery_manifest.json')
    write_json(ROOT/'reports/execution_scope_override.json',scope)
    print('DELIVERY_COMPLETE',manifest['total_run_files'],'run files; final shutdown authorized; OFF unverified')

if __name__=='__main__': main()
