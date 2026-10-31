"""One-time audited source revision and checkpoint metadata migration, while paused."""
import copy
import os
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import digest,read_json,sha,utc_now,write_json


def main():
    archive=ROOT/'reports/performance_revision_20261004'
    old_source=read_json(archive/'source_identity.before.json')
    old_gate=read_json(archive/'gpu_readiness.before.json')
    accepted=read_json(ROOT/'reports/performance_acceptance.json')
    assert accepted['status']=='PASS'
    assert accepted['performance_sha256']==sha(ROOT/'performance.py')
    assert accepted['config_sha256']==sha(ROOT/'configs/performance.json')
    assert not (ROOT/'reports/driver.lock').exists() and not (ROOT/'reports/campaign.lock').exists()
    assert read_json(ROOT/'reports/driver_state.json')['stage']=='TECHNICAL_PAUSE'
    assert old_gate['source_sha256']==old_source['sha256']
    changed={'train.py','report.py'}
    for rel,h in old_source['files'].items():
        if rel not in changed:assert sha(ROOT/rel)==h, 'Nonperformance source changed: '+rel
        else:assert sha(archive/'source'/rel)==h
    files=dict(old_source['files'])
    for rel in changed|{'performance.py','configs/performance.json','ops/probe_performance.py',
                        'ops/apply_performance_revision.py','ops/resume_driver.py'}:
        files[rel]=sha(ROOT/rel)
    source={'files':files,'sha256':digest(files),'created_at':utc_now(),
            'previous_source_sha256':old_source['sha256'],
            'revision_reason':'User authorized IO/cache/CPU audit acceleration; model/loss/optimizer/stream unchanged'}
    run=ROOT/'runs/E_NEG_seed42';old_identity=read_json(run/'identity.json')
    assert read_json(run/'status.json')['stage']=='TECHNICAL_PAUSE'
    state=torch.load(run/'last.pth',map_location='cpu',weights_only=False)
    assert state['identity']==old_identity and state['identity_sha256']==digest(old_identity)
    assert old_identity['source_sha256']==old_source['sha256']
    assert state['completed']==1600 and state['scheduler']['completed']==1600
    assert state['cursor']==1600*8
    assert (run/'valid_audit_u1600.json').exists()
    rows=[__import__('json').loads(x) for x in (ROOT/'reports/stream_E_NEG_seed42.jsonl').read_text().splitlines()]
    assert state['next_rows_sha256']==digest(rows[1600*8:1600*8+8])
    assert sha(ROOT/'configs/runtime.server.json')==old_identity['runtime_sha256']==old_gate['runtime_sha256']
    previous_checkpoint=archive/'checkpoint_before_performance_u1600.pth'
    assert not previous_checkpoint.exists(), 'Revision already applied; do not apply twice'
    os.link(run/'last.pth',previous_checkpoint)
    before_sha=sha(previous_checkpoint)
    identity=copy.deepcopy(old_identity)
    identity['source_sha256']=source['sha256']
    identity['performance_config_sha256']=accepted['config_sha256']
    state['identity']=identity;state['identity_sha256']=digest(identity)
    temporary=run/'last.performance.tmp'
    torch.save(state,temporary);os.replace(temporary,run/'last.pth')
    write_json(run/'identity.json',identity)
    write_json(ROOT/'reports/source_identity.json',source)
    gate=copy.deepcopy(old_gate)
    gate.update(source_sha256=source['sha256'],performance_acceptance_sha256=sha(ROOT/'reports/performance_acceptance.json'),
        performance_revision_at=utc_now(),original_gpu_acceptance_preserved=str((archive/'gpu_readiness.before.json').relative_to(ROOT)))
    gate['E_NEG_paired_acceptance']['execution_IO_revision']={
        'cached_decoding_and_worker_count_changed':True,'historical_micro2_acc4_model_loss_optimizer_schedule_preserved':True,
        'deterministic_comparison_model_gradients_optimizer_RNG_audit_equal':True,
        'acceptance':'reports/performance_acceptance.json'}
    a=accepted['arms'];val=old_gate['measurements']
    e_seconds=2000*a['E_NEG']['optimized_update_mean_seconds']+22000*a['F4']['optimized_update_mean_seconds']+7*207.498169776
    f_seconds=sum(a[x]['optimized_update_mean_seconds']*12000+val[x]['full76val_seconds']*4 for x in ('F4','F12'))
    gate['optimized_estimate']={
        'discovery_hours':1.2*(e_seconds+f_seconds)/3600+.5,
        'all_promoted_hours':2.4*(e_seconds+f_seconds)/3600+1.,
        'E_after2k_uses_F4_compute_estimate':True,'validation_unchanged':True,'final_test_compression_not_included':True}
    write_json(ROOT/'reports/gpu_readiness.json',gate)
    receipt={'status':'APPLIED','time':utc_now(),'user_authorized':True,'run':'E_NEG_seed42','resume_completed':1600,
        'old_source_sha256':old_source['sha256'],'new_source_sha256':source['sha256'],
        'original_checkpoint_sha256':before_sha,'migrated_checkpoint_sha256':sha(run/'last.pth'),
        'checkpoint_mutation_only':['identity.source_sha256','identity.performance_config_sha256','identity_sha256'],
        'model_optimizer_scheduler_RNG_cursor_unchanged':True,'old_identity':old_identity,'new_identity':identity,
        'changed_existing_files':sorted(changed),'performance_acceptance_sha256':sha(ROOT/'reports/performance_acceptance.json'),
        'discarded_uncheckpointed_updates':0,'old_logs_preserved':True,'new_gate_estimate':gate['optimized_estimate']}
    write_json(ROOT/'reports/performance_revision.json',receipt)
    print(receipt['status'],receipt['new_gate_estimate'],flush=True)


if __name__=='__main__':main()
