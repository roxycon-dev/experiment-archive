"""Read small local receipts/core only; never SSH, read checkpoints, or use test to tune."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
RECEIPTS=ROOT/'cache_execution/server_receipts/residual'
checks=[]
def read(rel):
    path=RECEIPTS/rel
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
def check(name,value):checks.append({'requirement':name,'status':'PENDING' if value is None else 'PASS' if value else 'FAIL'})
def counts_valid(row,images):
    if row is None:return None
    try:
        assert row['images']==images and row['class_order']==['EX','HE','SE','MA']
        assert row['threshold']=='strictly > .5' and len(row['counts'])==4
        for i,(tp,fp,fn) in enumerate(row['counts']):
            assert all(type(n) is int and n>=0 for n in (tp,fp,fn))
            for key,denom in (('IoU_percent',tp+fp+fn),('precision_percent',tp+fp),('recall_percent',tp+fn)):
                assert math.isclose(100*tp/denom if denom else 0,row[key][i],abs_tol=1e-8)
        assert math.isclose(sum(row['IoU_percent'])/4,row['mIoU_percent'],abs_tol=1e-8)
        return True
    except (AssertionError,TypeError,KeyError,ValueError):return False

def main():
    protocol=read('training_runs/residual/protocol.json');digest=protocol['identity_SHA'] if protocol else None
    if protocol:
        material=protocol['identity'];p=material['protocol']
        check('fixed2048/batch2/40000 fresh seed42 private20261008, no task weights',
              p['canvas']==2048 and p['global_batch']==2 and p['updates']==40000 and p['seed']==42
              and p['private_seed']==20261008 and protocol['initialization'].startswith('fresh ImageNet')
              and hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()==digest)
        check('single-model deployment without teacher/GT/TTA/onlineH4',
              not p['inference_teacher'] and not p['student_TTA'] and not p['inference_online_H4']
              and 'detach' in p['gradient_route'])
        check('actual small core sources and screen specification match trained identity',
              all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==d for n,d in material['code'].items())
              and hashlib.sha256((HERE/'architecture_spec.json').read_bytes()).hexdigest()==material['spec_SHA']
              and hashlib.sha256((HERE/'screen_protocol.json').read_bytes()).hexdigest()==material['screen_protocol_SHA']
              and hashlib.sha256((HERE/'screen_gate.py').read_bytes()).hexdigest()==material['screen_gate_code_SHA'])
        # Use previously audited metadata identities, never rescan image/cache data.
        metadata_path=ROOT/'cache_execution/server_receipts/metadata/data_identity.json'
        if metadata_path.exists():
            metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
            sets={n:{r['sid'] for r in metadata['splits'][n]['samples']} for n in ('train','val','test')}
            hashes={n:{r['image_sha256'] for r in metadata['splits'][n]['samples']} for n in ('train','val','test')}
            check('original audited disjoint454/76/227 splits',
                  hashlib.sha256(metadata_path.read_bytes()).hexdigest()==material['data_identity_SHA']
                  and tuple(len(sets[n]) for n in ('train','val','test'))==(454,76,227)
                  and all(not(d[a]&d[b]) for d in (sets,hashes) for a,b in (('train','val'),('train','test'),('val','test'))))
        else:check('original audited disjoint454/76/227 splits',None)
    else:check('protocol and identity',None)
    cpu=read('residual_execution/CPU_RESIDUAL_PASS.json');smoke=read('training_runs/residual/GPU_SMOKE_PASS.json')
    repair=read('reports/residual/METRIC_VALIDATOR_REPAIR.json')
    adapter=read('residual_execution/METRIC_ADAPTER_CPU_PASS.json')
    check('scalar metric adapter identity bound without changing training or screen criteria',
          None if not repair or not adapter else repair['status']==adapter['status']=='PASS'
          and repair['training_identity_SHA']==digest
          and repair['validator_identity_SHA']==hashlib.sha256(json.dumps(repair['validator_identity'],sort_keys=True).encode()).hexdigest()
          and adapter['validation_helper_SHA256']==repair['validator_identity']['validation_helper_SHA256']
          ==hashlib.sha256((ROOT/'sacd_micro_execution/gate.py').read_bytes()).hexdigest()
          and repair['all_core_network_forward_loss_backward_sampler_optimizer_sources_unchanged']
          and repair['full_screen_gate_CPU_fixture_PASS']
          and not repair['screen_criteria_changed'] and not repair['raw_counts_or_val_artifacts_changed'])
    check('pinned CPU routing/Dice/deployment/SGD-RNG verification',None if not cpu else
          cpu['status']=='PASS' and cpu['torch']=='2.0.1+cu117'
          and 'actual_student_wrapper_private_base_gradient_isolation' in cpu['checks']
          and 'SGD_momentum_RNG_serialization_and_uncompressed40k_schedule' in cpu['checks'])
    check('full-shape fresh GPU smoke and all finite gradients',None if not smoke else
          smoke['status']=='PASS' and smoke['identity_SHA']==digest and smoke['canvas']==2048 and smoke['batch']==2
          and smoke['fresh_initialization_verified'] and smoke['formal_updates']==0 and smoke['all_gradients_finite'])
    start=read('reports/residual/SCREEN_STARTED.json');gate=read('training_runs/residual/SCREEN_PASS.json')
    extension=read('reports/residual/DEADLINE_EXTENSION_ACTIVATED.json')
    check('new fresh screen registered before results',None if not start else
          start['identity_SHA']==digest and start['initialization']=='fresh ImageNet W48 seed42/private20261008'
          and not start['warm_diagnostic_checkpoint_used'])
    check('registered screen actually passed before original deadline',None if not gate else
          gate['status']=='PASS' and gate['identity_SHA']==digest and gate['full_val_counts_verified']
          and all(gate['decisions'].values()) and gate['time_unix']<1791568800)
    check('conditional server extension on same fresh run',None if not extension else
          extension['status']=='ACTIVE' and extension['full40k_authorized'] and extension['same_fresh_run_continues']
          and extension['screen_identity_SHA']==digest and gate is not None and extension['screen_pass_unix']==gate['time_unix'])
    marker=read('training_runs/residual/TRAINING_COMPLETE.json');outcome=read('reports/residual/GPU_CAMPAIGN_OUTCOME.json')
    val=read('training_runs/residual/val_40000.json');test=read('training_runs/residual/test_40000.json')
    lock=read('reports/residual/LOCKED_TEST_CHECKPOINT.json')
    check('40000 completed and persistent own checkpoint loaded',None if not marker else
          marker['status']=='COMPLETE' and marker['completed']==40000 and marker['identity_SHA']==digest
          and marker['durable_checkpoint_loaded_and_verified'] and marker['primary_checkpoint']=='last.pt')
    check('fixed40k full76 val counts/metrics',counts_valid(val,76))
    ref=json.loads((ROOT/'sacd_execution/val_label_reference.json').read_text())
    check('fixed40k val GT totals match independent76-label reference',None if not val else
          [row[0]+row[2] for row in val['counts']]==ref['ground_truth_positive_pixels'])
    val_pass=val['mIoU_percent']>45.32183122738841 if val else None
    # Goal failure is different from contradictory experimental evidence.
    if val_pass:
        check('locked fixed40k test checkpoint selected before test',None if not lock or not test else
              lock['completed']==40000 and lock['identity_SHA']==digest and not lock['test_used_for_selection']
              and lock['selected_by']=='pre-registered fixed40k endpoint'
              and lock['evaluation_code_SHA256']==hashlib.sha256((HERE/'test_eval.py').read_bytes()).hexdigest()
              and test['checkpoint_lock']==lock and lock['locked_before_test_unix']<test['completed_unix'])
        check('independent227 test metrics/counts and deployment-only inference',counts_valid(test,227))
        check('test belongs to actual locked40k identity',None if not test else
              test['identity_SHA']==digest and test['completed']==40000 and test['split']=='test'
              and not test['inference_teacher_or_GT'] and not test['TTA'] and not test['online_H4'] and not test['test_for_selection'])
    snapshot_path=HERE/'latest_snapshot.json'
    runtime=json.loads(snapshot_path.read_text(encoding='utf-8'))['runtime'] if snapshot_path.exists() else None
    check('controller terminal outcome, server kept on, no external process control',None if not outcome else
          outcome['status']=='COMPLETE' and outcome['server_kept_running'] and not outcome['shutdown_command_issued']
          and not outcome['external_diagnostics_controlled'])
    terminal=runtime and outcome and runtime['checked_at_unix']>outcome['time_unix']
    check('actual post-completion SSH online and own training/controller/RAM ended',None if not terminal else
          runtime['server_online'] and not runtime['shutdown_command_issued']
          and all(not v['alive'] for v in runtime['own_process_checks'].values()) and not runtime['own_RAM']['alive'])
    test_pass=test['mIoU_percent']>45.860383832668 if test else None
    failed=[r for r in checks if r['status']=='FAIL'];pending=[r for r in checks if r['status']=='PENDING']
    status='CONTRADICTED' if failed else 'INCOMPLETE' if pending else 'GOAL_VERIFIED' if val_pass and test_pass else 'GOAL_NOT_ACHIEVED'
    result={'status':status,'goal_achieved':status=='GOAL_VERIFIED','checks':checks,
      'fixed40k_val_mIoU':val['mIoU_percent'] if val else None,'fixed40k_test_mIoU':test['mIoU_percent'] if test else None,
      'val_target':45.32183122738841,'test_target':45.860383832668,
      'val_target_exceeded':val_pass,'test_target_exceeded':test_pass,
      'audited_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'checkpoint_download':False,'data_cache_hash_rescan':False,'test_for_tuning':False}
    (HERE/'FINAL_OUTCOME_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':status,'goal_achieved':result['goal_achieved'],'failed':[r['requirement'] for r in failed],'pending':[r['requirement'] for r in pending]}))

if __name__=='__main__':main()
