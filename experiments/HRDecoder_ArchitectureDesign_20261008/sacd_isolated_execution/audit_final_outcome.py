"""Audit archived evidence for the actual val+test goal, without SSH or weights."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RECEIPTS=ROOT/'cache_execution/server_receipts/sacd_isolated'
checks=[]
CLASSES=['EX','HE','SE','MA']

def read(relative):
    p=RECEIPTS/relative
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None

def check(name,condition,evidence):
    checks.append({'requirement':name,'status':'PENDING' if condition is None else 'PASS' if condition else 'FAIL',
                   'evidence':evidence})

def valid_counts(result,images):
    if result is None:return None
    try:
        assert result['images']==images and result['class_order']==CLASSES
        assert result['threshold']=='strictly > .5'
        assert len(result['counts'])==4
        computed=[]
        for i,row in enumerate(result['counts']):
            assert len(row)==3 and all(type(x) is int and x>=0 for x in row)
            tp,fp,fn=row
            for key,denom in [('IoU_percent',tp+fp+fn),('precision_percent',tp+fp),('recall_percent',tp+fn)]:
                value=100*tp/denom if denom else 0
                assert math.isclose(value,result[key][i],abs_tol=1e-8)
            computed.append(result['IoU_percent'][i])
        assert math.isclose(sum(computed)/4,result['mIoU_percent'],abs_tol=1e-8)
        return True
    except (AssertionError,KeyError,TypeError,ValueError):return False

protocol=read('training_runs/sacd_isolated/protocol.json')
digest=protocol['identity_SHA'] if protocol else None
if protocol:
    identity=protocol['identity'];p=identity['protocol']
    recomputed=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    check('2048 batch2 40000 freshImageNet seed42 single-model protocol',
          p['canvas']==2048 and p['global_batch']==2 and p['updates']==40000 and p['seed']==42
          and p['private_seed']==20261008 and not p['student_TTA'] and not p['inference_teacher']
          and not p['inference_online_H4'] and protocol['initialization'].startswith('fresh ImageNet')
          and recomputed==digest,'training_runs/sacd_isolated/protocol.json')
    matches=all(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==value for n,value in identity['code'].items())
    check('current reviewed core source matches training identity',matches,'protocol.identity.code + local core files; no data hash rescan')
    check('reviewed architecture specification matches trained identity',
          hashlib.sha256((HERE/'architecture_spec.json').read_bytes()).hexdigest()==identity['spec_SHA'],
          'protocol.identity.spec_SHA + local architecture_spec.json')
    metadata_path=ROOT/'cache_execution/server_receipts/metadata/data_identity.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    sid_sets={name:{row['sid'] for row in metadata['splits'][name]['samples']}
              for name in ('train','val','test')}
    image_sets={name:{row['image_sha256'] for row in metadata['splits'][name]['samples']}
                for name in ('train','val','test')}
    check('exact original data identity and disjoint454/76/227 held-out splits',
          hashlib.sha256(metadata_path.read_bytes()).hexdigest()==identity['data_identity_SHA']
          and metadata['status']=='PASS' and metadata['classes']==CLASSES
          and tuple(len(sid_sets[name]) for name in ('train','val','test'))==(454,76,227)
          and all(not (sets[a]&sets[b]) for sets in (sid_sets,image_sets)
                  for a,b in (('train','val'),('train','test'),('val','test'))),
          'archived metadata/data_identity.json matches training protocol; existing metadata hashes only')
else:check('training protocol and identity',None,'training_runs/sacd_isolated/protocol.json missing')

cpu=read('sacd_isolated_execution/CPU_SACD_PASS.json');resume=read('sacd_isolated_execution/CHECKPOINT_RESUME_PASS.json')
smoke=read('training_runs/sacd_isolated/GPU_SMOKE_PASS.json')
check('pinned-environment CPU checks and SGD/RNG resume',
      None if not cpu or not resume else cpu['status']==resume['status']=='PASS'
      and cpu['torch']==resume['torch']=='2.0.1+cu117'
      and resume['SGD_momentum_resume_bitwise_equal'] and resume['RNG_restored'],
      'sacd_isolated_execution/CPU_SACD_PASS.json + CHECKPOINT_RESUME_PASS.json')
check('full2048/batch2 GPU gate, fresh initialization, discarded probes',
      None if not smoke else smoke['status']=='PASS' and smoke['identity_SHA']==digest
      and smoke['batch']==2 and smoke['canvas']==2048 and smoke['formal_updates']==0
      and smoke['discarded_optimizer_steps']==12 and smoke['fresh_initialization_verified']
      and smoke['all_gradients_finite'],'training_runs/sacd_isolated/GPU_SMOKE_PASS.json')
registry=read('reports/sacd_isolated/SCREEN_STARTED.json')
screen=read('training_runs/sacd_isolated/SCREEN_PASS.json')
extension=read('reports/sacd_isolated/DEADLINE_EXTENSION_ACTIVATED.json')
check('one fresh candidate registered and screen passed before full40k continuation',
      None if not registry or not screen or not extension else
      registry['candidate_count']==registry['candidate_limit']==1 and registry['identity_SHA']==digest
      and not registry['warm_diagnostic_checkpoint_used'] and not registry['full40k_authorized_now']
      and screen['status']=='PASS' and screen['identity_SHA']==digest and screen['screen_updates']==8000
      and screen['full_val_counts_verified'] and all(screen['decisions'].values())
      and screen['time_unix']<1791568800 and extension['full40k_authorized']
      and extension['screen_identity_SHA']==digest and extension['same_fresh_run_continues'],
      'SCREEN_STARTED + SCREEN_PASS + DEADLINE_EXTENSION_ACTIVATED; original02:00 cutoff superseded conditionally')
check('reviewed fresh screening gate source matches training identity',None if not protocol else
      identity['screen_gate_code_SHA']==hashlib.sha256((HERE/'screen_gate.py').read_bytes()).hexdigest()
      and identity['screen_protocol_SHA']==hashlib.sha256((HERE/'screen_protocol.json').read_bytes()).hexdigest(),
      'identity.screen_gate_code_SHA + identity.screen_protocol_SHA')
isolation=read('sacd_isolated_execution/FEATURE_ISOLATION_CPU_PASS.json')
check('pinned CPU feature-path cut and unchanged forward/direct-base gradients',None if not isolation else
      isolation['status']=='PASS' and isolation['torch']=='2.0.1+cu117'
      and isolation['private_feature_gradient_cut'] and isolation['forward_and_loss_bitwise_equal']
      and isolation['direct_z0_gradient_preserved_bitwise'],
      'FEATURE_ISOLATION_CPU_PASS.json; synthetic proof is not an accuracy claim')

complete=read('training_runs/sacd_isolated/TRAINING_COMPLETE.json')
outcome=read('reports/sacd_isolated/GPU_CAMPAIGN_OUTCOME.json')
val=read('training_runs/sacd_isolated/val_40000.json')
test=read('training_runs/sacd_isolated/test_40000.json')
lock=read('reports/sacd_isolated/LOCKED_TEST_CHECKPOINT.json')
check('40000 updates and persistent checkpoint loaded/verified on server',None if not complete else
      complete['status']=='COMPLETE' and complete['completed']==40000 and complete['identity_SHA']==digest
      and complete['primary_checkpoint']=='last.pt' and complete['durable_checkpoint_loaded_and_verified'],
      'training_runs/sacd_isolated/TRAINING_COMPLETE.json; no checkpoint download')
check('complete76-image fixed40k val counts and metrics',valid_counts(val,76),'training_runs/sacd_isolated/val_40000.json')
reference_path=ROOT/'sacd_execution/val_label_reference.json'
reference=json.loads(reference_path.read_text()) if reference_path.exists() else None
check('40k counts preserve independently verified76-label ground-truth totals',
      None if not val or not reference else reference['status']=='PASS'
      and reference['images']==76 and reference['class_order']==CLASSES
      and reference['metadata_file_SHA256']==protocol['identity']['data_identity_SHA']
      and reference['shapes_and_labels_verified']
      and all([row[0]+row[2] for row in result['counts']]==reference['ground_truth_positive_pixels']
              and all(row[0]+row[1]<=reference['valid_pixels'] for row in result['counts'])
              for result in (val,val['base_head_metrics'])),
      'val_label_reference.json independently counted existing local76 val labels; no test-label access')
check('fixed40k val strictly exceeds45.32183122738841%',None if not val else
      val['completed']==40000 and val['mIoU_percent']>45.32183122738841,'val_40000.json; best-val is secondary')
check('fixed40k checkpoint locked before test, exact evaluation source',None if not lock else
      lock['completed']==40000 and lock['identity_SHA']==digest and not lock['test_used_for_selection']
      and lock['checkpoint']=='training_runs/sacd_isolated/last.pt'
      and lock['selected_by']=='pre-registered fixed40k endpoint'
      and lock['evaluation_code_SHA256']==hashlib.sha256((HERE/'test_eval.py').read_bytes()).hexdigest(),
      'reports/sacd_isolated/LOCKED_TEST_CHECKPOINT.json')
check('complete227-image test counts and metrics',valid_counts(test,227),'training_runs/sacd_isolated/test_40000.json')
check('test strictly exceeds45.860383832668%',None if not test else
      test['completed']==40000 and test['identity_SHA']==digest and test['mIoU_percent']>45.860383832668,
      'training_runs/sacd_isolated/test_40000.json')
check('test on locked checkpoint, held-out images, no GT/teacher/TTA inference or selection',None if not test or not lock else
      test['checkpoint_lock']==lock and test['completed_unix']>lock['locked_before_test_unix']
      and lock['custom_student_split_SID_disjoint'] and lock['existing_image_content_hashes_disjoint']
      and not test['test_for_selection'] and not test['inference_teacher_or_GT']
      and not test['TTA'] and not test['online_H4'],
      'locked test receipt + reviewed model/test source; custom student split, no official teacher-split independence claim')
check('test follows the user-authorized conditional extension',None if not test or not extension else
      extension['status']=='ACTIVE' and extension['screen_pass_unix']<1791568800
      and test['completed_unix']>extension['screen_pass_unix'],
      'User: through screening, server can be extended until completion; no obsolete02:00 final-test gate')
check('controller completed without shutdown',None if not outcome else
      outcome['status']=='COMPLETE' and outcome['exit_code']==0
      and not outcome['shutdown_command_issued'] and outcome['server_kept_running'],
      'reports/sacd_isolated/GPU_CAMPAIGN_OUTCOME.json')
snapshot_path=RECEIPTS/'latest_snapshot.json'
snapshot=json.loads(snapshot_path.read_text(encoding='utf-8')) if snapshot_path.exists() else None
check('server SSH online verified after final controller outcome',None if not outcome or not snapshot else
      snapshot['runtime']['server_online'] and snapshot['runtime']['checked_at_unix']>=outcome['time_unix'],
      'latest_snapshot.runtime.checked_at_unix; SSH observation required, not controller intent alone')

statuses={c['status'] for c in checks}
result={'audit_time':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'status':'CONTRADICTED' if 'FAIL' in statuses else 'INCOMPLETE' if 'PENDING' in statuses else 'GOAL_VERIFIED',
        'goal_achieved':statuses=={'PASS'},'primary_endpoint':40000,'checks':checks,
        'current_val_mIoU':val['mIoU_percent'] if val else None,'current_test_mIoU':test['mIoU_percent'] if test else None,
        'data_cache_hash_rescan':False,'checkpoint_download':False}
(HERE/'completion_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'goal_achieved':result['goal_achieved'],
                  'pending':[c['requirement'] for c in checks if c['status']=='PENDING'],
                  'failed':[c['requirement'] for c in checks if c['status']=='FAIL']}))
