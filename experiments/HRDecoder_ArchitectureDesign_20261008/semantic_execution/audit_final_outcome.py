"""Final audit of small local receipts, including own source-only resume chain."""
from datetime import datetime,timezone,timedelta
import hashlib,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;R=ROOT/'cache_execution/server_receipts/semantic'
checks=[]
def read(rel):
 p=R/rel
 return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(name,value):checks.append({'requirement':name,'status':'PENDING' if value is None else 'PASS' if value else 'FAIL'})
def counts_valid(v,n):
 if v is None:return None
 try:
  assert v['images']==n and v['class_order']==['EX','HE','SE','MA'] and v['threshold']=='strictly > .5'
  assert len(v['counts'])==4
  for i,(tp,fp,fn) in enumerate(v['counts']):
   assert all(type(x) is int and x>=0 for x in (tp,fp,fn))
   for k,d in (('IoU_percent',tp+fp+fn),('precision_percent',tp+fp),('recall_percent',tp+fn)):
    assert math.isclose(100*tp/d if d else 0,v[k][i],abs_tol=1e-8)
  assert math.isclose(sum(v['IoU_percent'])/4,v['mIoU_percent'],abs_tol=1e-8)
  return True
 except (AssertionError,TypeError,KeyError,ValueError):return False

def main():
 protocol=read('training_runs/semantic/protocol.json');digest=protocol['identity_SHA'] if protocol else None
 if protocol:
  material=protocol['identity'];p=material['protocol']
  check('fresh ImageNet W48 seed42 private20261008 fixed2048/batch2/40000',p['canvas']==2048 and p['global_batch']==2 and p['updates']==40000 and p['seed']==42 and p['private_seed']==20261008 and protocol['initialization'].startswith('fresh ImageNet') and hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()==digest)
  check('actual model/loss/data/training source and registered spec identity',all(sha(HERE/n)==d for n,d in material['code'].items()) and sha(HERE/'architecture_spec.json')==material['spec_SHA'] and sha(HERE/'screen_protocol.json')==material['screen_protocol_SHA'] and sha(HERE/'screen_gate.py')==material['screen_gate_code_SHA'])
  check('deployment without teacher/GT/TTA/H4',not p['inference_teacher'] and not p['student_TTA'] and not p['inference_online_H4'] and 'detach' in p['gradient_route'])
  metadata_path=ROOT/'cache_execution/server_receipts/metadata/data_identity.json'
  if metadata_path.exists():
   m=json.loads(metadata_path.read_text());sets={n:{x['sid'] for x in m['splits'][n]['samples']} for n in ('train','val','test')};hashes={n:{x['image_sha256'] for x in m['splits'][n]['samples']} for n in sets}
   check('audited custom student454/76/227 image-disjoint splits, no official-split claim',sha(metadata_path)==material['data_identity_SHA'] and tuple(len(sets[n]) for n in ('train','val','test'))==(454,76,227) and all(not(d[a]&d[b]) for d in (sets,hashes) for a,b in (('train','val'),('train','test'),('val','test'))))
  else:check('audited custom student splits',None)
 else:check('protocol and code identity',None)
 cpu=read('semantic_execution/CPU_ADAPTER_PASS.json');smoke=read('training_runs/semantic/GPU_SMOKE_PASS.json')
 expected={'zero_residual_exact_original_base_output','private_GT_KD_gradients_do_not_update_base','base_GT_gradients_do_not_update_private','all6_adapter_parts_finite_nonzero_gradients','single_SGD_two_groups4x_LR','exact_two_group_momentum_resume','empty_valid_mask_zero_loss','balanced_BCE_retains_positive_gradient_at_minus100','base_teacher_KL_updates_provider_not_private','base_KL_fullGT_group_denominators','separate_grad_clip5_resume_equivalence','actual_private_GT_retains_saturated_positive_gradient','balanced_private_KL_cannot_update_base','private_gradient_scaling_does_not_change_base_SGD_update','scale_preserving_concat_and_rawlogit_bypass'}
 check('pinned original-env CPU gradient/deployment/optimizer invariants',None if not cpu else cpu['status']=='PASS' and cpu['torch']=='2.0.1+cu117' and expected<=set(cpu['checks']) and all(sha(HERE/n)==d for n,d in cpu['source_SHA256'].items()))
 check('new identity real2048/b2 nonzero auxiliary GPU smoke',None if not smoke else smoke['status']=='PASS' and smoke['identity_SHA']==digest and smoke['canvas']==2048 and smoke['batch']==2 and smoke['fresh_initialization_verified'] and smoke['formal_updates']==0 and smoke['all_gradients_finite'] and all(smoke['last_smoke_step_private_nonzero_gradients'].values()))
 start=read('reports/semantic/SCREEN_STARTED.json');migration=read('reports/semantic/IDENTITY_MIGRATION.json');worker=read('semantic_execution/WORKER_RAM_PASS.json')
 check('new fresh registration has same identity and no task checkpoint',None if not start else start['identity_SHA']==digest and start['initialization']=='fresh ImageNet W48 seed42/private20261008' and not start['warm_diagnostic_checkpoint_used'] and protocol is not None and protocol.get('resume_from_own_update') is None)
 gate=read('training_runs/semantic/SCREEN_PASS.json');extension=read('reports/semantic/DEADLINE_EXTENSION_ACTIVATED.json');screenval=read('training_runs/semantic/val_04000.json')
 plan=json.loads((HERE/'screen_protocol.json').read_text());crit=plan['criteria']
 validator=read('reports/semantic/VALIDATOR_LOCK.json')
 check('pre-result validator/reference small-code identity sealed',None if not validator or not gate else validator['remote_sources_match'] and sha(ROOT/'sacd_micro_execution/gate.py')==validator['validation_helper_SHA'] and sha(ROOT/'sacd_execution/val_label_reference.json')==validator['val_reference_SHA'] and validator['locked_unix']<gate['time_unix'])
 check('4k screen sealed before actual results under renewed server authorization',None if not start or not gate else plan['registered_before_results'] and plan['screen_updates']==4000 and start['start_unix']<gate['time_unix'] and start['kind']=='ONE_FRESH_4K_SCREEN')
 check('4k screen PASS same current identity',None if not gate else gate['status']=='PASS' and gate['identity_SHA']==digest and gate['full_val_counts_verified'] and all(gate['decisions'].values()))
 if screenval:
  base=screenval['base_head_metrics']
  check('independently recomputed full76 screen numerical criteria',counts_valid(screenval,76) and counts_valid(base,76) and screenval['mIoU_percent']>=crit['4k_mIoU_at_least'] and screenval['mIoU_percent']-base['mIoU_percent']>=crit['final_minus_base_4k_pp_at_least'] and all(x>0 for x in screenval['IoU_percent']) and all(a>=b-crit['max_class_IoU_loss_final_vs_base_pp'] for a,b in zip(screenval['IoU_percent'],base['IoU_percent'])) and all(not(screenval['counts'][i][2]>crit['HE_MA_reject_if_FN_growth_over']*base['counts'][i][2] and screenval['counts'][i][1]>crit['HE_MA_requires_FP_ratio_when_FN_grows_at_most']*base['counts'][i][1]) for i in (1,3)))
 else:check('full76 fresh4k screen evidence',None)
 check('same fresh run conditional full40k continuation, renewed server availability',None if not extension or not gate else extension['status']=='ACTIVE' and extension['full40k_authorized'] and extension['same_fresh_run_continues'] and extension['screen_identity_SHA']==digest and extension['screen_pass_unix']==gate['time_unix'] and extension['server_reavailability_user_confirmed'])
 marker=read('training_runs/semantic/TRAINING_COMPLETE.json');outcome=read('reports/semantic/GPU_CAMPAIGN_OUTCOME.json');val=read('training_runs/semantic/val_40000.json');test=read('training_runs/semantic/test_40000.json');lock=read('reports/semantic/LOCKED_TEST_CHECKPOINT.json')
 check('fixed40000 durable own model/optimizer/RNG checkpoint loaded',None if not marker else marker['status']=='COMPLETE' and marker['completed']==40000 and marker['identity_SHA']==digest and marker['durable_checkpoint_loaded_and_verified'] and marker['primary_checkpoint']=='last.pt')
 check('fixed40k full76 counts/metrics',counts_valid(val,76))
 ref=json.loads((ROOT/'sacd_execution/val_label_reference.json').read_text())
 check('full76 GT counts equal independent label reference',None if not val else [x[0]+x[2] for x in val['counts']]==ref['ground_truth_positive_pixels'])
 val_pass=val['mIoU_percent']>45.32183122738841 if val else None
 if val_pass:
  check('fixed40k checkpoint locked before single test evaluation',None if not lock or not test else lock['completed']==40000 and lock['identity_SHA']==digest and not lock['test_used_for_selection'] and lock['selected_by']=='pre-registered fixed40k endpoint' and lock['evaluation_code_SHA256']==sha(HERE/'test_eval.py') and test['checkpoint_lock']==lock and lock['locked_before_test_unix']<test['completed_unix'])
  check('independent custom227 test counts/metrics',counts_valid(test,227))
  check('test actual locked40k deployment without labels/teacher/TTA/H4 input or test selection',None if not test else test['identity_SHA']==digest and test['completed']==40000 and test['split']=='test' and not test['inference_teacher_or_GT'] and not test['TTA'] and not test['online_H4'] and not test['test_for_selection'])
 check('terminal own controller outcome, serverON and external diagnostics untouched',None if not outcome else outcome['status']=='COMPLETE' and outcome['server_kept_running'] and not outcome['shutdown_command_issued'] and not outcome['external_diagnostics_controlled'])
 snapshot=HERE/'latest_snapshot.json';runtime=json.loads(snapshot.read_text())['runtime'] if snapshot.exists() else None
 terminal=runtime and outcome and runtime['checked_at_unix']>outcome['time_unix']
 check('actual post-completion SSH online and own training/controller/RAM ended',None if not terminal else runtime['server_online'] and not runtime['shutdown_command_issued'] and all(not x['alive'] for x in runtime['own_process_checks'].values()) and not runtime['own_RAM']['alive'] and runtime.get('own_session_live_processes')==[])
 test_pass=test['mIoU_percent']>45.860383832668 if test else None
 failed=[x for x in checks if x['status']=='FAIL'];pending=[x for x in checks if x['status']=='PENDING']
 status='CONTRADICTED' if failed else 'INCOMPLETE' if pending else 'GOAL_VERIFIED' if val_pass and test_pass else 'GOAL_NOT_ACHIEVED'
 result={'status':status,'goal_achieved':status=='GOAL_VERIFIED','checks':checks,'fixed40k_val_mIoU':val['mIoU_percent'] if val else None,'fixed40k_test_mIoU':test['mIoU_percent'] if test else None,'val_target_exceeded':val_pass,'test_target_exceeded':test_pass,'audited_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'checkpoint_download':False,'data_cache_hash_rescan':False,'test_for_tuning':False,'scope':'audited custom student454/76/227; no official teacher-split independence or never-previously-used dataset claim'}
 (HERE/'FINAL_OUTCOME_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'status':status,'goal_achieved':result['goal_achieved'],'failed':[x['requirement'] for x in failed],'pending':[x['requirement'] for x in pending]}))
if __name__=='__main__':main()
