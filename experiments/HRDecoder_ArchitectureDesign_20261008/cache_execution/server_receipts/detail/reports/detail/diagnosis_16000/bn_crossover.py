"""Registered BN-buffer crossover on immutable12k/21k checkpoints; no SGD/test."""
import os,time,json,sys,fcntl,hashlib
from pathlib import Path
import torch
ROOT=Path('/root/autodl-tmp/HRDecoder_v3_20261008');HERE=ROOT/'detail_execution';OUT=ROOT/'reports/detail/diagnosis_16000';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(HERE))
import train as t
from baseline import build_baseline,install_legacy_bf16_interpolation
from model import SACDStudent
identity='046abab773f7cb7a9e4f0a0f9cc44c4ad32d48431b971f22720dc8bd7936168a'
def write(name,value):
 p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
with open(ROOT/'reports/gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;install_legacy_bf16_interpolation()
 old=torch.load(ROOT/'training_runs/detail/best_val.pt',map_location='cpu')
 now=torch.load(ROOT/'training_runs/detail/aborted_parent_21000.pt',map_location='cpu')
 assert old['completed']==12000 and now['completed']==21000 and old['identity_SHA']==now['identity_SHA']==identity
 initial,_=build_baseline(t.REPO,t.W48,verified_weight_sha='d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960')
 model=SACDStudent(initial).cuda().eval();t.WORK=OUT/'validation_runtime';t.WORK.mkdir(exist_ok=True)
 states={12000:old['model'],21000:now['model']};bn_names=[k for k in states[12000] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
 assert len(bn_names)==921
 reference=json.loads((ROOT/'sacd_execution/val_label_reference.json').read_text())
 results=[]
 for weight_step,bn_step in [(12000,12000),(21000,21000),(21000,12000),(12000,21000)]:
  name='W%d_BN%d'%(weight_step,bn_step)
  write('CAUSAL_DIAGNOSTIC_PROGRESS.json',{'status':'RUNNING','pid':os.getpid(),'arm':name,'time_unix':time.time(),'formal_updates':0,'test_accessed':False})
  state=dict(states[weight_step]);state.update({k:states[bn_step][k] for k in bn_names});model.load_state_dict(state,strict=True);model.eval()
  torch.cuda.synchronize();start=time.time();value=t.evaluate(model)
  for c,row in enumerate(value['counts']):assert row[0]+row[2]==reference['ground_truth_positive_pixels'][c]
  value.update(arm=name,weight_step=weight_step,bn_buffer_step=bn_step,identity_SHA=identity,diagnostic_only=True,formal_updates=0,test_accessed=False,swapped_buffers_only=True,affine_parameters_follow_weights=True,time_unix=time.time(),wall_seconds=time.time()-start)
  write(name+'.json',value);results.append(value)
  print(json.dumps({'arm':name,'mIoU':value['mIoU_percent'],'IoU':value['IoU_percent'],'base':value['base_head_metrics']['mIoU_percent'],'seconds':value['wall_seconds']}),flush=True)
 matrix={q['arm']:q for q in results};repro=abs(matrix['W12000_BN12000']['mIoU_percent']-36.73584859847754)
 gain=matrix['W21000_BN12000']['mIoU_percent']-matrix['W21000_BN21000']['mIoU_percent'];damage=matrix['W12000_BN21000']['mIoU_percent']-matrix['W12000_BN12000']['mIoU_percent']
 verdict={'status':'COMPLETE','pid':os.getpid(),'time_unix':time.time(),'identity_SHA':identity,'counts_verified':True,'formal_updates':0,'test_accessed':False,'registered_before_interventions':True,'reference12k_reproduction_error_pp':repro,'reference_reproduction_pass':repro<=.05,'BN12_effect_on21k_pp':gain,'BN21_effect_on12k_pp':damage,'strong_buffer_mismatch_support':bool(repro<=.05 and gain>=10 and damage<=-10),'results':results,'training_or_checkpoint_mutated':False,'exact16k_cause_proven':False,'scope':'BN buffers at12k/21k only; isolates current checkpoint BN effect, not exact16k weights; does not establish final fresh accuracy'}
 write('BN_CROSSOVER_COMPLETE.json',verdict);write('CAUSAL_DIAGNOSTIC_PROGRESS.json',{'status':'COMPLETE','time_unix':time.time(),'formal_updates':0,'test_accessed':False})
 print(json.dumps({k:v for k,v in verdict.items() if k!='results'}),flush=True)
