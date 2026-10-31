"""BN batch-stat diagnostic only; no calibration, buffer updates, SGD or test."""
import sys,json,time,fcntl,types
from pathlib import Path
import torch
ROOT=Path('/root/autodl-tmp/HRDecoder_v3_20261008');HERE=ROOT/'detail_execution';OUT=ROOT/'reports/detail/diagnosis_16000';sys.path.insert(0,str(HERE))
import train as t
from baseline import build_baseline,install_legacy_bf16_interpolation
from model import SACDStudent
with open(ROOT/'reports/gpu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;install_legacy_bf16_interpolation()
 ck=torch.load(ROOT/'training_runs/detail/aborted_parent_21000.pt',map_location='cpu');assert ck['completed']==21000
 base,_=build_baseline(t.REPO,t.W48,verified_weight_sha='d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960')
 model=SACDStudent(base).cuda().eval();model.load_state_dict(ck['model'],strict=True)
 norms=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
 before=[(m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone()) for m in norms]
 original_train=model.train
 def train_with_batch_bn(self,mode=True):
  original_train(mode)
  for m in norms:m.train(True);m.track_running_stats=False
  return self
 model.train=types.MethodType(train_with_batch_bn,model)
 t.WORK=OUT/'batch_stat_validation_runtime';t.WORK.mkdir(exist_ok=True)
 start=time.time();value=t.evaluate(model)
 unchanged=all(torch.equal(m.running_mean,a) and torch.equal(m.running_var,b) and torch.equal(m.num_batches_tracked,c) for m,(a,b,c) in zip(norms,before));assert unchanged
 reference=json.loads((ROOT/'sacd_execution/val_label_reference.json').read_text())
 for c,row in enumerate(value['counts']):assert row[0]+row[2]==reference['ground_truth_positive_pixels'][c]
 value.update(status='COMPLETE',diagnostic_only=True,weight_step=21000,BN='per-inference-image batch statistics with no running buffer updates; dropout stays eval',BN_buffers_unchanged=unchanged,formal_updates=0,test_accessed=False,not_deployment_model=True,time_unix=time.time(),seconds=time.time()-start,scope='Detect train/eval BN dependence, not valid final score or proof of generalization')
 (OUT/'BN_BATCH_STAT_DIAGNOSTIC.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'status':'COMPLETE','mIoU':value['mIoU_percent'],'IoU':value['IoU_percent'],'base':value['base_head_metrics']['mIoU_percent'],'seconds':value['seconds'],'BN_buffers_unchanged':unchanged}),flush=True)
