"""Read own terminal checkpoint summaries on CPU, release own memfd cache only."""
import json
from pathlib import Path
import shlex
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect

code=r'''import json,os,time,subprocess
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_v3_20261008');p=r/'reports/residual_normalization'
outcome=json.loads((p/'OUTCOME.json').read_text());assert outcome['status']=='COMPLETE'
assert not outcome['deadline_extension'] and not outcome['automatic_fresh_training']
own_members=[]
for entry in Path('/proc').iterdir():
 if not entry.name.isdigit():continue
 try:
  parts=(entry/'stat').read_text().split(') ',1)[1].split()
  if int(parts[2])!=37655 or int(parts[3])!=37655:continue
  cmd=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if cmd:own_members.append({'pid':int(entry.name),'cmdline':cmd,'state':parts[0]})
 except OSError:continue
assert not own_members,'Own controller/train/worker session remains active'
for pid in (37655,37726):
 f=Path('/proc/%d/cmdline'%pid);assert not f.exists() or not f.read_bytes()
# Only these two own durable warm-diagnostic checkpoints; no CUDA or test access.
import torch
torch.set_num_threads(2)
work=r/'training_runs/residual_normalization'
control=torch.load(work/'control/probe_last.pt',map_location='cpu')
variant=torch.load(work/'gt_stratified/probe_last.pt',map_location='cpu')
assert control['parent_completed']==variant['parent_completed']==8000
assert control['probe_updates']==variant['probe_updates']==500
summary={}
for prefix in ('base.','private.'):
 names=[n for n in control['model'] if n.startswith(prefix)]
 assert names and all(n in variant['model'] for n in names)
 mismatches=[];maximum=0.;different=0;number=0
 for name in names:
  a,b=control['model'][name],variant['model'][name]
  assert a.shape==b.shape and a.dtype==b.dtype
  number+=a.numel()
  if not torch.equal(a,b):
   delta=(a-b).abs();maximum=max(maximum,float(delta.max()))
   different+=int((a!=b).sum());mismatches.append(name)
 summary[prefix]={'tensor_count':len(names),'differing_tensors':len(mismatches),
   'different_elements':different,'total_elements':number,'maximum_absolute_difference':maximum,
   'first10_differing_names':mismatches[:10]}
np_a,np_b=control['rng']['numpy'],variant['rng']['numpy']
rng={'python_equal':control['rng']['python']==variant['rng']['python'],
 'numpy_equal':np_a[0]==np_b[0] and (np_a[1]==np_b[1]).all().item() and np_a[2:]==np_b[2:],
 'torch_CPU_equal':torch.equal(control['rng']['torch'],variant['rng']['torch']),
 'torch_CUDA_state_bytes_equal_without_CUDA_execution':len(control['rng']['cuda'])==len(variant['rng']['cuda']) and all(torch.equal(a,b) for a,b in zip(control['rng']['cuda'],variant['rng']['cuda']))}
comparison={'status':'CPU_CHECKPOINT_COMPARISON_COMPLETE','parent_completed':8000,'probe_updates':500,
 'model_state_comparison':summary,'final_RNG_comparison':rng,'GPU_execution':False,
 'checkpoint_download':False,'test_accessed':False,
 'root_cause_of_provider_divergence_proven':False}
(p/'CPU_STATE_COMPARISON.json').write_text(json.dumps(comparison,indent=2)+'\n')
del control,variant
ready=json.loads((r/'reports/sacd/RAM_CACHE_READY.json').read_text());pid=ready['pid']
assert pid==25079
path=Path('/proc/%d/cmdline'%pid)
if path.exists() and path.read_bytes():
 cmd=path.read_bytes();assert str(r/'sacd_execution/ram_cache_owner.py').encode() in cmd
 # Its own loop exits cleanly and closes memfds; teacher disk files untouched.
 (r/'reports/sacd/STOP_RAM_OWNER.enabled').write_text('Own experiment and paired diagnostic ended; original deadline passed\n')
 for _ in range(12):
  if not path.exists() or not path.read_bytes():break
  time.sleep(.5)
assert not path.exists() or not path.read_bytes(),'Own RAM owner not yet ended'
receipt={'status':'COMPLETE','checked_at_unix':time.time(),'server_online':True,
 'own_controller_training_workers_session_empty':True,'own_session_id':37655,
 'own_RAM_pid':pid,'own_RAM_process_ended':True,'released_owned_memfd_bytes':ready['bytes'],
 'teacher_disk_cache_and_valid_checkpoints_preserved':True,'checkpoint_download':False,
 'shutdown_command_issued':False,'external_diagnostics_controlled':False,
 'GPU_status':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv'],universal_newlines=True),
 'goal_achieved':False,'no_new_GPU_work_after_deadline':True,
 'cpu_state_comparison':comparison}
(p/'FINAL_TERMINAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
os.sync();print(json.dumps(receipt))
'''
client=connect()
try:
    _,output,error=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,err=output.read().decode(),error.read().decode();assert output.channel.recv_exit_status()==0,err
    receipt=json.loads(raw)
    (HERE/'FINAL_TERMINAL_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    archive=ROOT/'cache_execution/server_receipts/residual_normalization/reports/residual_normalization'
    archive.mkdir(parents=True,exist_ok=True)
    for name in ('FINAL_TERMINAL_RECEIPT.json','CPU_STATE_COMPARISON.json'):
        with client.open_sftp() as ftp:
            with ftp.open('/root/autodl-tmp/HRDecoder_v3_20261008/reports/residual_normalization/'+name,'rb') as file:
                (archive/name).write_bytes(file.read())
    print(json.dumps(receipt))
finally:client.close()
