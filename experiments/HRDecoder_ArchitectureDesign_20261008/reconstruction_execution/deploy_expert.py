"""New candidate CPU+full GPU smoke, then one fresh2k/4k screened run."""
import hashlib
import json
from pathlib import Path
import py_compile
import shlex
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;NEW=ROOT/'expert_execution'
OLD=ROOT/'residual_execution';REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect

s=(OLD/'controller.py').read_text(encoding='utf-8').replace('residual','expert')
s=s.replace('CPU_RESIDUAL_PASS','CPU_EXPERT_PASS').replace('8000','4000').replace('8K_SCREEN','4K_SCREEN').replace('fresh8k','fresh4k')
(NEW/'controller.py').write_text(s,encoding='utf-8')
s=(NEW/'train.py').read_text(encoding='utf-8')
needle="'ImageNet_W48_SHA':old['ImageNet_W48_SHA']"
s=s.replace(needle,"'validation_helper_SHA':file_sha(ROOT/'sacd_micro_execution/gate.py'),"+needle)
(NEW/'train.py').write_text(s,encoding='utf-8')
for p in NEW.glob('*.py'):py_compile.compile(str(p),doraise=True)
files=[p for p in NEW.iterdir() if p.is_file() and p.suffix in ('.py','.json') and p.name!='CPU_EXPERT_PASS.json']
sealed={'sealed_before_server_training':True,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
 'user_authorization':'Rejected previous screen leads autonomous architecture revision, without asking for choices',
 'previous_residual_screen_rejection_remains_unchanged':True,'server_must_stay_online':True,'test_for_selection':False}
(NEW/'SEALED_LOCAL_SOURCE.json').write_text(json.dumps(sealed,indent=2)+'\n')
c=connect()
try:
    with c.open_sftp() as ftp:
        for d in ('expert_execution','reports/expert'):
            try:ftp.mkdir(REMOTE+'/'+d)
            except OSError:pass
        for p in files+[NEW/'SEALED_LOCAL_SOURCE.json']:ftp.put(str(p),REMOTE+'/expert_execution/'+p.name)
    code='''import hashlib,json,os,subprocess,time
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_v3_20261008');h=r/'expert_execution';p=r/'reports/expert'
assert not(p/'DISPATCHED.json').exists()
rejection=json.loads((r/'training_runs/residual/SCREEN_REJECTED.json').read_text())
assert rejection['status']=='REJECTED' and not rejection['decisions']['private_net_gain_over_base']
for pid in (33845,33638):assert not Path('/proc/%d'%pid).exists(),'Previous own run not terminal'
sealed=json.loads((h/'SEALED_LOCAL_SOURCE.json').read_text())
for n,d in sealed['files'].items():assert hashlib.sha256((h/n).read_bytes()).hexdigest()==d,n
assert (r/'sacd_execution/val_label_reference.json').exists()
launcher=h/'launch_verified.py'
launcher.write_text("import subprocess,sys,json,time\\nfrom pathlib import Path\\nh=Path(__file__).resolve().parent\\np=h.parent/'reports/expert'\\ntry:\\n subprocess.run([sys.executable,'verify_expert.py'],cwd=str(h),check=True)\\n subprocess.run([sys.executable,'verify_screen_fixture.py'],cwd=str(h),check=True)\\n subprocess.run([sys.executable,'controller.py'],cwd=str(h),check=True)\\nexcept BaseException as e:\\n (p/'LAUNCH_ERROR.json').write_text(json.dumps({'status':'FAILED','error':str(e),'time_unix':time.time(),'server_kept_running':True},indent=2))\\n raise\\n")
env=dict(os.environ,PYTHONPATH=str(r/'vendor/champion_repo'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
with open(p/'launch.log','a',buffering=1) as f:
 q=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(launcher)],cwd=str(h),stdout=f,stderr=f,start_new_session=True,env=env)
receipt={'status':'CPU_GATE_THEN_GPU_SMOKE_THEN_NEW_FRESH_SCREEN','supervisor_pid':q.pid,'time_unix':time.time(),
 'canvas':2048,'batch':2,'fresh_screen_updates':4000,'extra_full76_val_probe':2000,
 'full40k_only_after_screen_PASS':True,'fresh_ImageNet_seed':42,'private_seed':20261008,
 'old_task_checkpoint_used':False,'shutdown':False,'prior_rejection_not_overridden':True}
(p/'DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    _,o,e=c.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=o.read().decode(),e.read().decode();assert o.channel.recv_exit_status()==0,error
    receipt=json.loads(raw);(NEW/'DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
finally:c.close()
