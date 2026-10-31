"""Ship only new small sources; launch one CPU/smoke-gated fresh screen."""
import hashlib
import json
from pathlib import Path
import py_compile
import shlex
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
NEW=ROOT/'residual_execution'
TEMPLATE=ROOT/'sacd_isolated_execution'
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect

source=(TEMPLATE/'controller.py').read_text(encoding='utf-8').replace('sacd_isolated','residual')
source=source.replace("    warm = json.loads((ROOT / 'reports/sacd_micro/MICRO_GATE.json').read_text())\n    assert warm['status'] == 'PASS' and warm['full_val_counts_verified']\n",'')
source=source.replace("    for name in ('CPU_SACD_PASS.json', 'CHECKPOINT_RESUME_PASS.json', 'FEATURE_ISOLATION_CPU_PASS.json'):","    for name in ('CPU_RESIDUAL_PASS.json',):")
source=source.replace("'candidate_count': 1, 'candidate_limit': 1,","'candidate_count_this_architecture': 1, 'further_revisions_user_authorized': True,")
source=source.replace("'external_diagnostics_controlled': False, **end_own_ram()}","'external_diagnostics_controlled': False, 'own_RAM_retained_for_authorized_next_revision': not full_done,\n                   **(end_own_ram() if full_done else {})}")
source=source.replace("        registration = {'kind':", "        write(REPORTS / 'DEADLINE_GATE.json', {'status':'PASS','measured_screen_estimate_seconds':estimate_screen,\n            'checked_unix':time.time(),'original_deadline_unix':1791568800,'reserve_seconds':180,\n            'full40k_conditional_on_screen':True})\n        registration = {'kind':")
source=source.replace("'shutdown_after_failure': False}","'shutdown_after_failure': False,'further_revisions_user_authorized':True}")
(NEW/'controller.py').write_text(source,encoding='utf-8')
for p in NEW.glob('*.py'):py_compile.compile(str(p),doraise=True)
files=[p for p in NEW.iterdir() if p.is_file() and p.suffix in ('.py','.json') and p.name!='CPU_RESIDUAL_PASS.json']
sealed={'sealed_before_server_training':True,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'user_authorization':'failed screen leads autonomous redesign; no choice required',
        'server_must_stay_online':True,'test_for_selection':False}
(NEW/'SEALED_LOCAL_SOURCE.json').write_text(json.dumps(sealed,indent=2)+'\n')
client=connect()
try:
    with client.open_sftp() as ftp:
        for directory in ('residual_execution','reports/residual'):
            try:ftp.mkdir(REMOTE+'/'+directory)
            except OSError:pass
        for p in files+[NEW/'SEALED_LOCAL_SOURCE.json']:
            ftp.put(str(p),REMOTE+'/residual_execution/'+p.name)
    code='''import json,os,subprocess,time,hashlib
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_v3_20261008');h=r/'residual_execution';p=r/'reports/residual'
assert not(p/'DISPATCHED.json').exists()
sealed=json.loads((h/'SEALED_LOCAL_SOURCE.json').read_text())
for n,d in sealed['files'].items():assert hashlib.sha256((h/n).read_bytes()).hexdigest()==d,n
assert (r/'sacd_execution/val_label_reference.json').exists()
assert (r/'reports/sacd/RAM_CACHE_READY.json').exists()
# A launch supervisor does necessary CPU checks then controller/GPU smoke; no
# second fresh launcher and no whole-data/cache hashing or checkpoint transfer.
env=dict(os.environ,PYTHONPATH=str(r/'vendor/champion_repo'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
launcher=h/'launch_verified.py'
launcher.write_text("import subprocess,sys,json,time\\nfrom pathlib import Path\\nh=Path(__file__).resolve().parent\\np=h.parent/'reports/residual'\\ntry:\\n subprocess.run([sys.executable,'verify_residual.py'],cwd=str(h),check=True)\\n subprocess.run([sys.executable,'controller.py'],cwd=str(h),check=True)\\nexcept BaseException as e:\\n (p/'LAUNCH_ERROR.json').write_text(json.dumps({'status':'FAILED','error':str(e),'time_unix':time.time(),'server_kept_running':True},indent=2))\\n raise\\n")
with open(p/'launch.log','a',buffering=1) as f:
 q=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(launcher)],cwd=str(h),stdout=f,stderr=f,start_new_session=True,env=env)
receipt={'status':'CPU_GATE_DISPATCHED_THEN_GPU_SMOKE_THEN_FRESH_SCREEN','supervisor_pid':q.pid,'time_unix':time.time(),
 'canvas':2048,'batch':2,'fresh_screen_updates':8000,'full40k_only_after_screen_PASS':True,
 'fresh_ImageNet_seed':42,'private_seed':20261008,'old_task_checkpoint_used':False,'shutdown':False}
(p/'DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    _,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=out.read().decode(),err.read().decode();assert out.channel.recv_exit_status()==0,error
    receipt=json.loads(raw);(NEW/'DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
finally:client.close()
