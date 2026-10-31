"""One source-only worker RAM repair at own durable checkpoint, no fresh restart."""
import json,os,signal,subprocess,time,traceback,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;REPORTS=ROOT/'reports/adapter';WORK=ROOT/'training_runs/adapter'
PYTHON='/root/miniconda3/bin/python'
OLD_ID='7527da02a7fad6366a190b366f492f92ec38f67a7a9e93f51570abf3bc1113f2'
OLD_CONTROLLER=42285;OLD_TRAIN=42975

def write(name,value):
 p=REPORTS/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
def cmd(pid):return Path('/proc/%d/cmdline'%pid).read_bytes().replace(b'\0',b' ').decode()
def main():
 assert cmd(OLD_CONTROLLER).find(str(HERE/'controller.py'))>=0
 assert 'train.py --train' in cmd(OLD_TRAIN)
 assert Path('/proc/%d/cwd'%OLD_TRAIN).resolve()==HERE
 assert os.getsid(OLD_TRAIN)==OLD_CONTROLLER
 proof=json.loads((HERE/'WORKER_RAM_PASS.json').read_text());assert proof['status']=='PASS'
 write('RAM_REPAIR_PENDING.json',{'status':'WAITING_OWN_DURABLE_CHECKPOINT','controller_pid':os.getpid(),'preserve_RAM_pid':39434,'old_training_continues':True,'time_unix':time.time()})
 deadline=time.time()+1200
 while time.time()<deadline:
  if not Path('/proc/%d'%OLD_TRAIN).exists():raise RuntimeError('Own original training ended before repair checkpoint')
  progress=json.loads((WORK/'progress.json').read_text())
  if progress.get('saved_update',0)>=1000 and (WORK/'last.pt').exists():break
  time.sleep(1)
 else:raise RuntimeError('Own durable checkpoint did not appear')
 # Own session only, after verified source/PID/cwd. Cache has a different session.
 assert os.getsid(OLD_CONTROLLER)==OLD_CONTROLLER and os.getsid(OLD_TRAIN)==OLD_CONTROLLER
 os.killpg(OLD_CONTROLLER,signal.SIGTERM)
 for _ in range(15):
  live=[]
  for path in Path('/proc').iterdir():
   if path.name.isdigit():
    try:
     if os.getsid(int(path.name))==OLD_CONTROLLER and (path/'stat').read_text().split(')')[1].strip().split()[0]!='Z':live.append(int(path.name))
    except (OSError,ProcessLookupError):pass
  if not live:break
  time.sleep(1)
 if live:os.killpg(OLD_CONTROLLER,signal.SIGKILL)
 assert 'sacd_execution/ram_cache_owner.py' in cmd(39434)
 import torch
 checkpoint=torch.load(WORK/'last.pt',map_location='cpu')
 assert checkpoint['identity_SHA']==OLD_ID and checkpoint['completed']>=1000
 assert checkpoint['optimizer']['state'] and all(k in checkpoint['rng'] for k in ('python','numpy','torch','cuda'))
 completed=checkpoint['completed'];del checkpoint
 immutable=WORK/('before_worker_RAM_resume_%05d.pt'%completed)
 assert not immutable.exists();os.link(WORK/'last.pt',immutable)
 archive=REPORTS/'source_before_worker_RAM';archive.mkdir(exist_ok=False)
 for n in ('train.py','loader.py'):shutil.copy2(HERE/n,archive/n)
 old_protocol=json.loads((WORK/'protocol.json').read_text())
 for p in (WORK/'protocol.json',WORK/'GPU_SMOKE_PASS.json',REPORTS/'GPU_CAMPAIGN_RUNNING.json'):
  if p.exists():shutil.copy2(p,archive/p.name)
 for src,dest in (('loader_ram_patch.py','loader.py'),('train_ram_patch.py','train.py')):
  shutil.copyfile(HERE/src,HERE/(dest+'.installing'));os.replace(HERE/(dest+'.installing'),HERE/dest)
 env=dict(os.environ,PYTHONPATH=str(ROOT/'vendor/champion_repo'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
 identity=json.loads(subprocess.check_output([PYTHON,'-c','import train,json; m,d=train.identity();print(json.dumps({"identity":m,"identity_SHA":d}))'],cwd=str(HERE),env=env).decode())
 before=old_protocol['identity'];after=identity['identity']
 changed=[n for n in before['code'] if before['code'][n]!=after['code'][n]]
 assert set(changed)=={'train.py','loader.py'}
 assert {k:v for k,v in before.items() if k!='code'}=={k:v for k,v in after.items() if k!='code'}
 write('IDENTITY_MIGRATION.json',{'status':'APPROVED_SOURCE_ONLY_PERFORMANCE_REPAIR','from_identity_SHA':OLD_ID,'to_identity_SHA':identity['identity_SHA'],'resume_completed':completed,'scope':'spawn_worker_RAM_initialization_only','model_loss_data_schedule_unchanged':True,'worker_RAM_bitwise_test_PASS':True,'changed_code_files':changed,'before_code_SHA':before['code'],'after_code_SHA':after['code'],'immutable_checkpoint':str(immutable),'fresh_initialization_and_accepted_update_path_preserved':True,'old_last_logged_completed':progress.get('completed'),'uncommitted_updates_may_be_replayed':True,'test_accessed':False,'time_unix':time.time()})
 write('RAM_REPAIR_PENDING.json',{'status':'MIGRATION_PREPARED_REAL_GPU_GATE_AND_OWN_RESUME_NEXT','resume_completed':completed,'identity_SHA':identity['identity_SHA'],'time_unix':time.time()})
 os.execve(PYTHON,[PYTHON,'-u',str(HERE/'resume_controller.py')],env)
if __name__=='__main__':
 try:main()
 except BaseException as error:
  write('RAM_REPAIR_ERROR.json',{'error':str(error),'traceback':traceback.format_exc(),'server_kept_running':True,'external_processes_controlled':False,'time_unix':time.time()});raise
