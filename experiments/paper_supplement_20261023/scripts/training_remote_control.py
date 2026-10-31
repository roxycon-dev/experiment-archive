"""Durable key-authenticated control of only the two user-authorized experiment directories."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import paramiko

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/root/autodl-tmp/paper_supplement_20261023_run/paper_supplement_20261023'
LAN='/home/studentuser001/paper_supplement_20261023'

def connect():
    client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('connect.nmb1.seetacloud.com',port=40411,username='root',
                   key_filename='C:/Users/GuanXingRen/.ssh/codex_hrdecoder_next_20261003',
                   look_for_keys=False,allow_agent=False,timeout=15,auth_timeout=15)
    pin=base64.b64encode(hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).digest()).decode().rstrip('=')
    assert pin=='yswdgtshuc+hy/m0jR576i7Nu0Es7lZ71jIrFKXnkE0'
    return client

def execute(client,command,target='root',timeout=45):
    if target=='59':
        command=shlex.join(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
            '-i','/root/.ssh/codex_supp_to59_20261023','-p','40560','studentuser001@127.0.0.1',command])
    _,out,err=client.exec_command(command,timeout=timeout)
    stdout=out.read().decode('utf-8','replace');stderr=err.read().decode('utf-8','replace')
    return {'exit_code':out.channel.recv_exit_status(),'stdout':stdout,'stderr':stderr}

def snapshot(client):
    code="""
import json,os,shutil,subprocess
from pathlib import Path
root=Path(ROOT_VALUE)
def read(p):return json.loads(p.read_text()) if p.exists() else None
result={'queue':read(root/'training_queue_progress.json'),'ETA_at_last_job_boundary':read(root/'training_ETA.json'),'runs':{},
 'endpoint_publisher':read(root/'paired_endpoint_publisher_status.json'),'disk':shutil.disk_usage(root)._asdict(),
 'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)}
for arm in ['C','S']:
 for seed in [2024,3407]:
  run=root/f'training_v1/runs/{arm}_seed{seed}'
  result['runs'][f'{arm}_seed{seed}']={'status':read(run/'status.json'),'completion':read(run/'completion.json')}
benchmark=result['ETA_at_last_job_boundary'] or {}
if 'C_seconds_per_update' in benchmark:
 result['estimated_remaining_adaptation_training_hours_now']=sum(max(0,20000-((v['status'] or {}).get('completed',0)))*benchmark[name[0]+'_seconds_per_update'] for name,v in result['runs'].items())/3600
print(json.dumps(result))
""".replace('ROOT_VALUE',repr(REMOTE))
    command='/root/miniconda3/envs/hrdecoder_q_20261003/bin/python -c '+shlex.quote(code)
    a=execute(client,command)
    assert a['exit_code']==0,a
    code59="""
import json,re,os,shutil,subprocess
from pathlib import Path
r=Path(ROOT_VALUE)
def read(p):return json.loads(p.read_text()) if p.exists() else None
p=r/'logs/baseline_train.log';text=p.read_text(errors='replace')[-128000:] if p.exists() else ''
iterations=re.findall(r'Iter \\[(\\d+)/40000\\].*?time: ([0-9.]+)',text)
evaluation=r/'evaluation_workspace/paper_supplement_20261023'
cache={}
for cohort in ('canary','ddr','idrid'):
 for branch in ('A','U','P0'):
  p=evaluation/'results/common_cache59'/cohort/branch
  cache[cohort+'_'+branch]={'progress':read(p/'progress.json'),'completion':read(p/'completion.json')}
pairs={}
for arm in ('C','S'):
 for seed in (42,2024,3407):
  p=evaluation/'results/paired_seed59'/(arm+'_seed'+str(seed))
  pairs[p.name]={'progress':read(p/'status.json'),'completion':read(p/'completion.json'),'failure':read(p/'failure.json')}
audit=read(evaluation/'evidence/shared_cache59_integrity.json')
if audit:audit.pop('rows',None)
baseline_evaluation=evaluation/'results/baseline_retrained59'
v2log=r/'logs/baseline_v2_train.log';v2text=v2log.read_text(errors='replace')[-128000:] if v2log.exists() else ''
v2iterations=re.findall(r'Iter \\[(\\d+)/40000\\].*?time: ([0-9.]+)',v2text)
v2eval=evaluation/'results/baseline_reconstructed_v2_59'
print(json.dumps({'queue':read(r/'logs/baseline_queue_status.json'),'environment':read(r/'logs/environment_ready.json'),
 'data':read(r/'logs/data_acceptance.json'),'last_iteration':int(iterations[-1][0]) if iterations else None,
 'recent_seconds_per_iteration':[float(x[1]) for x in iterations[-10:]],'log_tail':text[-2200:],
 'disk':shutil.disk_usage(r)._asdict(),
 'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
 'common_cache_queue_original':read(r/'logs/common_cache59_queue.json'),'common_U_recovery':read(r/'logs/common_U_recovery.json'),
 'evaluation_environment':read(r/'logs/evaluation_environment.json'),'paired_queue':read(r/'logs/paired_evaluation59_queue.json'),
 'common_cache':cache,'shared_cache_integrity':audit,'paired_evaluation':pairs,
 'baseline_v2':{'queue':read(r/'logs/baseline_v2_queue_status.json'),
 'last_iteration':int(v2iterations[-1][0]) if v2iterations else None,
 'recent_seconds_per_iteration':[float(x[1]) for x in v2iterations[-10:]],'log_tail':v2text[-1800:],
 'status':read(v2eval/'status.json'),'completion':read(v2eval/'completion.json'),'failure':read(v2eval/'failure.json')},
 'baseline_evaluation':{'status':read(baseline_evaluation/'status.json'),'completion':read(baseline_evaluation/'completion.json'),'failure':read(baseline_evaluation/'failure.json')}}))
""".replace('ROOT_VALUE',repr(LAN))
    b=execute(client,'/usr/bin/python3 -c '+shlex.quote(code59),'59')
    result={'adaptation':json.loads(a['stdout']),'baseline':json.loads(b['stdout']) if b['exit_code']==0 else b}
    (ROOT/'results/training_live_status.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--target',choices=['root','59'],default='root')
    parser.add_argument('--command-file',type=Path);parser.add_argument('--get-remote');parser.add_argument('--local',type=Path)
    args=parser.parse_args()
    with connect() as client:
        if args.command_file:
            print(json.dumps(execute(client,args.command_file.read_text(encoding='utf-8'),args.target),ensure_ascii=False))
        elif args.get_remote:
            assert args.local and args.target=='root'
            args.local.parent.mkdir(parents=True,exist_ok=True)
            with client.open_sftp() as s:s.get(args.get_remote,str(args.local))
            print(json.dumps({'downloaded':str(args.local)}))
        else:print(json.dumps(snapshot(client),ensure_ascii=False))

if __name__=='__main__':main()
