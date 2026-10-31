"""User-authorized shutdown only after reviewed completion and verified local backup."""
import argparse
import json
from pathlib import Path
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,LOCAL,connect,remote_exec,write_json,file_sha

def check_local(ready):
    assert ready['status']=='REVIEWED_COMPLETE', 'Final experiment review is required'
    assert ready['user_authorized_shutdown'] is True
    assert ready['report'] and Path(ready['report']).is_file()
    for rel,h in ready['required_files'].items():
        p=LOCAL/rel
        assert p.is_file() and file_sha(p)==h,'Missing/mismatched backup '+rel
    for item in ready['historical_assets']:
        assert file_sha(item['path'])==item['sha256'],'Retained historical asset mismatch'
    manifest=json.loads((ROOT/'reports/mirror_manifest.json').read_text(encoding='utf-8-sig'))
    for rel,h in ready['required_probability_files'].items():
        p=LOCAL/rel
        assert p.is_file() and p.stat().st_size==manifest[rel]['size']
        assert manifest[rel]['sha256']==h,'Probability backup not verified '+rel

GUARD=r'''
import pathlib,json,hashlib,os,subprocess,time
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Next_20261003');request=json.loads(PAYLOAD)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
completion=r/'reports/campaign_completion.json'
assert sha(completion)==request['completion_sha256']
j=json.loads(completion.read_text());assert j['status']=='EXPERIMENTS_COMPLETE_BACKUP_REVIEW_PENDING'
assert json.loads((r/'reports/server_stage.json').read_text())['status']=='complete'
for rel,h in request['required_files'].items():assert sha(r/rel)==h,rel
assert request['required_files']==j['required_files']
assert not (r/'reports/shutdown_request.json').exists(),'A request already exists; inspect without repeating'
jobs=['run_campaign.py','final_test.py','ops/stage_g.py','ops/resume_g_optimized.py','ops/continue_campaign.py','ops/final_g.py','ops/error_review.py','ops/adaptive_budget_guard.py','ops/preserve_remaining_g.py']
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  tokens=[x.decode('utf-8','replace') for x in (p/'cmdline').read_bytes().split(b'\0')]
  assert not any(x in jobs for x in tokens),'Experiment process remains '+p.name
 except FileNotFoundError:pass
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not gpu,'GPU compute process remains'
assert os.access('/usr/bin/shutdown',os.X_OK)
if not request['execute']:print(json.dumps(dict(status='dry_run_ready',shutdown_executed=False)));raise SystemExit
request['requested_at']=time.time();(r/'reports/shutdown_request.json').write_text(json.dumps(request,indent=2))
result=subprocess.run(['/bin/bash','/usr/bin/shutdown'],capture_output=True,text=True,timeout=25)
receipt=dict(status='shutdown_request_accepted' if result.returncode==0 else 'shutdown_command_failed',
 returncode=result.returncode,requested_at=request['requested_at'])
(r/'reports/shutdown_command_result.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    ready_path=ROOT/'reports/completion_ready.json';state=ROOT/'reports/shutdown_state.json'
    write_json(ROOT/'reports/shutdown_guard_pid.json',dict(pid=__import__('os').getpid(),time=time.time()))
    while not ready_path.exists():
        if not a.wait:print('Final reviewed completion not present; shutdown not attempted');return
        time.sleep(30)
    if state.exists() and json.loads(state.read_text(encoding='utf-8-sig')).get('status') in ['dispatching','shutdown_request_accepted','shutdown_dispatched_unconfirmed']:
        raise ValueError('A prior request is recorded; inspect it instead of retrying')
    ready=json.loads(ready_path.read_text(encoding='utf-8-sig'));check_local(ready)
    request=dict(completion_sha256=ready['completion_sha256'],required_files=ready['required_files'],execute=a.execute)
    code=GUARD.replace('PAYLOAD',repr(json.dumps(request)))
    c=connect()
    try:
        # Verify every condition once before marking dispatch intent.
        probe=code.replace(repr(json.dumps(request)),repr(json.dumps(dict(request,execute=False))))
        result=remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+probe+"\nPY",120)
        if not a.execute:print(result);return
        write_json(state,dict(status='dispatching',time=time.time(),completion_sha256=ready['completion_sha256']))
        write_json(ROOT/'reports/stop_mirror.json',dict(reason='Final verified backup complete; authorized shutdown dispatched'))
        try:
            result=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",120))
            write_json(state,result)
            print(result,flush=True)
        except Exception as exc:
            write_json(state,dict(status='shutdown_dispatched_unconfirmed',time=time.time(),reason=str(exc),
                note='Inspect remote receipt/reachability before any repeat; transport closure does not confirm platform OFF'))
            raise
    finally:c.close()

if __name__=='__main__':main()
