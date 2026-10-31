"""Serial, resumable experiment queue. Failures remain engineering failures.

External assets do not block the DDR-only heterogeneous candidate. Existing U0
and H0 are reused as descriptive history; never silently claimed as matched
controls after implementation changes. Formal retries do not run automatically.
"""
import argparse
import atexit
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.configure_server import sha

def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');tmp.replace(path)

def run_command(cmd,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as f:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        return process.wait()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--configs',default=str(ROOT/'configs/resolved/resolved_paths.json'))
    ap.add_argument('--arms',default='U1,H1,A0')
    a=ap.parse_args()
    paths=json.loads(Path(a.configs).read_text(encoding='utf-8'))
    state_path=ROOT/'work_dirs/campaign_state.json'
    lock=ROOT/'work_dirs/campaign.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.exists():
        try:os.kill(int(lock.read_text()),0)
        except (ProcessLookupError,ValueError):lock.unlink()
        else:raise RuntimeError('Campaign already running')
    lock.write_text(str(os.getpid()))
    atexit.register(lambda:lock.unlink() if lock.exists() else None)
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else dict(jobs={})
    state.update(pid=os.getpid(),status='running',started_at_epoch=time.time(),
                 skipped_historical=['U0_complete_12000','H0_partial_6000_no_local_restart'])
    atomic(state_path,state)
    for arm in a.arms.split(','):
        cfg_path=Path(paths[arm]);cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        stages=['external_pretrain','ddr_finetune'] if arm in ('H1','U1') else ['ddr_finetune']
        external_failed=False
        for stage in stages:
            job=f'{arm}_{stage}';out=ROOT/'work_dirs'/job
            summary=out/'completion_summary.json'
            if summary.exists():
                done=json.loads(summary.read_text(encoding='utf-8'))
                if done.get('status') in ['complete','stopped']:
                    state['jobs'][job]=dict(status='reused',summary=done);atomic(state_path,state);continue
            if external_failed:continue
            if stage=='ddr_finetune' and arm in ('H1','U1'):
                best=ROOT/f'work_dirs/{arm}_external_pretrain/ckpt_best.pth'
                if not best.exists():
                    state['jobs'][job]=dict(status='data_or_dependency_blocked',reason='no formal external checkpoint')
                    atomic(state_path,state);continue
                cfg['initialization']['external_pretrain_checkpoint']=dict(file=str(best),sha256=sha(best))
                cfg_path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
            state.update(active_job=job)
            state['jobs'][job]=dict(status='running',started_at_epoch=time.time(),config_sha256=sha(cfg_path))
            atomic(state_path,state)
            cmd=[sys.executable,'-u','train.py','--config',str(cfg_path),'--stage',stage,
                 '--mode','formal','--out-dir',str(out)]
            # A saved checkpoint is resumed only under the strict existing
            # config/manifest checks; no automatic mismatch exemptions.
            if (out/'ckpt_last.pth').exists():cmd+=['--resume',str(out/'ckpt_last.pth')]
            rc=run_command(cmd,out/'train.log')
            result=json.loads(summary.read_text(encoding='utf-8')) if summary.exists() else {}
            state['jobs'][job].update(status=result.get('status','engineering_failed') if rc==0 else 'engineering_failed',
                                     exit_code=rc,finished_at_epoch=time.time(),summary=result)
            atomic(state_path,state)
            if rc!=0:
                # Continue independent families while retaining logs and the
                # failure for repair; never classify a crash as method failure.
                external_failed=stage=='external_pretrain'
    state.update(status='main_training_done',active_job=None,finished_at_epoch=time.time(),
        next_action='nominate at most two families; export probabilities; constrained cross-fit; promote only useful candidates')
    atomic(state_path,state)
    print(json.dumps(state,indent=2))

if __name__=='__main__':main()
