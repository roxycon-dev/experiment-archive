"""Serial finite queue, with no automatic retries or extra research recipes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from decision import ENDPOINT,NODES,promoted,select
from qlib.common import read_json,sha,utc_now,write_json
from train import ROOT


def nodes_for(arm,seed):
    return {n:read_json(ROOT/f'runs/{arm}_seed{seed}/val_u{n}/metrics.json') for n in NODES[arm]
            if (ROOT/f'runs/{arm}_seed{seed}/val_u{n}/metrics.json').exists()}


def invoke(arm,seed,runtime,resume):
    run=ROOT/f'runs/{arm}_seed{seed}'
    status=read_json(run/'status.json') if (run/'status.json').exists() else {}
    if status.get('stage') in ('COMPLETE','CATASTROPHIC_STOP'):
        return status
    cmd=[sys.executable,'-u','train.py','--runtime',runtime,'--arm',arm,'--seed',str(seed)]
    if (run/'last.pth').exists():
        if not resume:
            raise ValueError('Existing checkpoint: review technical pause and explicitly --resume-paused')
        cmd+=['--resume']
    with (ROOT/f'reports/{arm}_seed{seed}.log').open('ab') as log:
        code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    status=read_json(run/'status.json')
    if code or status['stage'] not in ('COMPLETE','CATASTROPHIC_STOP'):
        raise RuntimeError('Technical pause in '+arm+' seed'+str(seed)+'; inspect its log')
    return status


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',default='configs/runtime.server.json')
    ap.add_argument('--resume-paused',action='store_true');args=ap.parse_args()
    references=read_json(ROOT/'configs/references.json')
    state={'stage':'DISCOVERY','authorized_seed2024_arms':[],'finite_queue':True,'updated_at':utc_now()}
    write_json(ROOT/'reports/campaign.json',state)
    runs={}
    for arm in ('E_NEG','F4','F12'):
        state.update(active=f'{arm}_seed42',updated_at=utc_now());write_json(ROOT/'reports/campaign.json',state)
        runs[arm,42]=invoke(arm,42,args.runtime,args.resume_paused)
    signals={arm:promoted(arm,nodes_for(arm,42),42,references) for arm in ('E_NEG','F4','F12')}
    arms=[]
    if signals['E_NEG']:
        arms.append('E_NEG')
    if signals['F4'] or signals['F12']:
        arms.extend(('F4','F12'))
    state.update(stage='CONFIRMATION',promotion_signals=signals,authorized_seed2024_arms=arms,updated_at=utc_now())
    write_json(ROOT/'reports/campaign.json',state)
    for arm in arms:
        state.update(active=f'{arm}_seed2024',updated_at=utc_now());write_json(ROOT/'reports/campaign.json',state)
        runs[arm,2024]=invoke(arm,2024,args.runtime,args.resume_paused)
    lock=select(runs,references)
    lock.update(locked_at=utc_now(),runtime_sha256=sha(args.runtime),all_tried_nodes={
        f'{arm}_seed{seed}':nodes_for(arm,seed) for arm,seed in runs})
    lock['validation_file_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'runs').glob('*/val_u*/metrics.json')}
    if lock['selected']:
        selected_arm=lock['selected']['arm']
        lock['selected_checkpoint_sha256']={str(seed):sha(ROOT/f'runs/{selected_arm}_seed{seed}/checkpoint_u{ENDPOINT[selected_arm]}.pth') for seed in (42,2024)}
    target=ROOT/'reports/final_selection.json'
    if target.exists():
        old=read_json(target)
        assert old['selected']==lock['selected'] and old['test_authorized']==lock['test_authorized']
        assert old['validation_file_sha256']==lock['validation_file_sha256']
    else:
        write_json(target,lock)
    state.update(stage='LOCKED_TEST' if lock['test_authorized'] else 'NO_PROMOTION_FINISHED',
                 active=None,final_selection_sha256=sha(target),updated_at=utc_now())
    write_json(ROOT/'reports/campaign.json',state)
    if lock['test_authorized']:
        subprocess.run([sys.executable,'-u','locked_test.py','--runtime',args.runtime],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-u','report.py'],cwd=ROOT,check=True)
    state.update(stage='READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN',updated_at=utc_now())
    write_json(ROOT/'reports/campaign.json',state)
    print(state['stage'],flush=True)


if __name__=='__main__':
    descriptor=None
    try:
        descriptor=os.open(ROOT/'reports/campaign.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(descriptor,str(os.getpid()).encode())
        main()
    except FileExistsError:
        raise RuntimeError('Existing campaign.lock; confirm owner is inactive before manual recovery')
    except BaseException as error:
        state=read_json(ROOT/'reports/campaign.json') if (ROOT/'reports/campaign.json').exists() else {}
        state.update(stage='TECHNICAL_PAUSE',reason=str(error),automatic_retry=False,updated_at=utc_now())
        write_json(ROOT/'reports/campaign.json',state)
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor);(ROOT/'reports/campaign.lock').unlink()
