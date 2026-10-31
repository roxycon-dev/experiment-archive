import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
"""Prelocked candidate only, both seeds once; no test-driven reselection."""
import argparse,os,json
import torch
from qlib.common import read_json,write_json,sha,digest,utc_now
from qlib.metrics import paired_bootstrap
from experiment import ROOT,initialize,val_dataset
from validation import evaluate

def run(lock_path,runtime_path):
    lock=read_json(lock_path);raw={k:v for k,v in lock.items() if k!='lock_sha256'};assert digest(raw)==lock['lock_sha256']
    decision=read_json(ROOT/'reports/decision.json');assert decision['test_authorized'] and decision['selection_lock_sha256']==lock['lock_sha256']
    assert read_json(ROOT/'reports/source_identity.json')['sha256']==lock['source_sha256']
    assert lock['runtime_sha256']==sha(runtime_path)
    for rel,expected in lock['validation_files_sha256'].items():assert sha(ROOT/rel)==expected
    runtime=read_json(runtime_path);runtime['final_test_authorized']=True
    state_path=ROOT/'reports/final_test_state.json'
    if state_path.exists():
        previous=read_json(state_path);assert previous['lock_sha256']==lock['lock_sha256']
        if previous['stage']=='COMPLETE':return read_json(ROOT/'reports/final_test.json')
        raise RuntimeError('Partial locked test requires explicit diagnosed recovery; no automatic replay')
    write_json(state_path,{'stage':'RUNNING','lock_sha256':lock['lock_sha256'],'time':utc_now(),'reselection_permitted':False})
    if lock['formula']!='single':
        import subprocess
        subprocess.run([runtime['legacy_python'],'-u',str(ROOT/'p0_test.py'),'--runtime',str(runtime_path)],cwd=ROOT,check=True)
        runtime['p0_combined_cache']=runtime['test_p0_cache']
    result={'stage':'RUNNING','lock_sha256':lock['lock_sha256'],'seeds':{},'time':utc_now()}
    for seed in (42,):
        output=ROOT/f'runs/LOCKED_TEST_seed{seed}'
        if (output/'metrics.json').exists():
            saved=read_json(output/'test_identity.json');assert saved['lock_sha256']==lock['lock_sha256']
            metrics=read_json(output/'metrics.json')
        else:
            output.mkdir(exist_ok=True)
            marker=output/'test_identity.json'
            # A partial test is a technical pause needing explicit review; never
            # silently run another test attempt or swap a checkpoint.
            with marker.open('x',encoding='utf-8') as f:json.dump({'lock_sha256':lock['lock_sha256'],'seed':seed,'time':utc_now()},f)
            checkpoint=ROOT/lock['checkpoints'][str(seed)]['path'];assert sha(checkpoint)==lock['checkpoints'][str(seed)]['sha256']
            from kernel import load_kernel
            if lock['arm'] in ('S','SD'):load_kernel()
            model,_=initialize(runtime,lock['arm'],seed)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['completed']==lock['update']
            model.load_state_dict(state['ema'],strict=True);del state
            metrics=evaluate(model,val_dataset(runtime,'test',True),runtime,output,lock['update'],locked_formula=lock['formula'])
            del model;torch.cuda.empty_cache()
        result['seeds'][str(seed)]=metrics[lock['formula']]
        # Historical R integer counts are immutable inputs, never a new search.
        reference=read_json(ROOT/f'configs/R_test_counts_seed{seed}.json')
        assert [r['sid'] for r in reference['per_image']]==[r['sid'] for r in metrics['per_image']]
        old=[r['R'] for r in reference['per_image']];new=[r[lock['formula']] for r in metrics['per_image']]
        result['seeds'][str(seed)]['conditional_paired_image_bootstrap']=paired_bootstrap(old,new)
        write_json(ROOT/'reports/final_test.json',result)
    result.update(stage='COMPLETE',test_mean_miou_pct=sum(r['miou_pct'] for r in result['seeds'].values())/len(result['seeds']),
                  seed_scope=[42],single_seed_exploratory=True,
                  deployment_seed=42,reselection_permitted=False)
    write_json(ROOT/'reports/final_test.json',result)
    write_json(state_path,{'stage':'COMPLETE','lock_sha256':lock['lock_sha256'],'completed_at':utc_now(),'reselection_permitted':False})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',default='reports/final_selection_lock.json');p.add_argument('--runtime',default='configs/runtime.server.json');a=p.parse_args();run(a.lock,a.runtime)
