"""Exactly one recipe/formula, both fixed seeds, after immutable validation selection."""
import argparse
import gc
import subprocess
from pathlib import Path
import torch
from decision import ENDPOINT,select
from train import ROOT,initialize
from qlib.common import assert_source,digest,read_json,sha,utc_now,write_json
from qlib.data import DDR
from qlib.evaluate import evaluate
from qlib.metrics import paired_bootstrap


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',default='configs/runtime.server.json')
    args=ap.parse_args();runtime=read_json(args.runtime);assert_source(ROOT)
    selection_path=ROOT/'reports/final_selection.json';lock=read_json(selection_path)
    assert lock['test_authorized'] and sha(args.runtime)==lock['runtime_sha256']
    arm,formula=lock['selected']['arm'],lock['selected']['formula']
    references=read_json(ROOT/'configs/references.json')
    for rel,h in lock['validation_file_sha256'].items():
        assert sha(ROOT/rel)==h, 'Validation evidence changed after lock'
    runs={}
    for a in ('E_NEG','F4','F12'):
        for seed in (42,2024):
            path=ROOT/f'runs/{a}_seed{seed}/status.json'
            if path.exists():runs[a,seed]=read_json(path)
    assert select(runs,references)['selected']==lock['selected']
    checkpoint_hashes={str(seed):sha(ROOT/f'runs/{arm}_seed{seed}/checkpoint_u{ENDPOINT[arm]}.pth') for seed in (42,2024)}
    assert checkpoint_hashes==lock['selected_checkpoint_sha256']
    state_path=ROOT/'reports/test_state.json'
    if state_path.exists():
        previous=read_json(state_path)
        assert previous['selection_sha256']==sha(selection_path) and previous['checkpoint_sha256']==checkpoint_hashes
        if previous['stage']=='COMPLETE':return
        raise RuntimeError('Partial test requires technical recovery review; no automatic test replay')
    state={'stage':'RUNNING','selection_sha256':sha(selection_path),'checkpoint_sha256':checkpoint_hashes,
           'arm':arm,'formula':formula,'deployment_seed':42,'started_at':utc_now(),'test_reselection_forbidden':True}
    write_json(state_path,state)
    subprocess.run([runtime['legacy_python'],'-u',str(ROOT/'p0_test.py'),'--runtime',args.runtime],cwd=ROOT,check=True)
    runtime.update(final_test_authorized=True,p0_combined_cache=runtime['test_p0_cache'])
    data=read_json(ROOT/'reports/data_identity.json')
    # No test dataset construction occurs before this point.
    dataset=DDR(runtime['ddr_root'],'test',data['splits']['test'],test_authorized=True)
    results={}
    try:
        for seed in (42,2024):
            model,_=initialize(runtime,arm,seed)
            state_dict=torch.load(ROOT/f'runs/{arm}_seed{seed}/checkpoint_u{ENDPOINT[arm]}.pth',map_location='cpu',weights_only=False)
            assert state_dict['completed']==ENDPOINT[arm]
            assert state_dict['identity']==read_json(ROOT/f'runs/{arm}_seed{seed}/identity.json')
            assert state_dict['identity_sha256']==digest(state_dict['identity'])
            model.load_state_dict(state_dict['model'],strict=True);del state_dict
            model=model.to('cuda')
            result=evaluate(model,dataset,runtime,ROOT/f'reports/final_test/seed{seed}',ENDPOINT[arm],locked_formula=formula)
            old=read_json(ROOT/f'historical/R_test_seed{seed}.json')
            old_rows={r['sid']:r['fixed_fusion'] for r in old['per_image']}
            result['paired_bootstrap_vs_R']=paired_bootstrap([old_rows[r['sid']] for r in result['per_image']],
                                                            [r[formula] for r in result['per_image']])
            write_json(ROOT/f'reports/final_test/seed{seed}/metrics.json',result)
            results[str(seed)]={'miou_pct':result[formula]['miou_pct'],'gain_vs_R_pp':result[formula]['miou_pct']-references['r_test_by_seed'][str(seed)],
                                'paired_image_bootstrap':result['paired_bootstrap_vs_R']}
            del model;gc.collect();torch.cuda.empty_cache()
        mean=sum(r['miou_pct'] for r in results.values())/2
        write_json(ROOT/'reports/final_test/summary.json',{'stage':'COMPLETE','selection_sha256':sha(selection_path),
                   'formula':formula,'arm':arm,'seed_results':results,'test_mean':mean,'deployment_seed':42,
                   'target47_2115_both_seeds':all(r['miou_pct']>=47.2115 for r in results.values()),
                   'target47_2115_mean_only':mean>=47.2115 and not all(r['miou_pct']>=47.2115 for r in results.values()),
                   'historical_test_reused_limitation':True,'completed_at':utc_now()})
        state.update(stage='COMPLETE',completed_at=utc_now());write_json(state_path,state)
    except BaseException as error:
        state.update(stage='TECHNICAL_PAUSE',reason=str(error),automatic_restart=False,updated_at=utc_now())
        write_json(state_path,state);raise


if __name__=='__main__':main()
