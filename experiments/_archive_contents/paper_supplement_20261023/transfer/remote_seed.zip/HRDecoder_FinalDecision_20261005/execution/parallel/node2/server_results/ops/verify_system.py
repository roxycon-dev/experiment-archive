"""Restore fixed Q controls and recompute original-pixel R counts; no old NPZ transfer."""
import sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from qlib.common import read_json,write_json,sha,utc_now
from experiment import ROOT,val_dataset,load_parent
from validation import evaluate,cache_p0

def main(runtime):
    old=Path(runtime['old_root']);assert sha(old/'configs/p0_original_lock.json')==runtime['old_p0_lock_sha256']
    assert sha(old/'reports/final_evaluation.py')==runtime['old_p0_driver_sha256']
    dataset=val_dataset(runtime);p0,p0_receipt=cache_p0(dataset,runtime)
    files=[{'sid':sid,'C_sha256':sha(Path(runtime['p0_c_cache'])/(sid+'.npz')),'U0_sha256':sha(Path(runtime['p0_u_cache'])/(sid+'.npz'))} for sid in dataset.ids]
    results={}
    for seed in (42,2024):
        model,loading=load_parent(runtime,seed,q_parent=True);model=model.cuda()
        current=evaluate(model,dataset,runtime,ROOT/f'reports/R_reconstruction_seed{seed}',16000,p0_ram=p0)
        original=read_json(old/f'runs/Q_DDR_seed{seed}/val_u16000/metrics.json')
        for row in current['per_image']:
            reference=next(r for r in original['per_image'] if r['sid']==row['sid'])
            assert row['fixed_fusion']==reference['fixed_fusion'],'Native restored R per-image counts differ: '+str(seed)+' '+row['sid']
        results[str(seed)]={'fixed_fusion':current['fixed_fusion'],'parent_sha256':loading['parent_sha256'],'original_metrics_sha256':sha(old/f'runs/Q_DDR_seed{seed}/val_u16000/metrics.json'),'all76_per_image_counts_equal':True}
        del model;gc.collect();torch.cuda.empty_cache()
    result={'status':'PASS','time':utc_now(),'P0_val_cache_files':files,'P0_RAM':p0_receipt,'R_val':results,'P0_and_R_formula_unchanged':True,'verification':'Both immutable Q16k models restored and full76 native validation recomputed; old probabilities remain only at their original local location.'}
    write_json(ROOT/'reports/system_acceptance.json',result);return result

if __name__=='__main__':main(read_json(ROOT/'configs/runtime.server.json'))
