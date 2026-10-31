"""Useful bounded train-only diagnosis: no optimizer update, no val/test input, no checkpoint writes."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'sacd_execution'))
import train as core
from objectives import losses,complementary_targets


def dot_stats(left,right):
    dot=sum((a*b).sum() for a,b in zip(left,right) if a is not None and b is not None)
    aa=sum(a.square().sum() for a in left if a is not None)
    bb=sum(b.square().sum() for b in right if b is not None)
    return {'cosine':float(dot/(aa*bb).sqrt().clamp_min(1e-20)),
            'task_norm':float(aa.sqrt()),'aux_norm':float(bb.sqrt())}


def main():
    report=ROOT/'reports/reconstruction'
    report.mkdir(parents=True,exist_ok=True)
    with open(ROOT/'reports/gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import subprocess
        gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],universal_newlines=True).strip()
        assert not gpu,'Refuse to control any foreign GPU work'
        assert torch.__version__=='2.0.1+cu117'
        core.install_legacy_bf16_interpolation();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
        material,digest=core.identity()
        state=torch.load(ROOT/'training_runs/sacd_micro/parent_20000.pt',map_location='cpu')
        assert state['completed']==20000 and state['identity_SHA']==digest
        base,_=core.build_baseline(core.REPO,core.W48,verified_weight_sha=material['ImageNet_W48_SHA'])
        model=core.SACDStudent(base).cuda().train();model.load_state_dict(state['model'],strict=True)
        # Only private task/aux gradients are measured. Freezing provider weights
        # avoids retaining unused provider autograd activations; forward and BN
        # mode stay identical, and private parameter derivatives are unchanged.
        model.base.requires_grad_(False)
        core.restore_rng(state['rng']);del state
        core.install_ram(ROOT)
        from data import TrainCacheDataset
        dataset=TrainCacheDataset(ROOT,core.DATA)
        # At most two train-split exposure passes. Finish sooner when the next
        # candidate is prepared; the stop flag is checked only at safe boundaries.
        loader=core.CUDAPrefetch(dataset,core.ResumableBatches(len(dataset),20000,20454),workers=12)
        params=model.private_parameters();rows=[]
        started=time.time()
        core.write_json(report/'DIAGNOSIS_RUNNING.json',{'status':'RUNNING','pid':os.getpid(),
            'parent_identity_SHA':digest,'start_unix':started,'max_train_batches':454,
            'optimizer_updates':0,'teacher_and_GT_inputs':'train split only','external_diagnostics_controlled':False})
        for number,batch in enumerate(loader,1):
            if (report/'STOP_DIAGNOSIS_AFTER_BATCH.enabled').exists():break
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(batch['image'],batch['label'],batch['valid'])
            task,aux,metrics=losses(output,batch['label'],batch['valid'],batch['A'],batch['U'],batch['S'],20000)
            assert bool(torch.isfinite(torch.stack((task,aux))).all())
            auxiliary=torch.autograd.grad(aux,params,retain_graph=True,allow_unused=True)
            primary=torch.autograd.grad(task,params,allow_unused=True)
            assert all(bool(g.isfinite().all()) for gradients in (primary,auxiliary) for g in gradients if g is not None)
            value=dot_stats(primary,auxiliary)
            with torch.no_grad():
                p=output['base_logits'].float().sigmoid()
                target=complementary_targets(batch['A'],batch['U'],batch['S'],batch['label'],batch['valid'],output['base_logits'])
                close=(p-.5).abs()<=.01
                y=torch.stack([batch['label'][:,0]==c for c in range(1,5)],1)
                valid=target['valid']
                value.update(train_batch=number,seconds=time.time()-started,
                    near_threshold_pixels=(close&valid).sum((0,2,3)).tolist(),
                    near_threshold_positive_pixels=(close&valid&y).sum((0,2,3)).tolist(),
                    action_region_pixels=[region.sum((0,2,3)).tolist() for region in target['regions']],
                    losses={k:float(v.detach()) if torch.is_tensor(v) else v for k,v in metrics.items()})
            rows.append(value)
            if number%10==0:
                core.write_json(report/'gradient_diagnosis_progress.json',{'status':'TRAIN_ONLY_DIAGNOSIS',
                    'train_batches':number,'optimizer_updates':0,'seconds':time.time()-started,
                    'private_task_aux_cosine_mean':sum(r['cosine'] for r in rows)/len(rows),
                    'conflicting_gradient_batches':sum(r['cosine']<0 for r in rows),
                    'time_unix':time.time()})
                print(json.dumps(value),flush=True)
            del output,target,p,primary,auxiliary,task,aux,batch,metrics,close,y,valid
        core.write_json(report/'GRADIENT_DIAGNOSIS_COMPLETE.json',{'status':'COMPLETE',
            'rows':rows,'optimizer_updates':0,'parent_checkpoint_unchanged':True,
            'train_split_only':True,'test_or_val_accessed':False,'time_unix':time.time()})
        core.write_json(report/'DIAGNOSIS_RUNNING.json',{'status':'COMPLETE','pid':os.getpid(),
            'train_batches':len(rows),'optimizer_updates':0,'time_unix':time.time()})


if __name__=='__main__':main()
