"""One real 2048 batch, sequential layout probes, no saved or formal SGD updates."""
import json
from pathlib import Path
import statistics
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vendor/champion_repo'))
import train
import torch
import cv2
from channels_last_runtime import enable_norm_layout
from ram_loader import install_ram_loader
from data import TrainCacheDataset

class DrySGD(torch.optim.SGD):
    def step(self,closure=None): return None

def main():
    assert (train.ROOT/'reports/shutdown_hold.enabled').exists()
    saved=json.loads((train.ROOT/'reports/FINAL_TUNING_CHECKPOINT_SAVED.json').read_text())['saved_completed']
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=True
    train.install_legacy_bf16_interpolation()
    ready=install_ram_loader(train.ROOT)
    dataset=TrainCacheDataset(train.ROOT,train.DATA)
    indexes=next(iter(train.ResumableBatches(len(dataset),saved,saved+1)))
    cpu_batch=train.torch.utils.data.default_collate([dataset[i] for i in indexes])
    batch=train.move(cpu_batch)
    checkpoint=torch.load(train.ROOT/'training_runs/v3/last.pt',map_location='cpu')
    assert checkpoint['completed']==saved
    baseline,_=train.build_baseline(train.REPO,train.W48)
    model=train.V3Student(baseline).cuda().train()
    del baseline
    optimizer=DrySGD(model.parameters(),lr=.005,momentum=.9,weight_decay=.0005)
    scaler=torch.cuda.amp.GradScaler(enabled=False)
    results=[]; reference_grads=None; reference_losses=None
    for enabled in (False,True,False):
        fmt=torch.channels_last if enabled else torch.contiguous_format
        enable_norm_layout(enabled)
        model.to(memory_format=fmt); model.load_state_dict(checkpoint['model'])
        batch['image']=batch['image'].contiguous(memory_format=fmt)
        train.restore_rng(checkpoint['rng'])
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        losses=train.one_step(model,optimizer,scaler,batch,saved+1,'v3')
        torch.cuda.synchronize()
        if reference_grads is None:
            reference_grads={n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            reference_losses=losses
        elif enabled:
            numerator=denominator=0.
            for name,p in model.named_parameters():
                if p.grad is None: continue
                g=p.grad.detach().float().cpu(); r=reference_grads[name]
                numerator+=float((g-r).square().sum()); denominator+=float(r.square().sum())
            grad_error=(numerator/max(denominator,1e-30))**.5
            loss_error=max(abs(losses[k]-reference_losses[k]) for k in ('dice','hr_dice','kl','signed_aux'))
        timings=[]
        for index in range(10):
            started=time.monotonic()
            train.one_step(model,optimizer,scaler,batch,saved+1,'v3')
            torch.cuda.synchronize()
            if index>=4: timings.append(time.monotonic()-started)
        row={'layout':'channels_last' if enabled else 'NCHW','median_seconds':statistics.median(timings),
            'mean_seconds':statistics.mean(timings),'timings':timings,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'losses':losses}
        results.append(row); print(json.dumps(row),flush=True)
    old=statistics.mean([r['median_seconds'] for r in results if r['layout']=='NCHW'])
    candidate=results[1]['median_seconds']
    passed=grad_error<=.05 and loss_error<=.01
    adopt=passed and candidate<old*.97
    result={'status':'PASS' if passed else 'REJECTED_NUMERICAL_PARITY','saved_completed':saved,
        'canvas':2048,'batch':2,'formal_updates':0,'checkpoint_written':False,
        'optimizer_step_disabled':True,'global_gradient_relative_L2':grad_error,
        'max_loss_absolute_difference':loss_error,'gradient_tolerance':.05,'loss_tolerance':.01,
        'measurements':results,'speedup_percent':100*(old/candidate-1),'adopt_channels_last':adopt,
        'RAM_cache_bytes':ready['bytes'],'time_unix':time.time()}
    train.write_json(train.ROOT/'reports/GPU_LAYOUT_BENCHMARK.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='measurements'}),flush=True)

if __name__=='__main__': main()
