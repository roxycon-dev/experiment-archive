"""Zero-SGD speed/memory comparison after preflight; weights discarded."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json,sys,time,traceback
from pathlib import Path
BASE=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
sys.path.insert(0,str(BASE/'code'))
import torch
from data import SegmentationData,ResumableBatches
from model import NativePyramid
from losses import SparseLesionLoss
from train import configure_runtime,seed_all,gradient_norms,write_json,runtime_identity

if __name__=='__main__':
    configure_runtime();torch.set_num_threads(4);seed_all(42)
    report=BASE/'preflight_attempt03/preflight.json'
    deadline=time.monotonic()+180
    while not report.is_file():
        if time.monotonic()>deadline:raise RuntimeError('Preflight did not finish before benchmark')
        time.sleep(2)
    assert json.loads(report.read_text())['status']=='PASS'
    assert not (BASE/'run_seed42_launch02').exists()
    dataset=SegmentationData(BASE/'data',BASE/'code/preparation_20261016/data_identity.json','train',2048)
    pair=next(iter(ResumableBatches(454,0,1,42)))
    batch=[dataset[item] for item in pair]
    image=torch.stack([b['image'] for b in batch]).cuda()
    label=torch.stack([b['label'] for b in batch]).cuda()
    criterion=SparseLesionLoss();denominator=criterion.batch_denominators(label)
    records=[]
    for checkpointing,layout in ((True,'contiguous'),(False,'contiguous'),(False,'channels_last')):
        model=momentum=None
        record={'checkpointing':checkpointing,'memory_format':layout,'formal_updates':0,'SGD_updates':0,
                'global_batch':2,'micro_batch':1,'test_accessed':False}
        try:
            model=NativePyramid(BASE/'assets/hrnetv2_w48-d2186c55.pth',checkpointing=checkpointing).cuda().train()
            if layout=='channels_last':model.to(memory_format=torch.channels_last)
            with torch.no_grad():
                for p in (model.decoder.auxiliary.weight,model.decoder.readout.weight):
                    p.copy_(torch.linspace(-.01,.01,p.numel(),device=p.device).reshape_as(p))
            # Reserve the real SGD momentum footprint without stepping weights.
            momentum=[torch.zeros_like(p) for p in model.parameters()]
            samples=[];torch.cuda.reset_peak_memory_stats()
            for iteration in range(4):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.monotonic()
                for j in range(2):
                    x=image[j:j+1]
                    if layout=='channels_last':x=x.contiguous(memory_format=torch.channels_last)
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(x)
                    loss,_=criterion(output,label[j:j+1],denominators=denominator)
                    loss.backward();del output,loss,x
                gradient_norms(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                gradient_norms(model)
                torch.cuda.synchronize();samples.append(time.monotonic()-started)
            record.update(status='PASS',seconds_per_pair=samples,
                          steady_seconds_per_pair=sum(samples[1:])/3,
                          peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                          peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
        except BaseException as error:
            record.update(status='FAILED',error_type=type(error).__name__,error=str(error))
        finally:
            del model,momentum;gc.collect();torch.cuda.empty_cache()
        records.append(record)
        write_json(BASE/'receipts/gpu_performance_zero_SGD.json',{'environment':runtime_identity(),'records':records})
        print(json.dumps(record),flush=True)
