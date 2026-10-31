"""One real 2048/batch2 v4 forward/backward, BF16 network, no optimizer."""
import gc,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch

root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
sys.path.insert(0,str(root/'training_execution'))
sys.path.insert(0,str(root/'diagnosis_v4'))
sys.path.insert(0,str(root/'vendor/champion_repo'))
from model import build_baseline,install_legacy_bf16_interpolation,tensor_state_sha
from data import TrainCacheDataset
from v4_architecture import V4Student,losses,routed_backward

out=root/'reports/diagnostics/v4/v4_full_resource_checks.json'
report={'status':'RUNNING','formal_updates':0,'optimizer_created':False,
        'canvas':2048,'batch':2,'device':torch.cuda.get_device_name(0),
        'precision':'BF16 autocast network; FP32 loss and probability mixture',
        'limitations':'One fresh forward/backward, not warmed-up throughput or full training stability; no SGD state allocated.'}
def save():out.write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    torch.set_num_threads(2)
    install_legacy_bf16_interpolation()
    base,_=build_baseline(root/'vendor/champion_repo',Path('/root/.cache/torch/hub/checkpoints/hrnetv2_w48-d2186c55.pth'))
    initial=tensor_state_sha(base.backbone)
    student=V4Student(base).cuda().train()
    assert tensor_state_sha(student.base.backbone)==initial
    dataset=TrainCacheDataset(root,Path('/root/HRDecoder/data/DDR_split_613'))
    idx={r['sid']:i for i,r in enumerate(dataset.rows)}
    sids=('007-3396-200','007-1811-100')
    rows=[dataset[(idx[sid],42000)] for sid in sids]
    batch={k:torch.stack([r[k] for r in rows]).cuda() for k in ('image','label','valid','A','U','S')}
    del rows;gc.collect()
    torch.manual_seed(20261009);np.random.seed(20261009)
    calls=[];hook=student.base.backbone.register_forward_hook(lambda *args:calls.append(1))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=student(batch['image'],batch['label'],batch['valid'])
    hook.remove();torch.cuda.synchronize()
    report['forward_seconds']=time.perf_counter()-start
    report['peak_after_forward_bytes']=torch.cuda.max_memory_allocated()
    report['stage']='forward_done';save();print(json.dumps(report),flush=True)
    task,comp=losses(prediction,batch['label'],batch['valid'],batch['A'],batch['U'],batch['S'],4000)
    assert task.dtype==comp.dtype==prediction['logits'].dtype==torch.float32
    torch.cuda.synchronize();start=time.perf_counter()
    routed_backward(task,comp,student.component_parameters())
    torch.cuda.synchronize()
    finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in student.parameters())
    nonzero=sum(int(p.grad is not None and bool((p.grad!=0).any())) for p in student.parameters())
    assert finite and nonzero>0 and len(calls)==1
    assert torch.equal(prediction['logits']>0,prediction['member_logits'][0]>0)
    assert bool(prediction['member_logits'][1].eq(0).all()) and bool(prediction['member_logits'][2].eq(0).all())
    report.update(status='PASS',stage='backward_done',sids=list(sids),augmentation_seed=42000,
                  backward_and_gradient_checks_seconds=time.perf_counter()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  total_GPU_bytes=torch.cuda.get_device_properties(0).total_memory,
                  estimated_SGD_momentum_bytes=sum(p.numel()*p.element_size() for p in student.parameters()),
                  task_loss=float(task.detach()),component_loss=float(comp.detach()),hr_loss=float(prediction['hr_loss'].detach()),
                  backbone_initialization_preserved=True,backbone_forward_calls=len(calls),all_gradients_finite=finite,
                  parameters_with_nonzero_gradient=nonzero,initial_threshold_decisions_match_anchor=True,
                  loss_dtype=str(task.dtype),accuracy_validated=False)
except Exception as e:
    report.update(status='FAIL',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save();print(json.dumps(report),flush=True);raise
save();print(json.dumps(report),flush=True)
