"""One public engineering preflight per family; no performance selection."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train
from trainer.config import load_yaml,ddr_data_config,resolve_stage_section,effective_batch_check
from trainer.models import build_model
from trainer.optim import build_param_groups,build_optimizer
from trainer.engine import StageRunner,build_effective_config
from trainer.seed import seed_everything

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True);ap.add_argument('--stage',default='external_pretrain')
    ap.add_argument('--out',required=True);ap.add_argument('--updates',type=int,default=3)
    a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    cfg=load_yaml(a.config);dc=ddr_data_config(cfg);sc=resolve_stage_section(cfg,a.stage)
    sc['num_workers']=2
    seed_everything(42,deterministic=True)
    model,meta=build_model(cfg,stage=a.stage,gradient_checkpointing=sc['gradient_checkpointing'])
    model.cuda()
    ds,dev,fp,manifest=train.build_dataset_for_stage(cfg,stage=a.stage,stage_cfg=sc,data_cfg=dc,
        train_split='train',val_split='val',make_augment=train.make_augment_factory(sc,train_crop=dc['train_crop']))
    groups=build_param_groups(model,backbone_lr=sc['backbone_lr'],decoder_lr=sc['decoder_lr'],
                             weight_decay=sc['weight_decay'],backbone_prefixes=meta['backbone_prefixes'])
    optimizer=build_optimizer(sc,groups)
    eff=build_effective_config(cfg=cfg,stage=a.stage,mode='calibration',stage_cfg=sc,
        batch=effective_batch_check(sc,stage=a.stage),seed=42,total_updates=sc['updates'],
        backbone_prefixes=meta['backbone_prefixes'],backbone_norm_eval=meta['backbone_norm_eval'],
        config_sha256_value=train._effective_digest(cfg),data_manifest_sha256=fp,
        manifest_path=manifest,out_dir=None)
    runner=StageRunner(model=model,dataset=ds,loss_fn=train.build_criterion(cfg),optimizer=optimizer,
                       cfg=eff,device='cuda')
    runner._set_train_mode()
    # Keep one representative tensor from both parameter groups, not a full
    # duplicate model in GPU memory.
    names=[]
    for prefix in [meta['backbone_prefixes'][0].rstrip('.'),'decoder' if hasattr(model,'decoder') else 'decode_head']:
        names.append(next(n for n,p in model.named_parameters() if n.startswith(prefix) and p.requires_grad and p.ndim>1))
    before={n:dict(model.named_parameters())[n].detach().cpu().clone() for n in names}
    stream=runner._infinite_batches();timings=[]
    for _ in range(a.updates):
        torch.cuda.synchronize();start=time.monotonic()
        ok,parts=runner.train_one_update_window(stream)
        torch.cuda.synchronize();timings.append(time.monotonic()-start)
        assert ok
    deltas={n:float((dict(model.named_parameters())[n].detach().cpu()-before[n]).abs().max()) for n in names}
    if not all(v>0 for v in deltas.values()):raise RuntimeError(f'First update failed: {deltas}')
    result=dict(family=meta['family'],stage=a.stage,gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__),peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
        timings_seconds=timings,effective_batch=sc['effective_batch'],micro_batch=sc['micro_batch'],
        grad_accum=sc['grad_accum'],parameter_delta=deltas,last_loss=parts,
        n_train=len(ds),n_dev=len(dev),weight_report=meta.get('weight_report'),
        calibration_only=True,checkpoint_saved=False)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='weight_report'},indent=2))

if __name__=='__main__':main()
