"""Resolve portable intent configurations; no training or server connection."""
import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def configure(ddr_root, out, micro_batch=2, grad_accum=4):
    if micro_batch*grad_accum != 8:
        raise ValueError('Effective batch must be 8')
    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    templates={p.stem[:2]:p for p in (ROOT/'configs').glob('[HU][01]*yaml')}
    result={}
    for arm,path in templates.items():
        cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
        h=arm.startswith('H')
        st=cfg['structure']
        weight=ROOT/'assets'/('hrnetv2_w48-d2186c55.pth' if h else 'convnext_small_1k_224_ema.pth')
        source=('https://download.openmmlab.com/pretrain/third_party/hrnetv2_w48-d2186c55.pth' if h else
                'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth')
        iw=dict(source=source,file=str(weight),sha256=sha(weight))
        cfg['initialization']['imagenet_weights']=iw
        if h:
            st['reference_root']=str(ROOT/'vendor/HRDecoder')
        else:
            st.update(encoder_weights_source=source,encoder_weights_file=str(weight),
                      encoder_weights_sha256=iw['sha256'],library_version='builtin-source-20261002')
        cfg['data']['ddr_root']=str(Path(ddr_root).resolve())
        ddr=cfg['ddr_finetune_stage']
        ddr.update(micro_batch=micro_batch,grad_accum=grad_accum,gradient_checkpointing=False,
                   log_every=100,num_workers=4 if platform.system()!='Windows' else 0,
                   periodic_save_updates=1000)
        ddr['screening']=dict(enabled=True,min_progress=.5,minimum_gain=.003,gap=.015,
            reference_miou=.426694,reference_class_iou=dict(EX=.457715,HE=.437583,SE=.551807,MA=.259672),
            class_margin=.005)
        cfg['metrics'].update(eval_mode='sliding_window',window_size=1024,window_stride=768,
                              resize_to=[2048,2048])
        cfg['execution_revision']='20261002: fp32 loss, real transfer, source sampling, explicit geometry'
        cfg['execution_fast_kernels']=True
        cfg['historical_comparison']='descriptive_only: historical controls used an earlier implementation'
        if arm.endswith('1'):
            cfg['initialization']['external_pretrain_checkpoint']='REQUIRED_AFTER_EXTERNAL_PRETRAIN'
            ps=cfg['pretrain_stage']
            ps.update(external_manifest=str(ROOT/'assets/external/external_train_dedup.jsonl'),
                      manifest_base_dir=str(ROOT/'assets/external'),seed=42,
                      micro_batch=micro_batch,grad_accum=grad_accum,mixed_precision=True,
                      gradient_checkpointing=False,num_workers=ddr['num_workers'],log_every=100,
                      updates=2400,eval_progress_points=[.125,.25,.5,1],periodic_save_updates=1000,
                      init_checkpoint_output=str(ROOT/f'work_dirs/{arm}_external_pretrain/ckpt_best.pth'),
                      overfit_monitor='48 train / 6 dev; nodes 300/600/1200/2400; plateau early exit; budget scaled before new validation')
            ps['screening']=dict(enabled=True,mode='external_dev_plateau',min_progress=.25,minimum_gain=.001)
        target=out/path.name
        target.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        result[arm]=str(target)
    a=copy.deepcopy(yaml.safe_load(Path(result['U0']).read_text(encoding='utf-8')))
    a['arm_id']='A0'
    a['structure']=dict(family='hacdr',paired_group='hacdr',training_entry='train.py',
        reference_root=str(ROOT/'vendor/HACDR-Net'),source_commit='93eb3d41e96f5add9027f1a4977e54eb4fdc9bb0',
        variant='official backbone plus project four-logit decoder; not paper reproduction')
    a['initialization']=dict(imagenet_weights=None,external_pretrain_checkpoint=None)
    a['ddr_finetune_stage'].update(backbone_lr=6e-5,decoder_lr=9e-4)
    a.pop('equivalence_lock',None)
    target=out/'A0_hacdr_adapted.yaml'
    target.write_text(yaml.safe_dump(a,allow_unicode=True,sort_keys=False),encoding='utf-8')
    result['A0']=str(target)
    (out/'resolved_paths.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ddr-root',required=True)
    ap.add_argument('--out',default=str(ROOT/'configs/resolved'))
    ap.add_argument('--micro-batch',type=int,default=2)
    ap.add_argument('--grad-accum',type=int,default=4)
    a=ap.parse_args()
    print(json.dumps(configure(a.ddr_root,a.out,a.micro_batch,a.grad_accum),indent=2))

if __name__=='__main__':
    main()
