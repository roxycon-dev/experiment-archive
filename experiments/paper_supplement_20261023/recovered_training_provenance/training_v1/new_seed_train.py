"""Fixed20k paired adaptation, reusing archived optimizer/loss/EMA and RNG code."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import cv2

ROOT=Path(__file__).resolve().parent
EXP=ROOT.parents[1]
sys.path.insert(0,str(EXP/'HRDecoder_Q_20261003/vendor/dinov3'))
from qlib.common import read_json,write_json,sha,digest,seed_all,utc_now
from qlib.data import DDR,catalog,draw_row,read_rgb
from qlib.geometry import Tile,rgb_window
from qlib.model import Q
from inputs import cache_sources,loader_for
from final_model import FinalModel
from schedule import optimizer_for,Schedule
from state import EMA,save,load
from engine import update

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['C','S'],required=True)
    parser.add_argument('--seed',type=int,choices=[2024,3407],required=True)
    parser.add_argument('--stop-at',type=int,choices=[100,20000],default=20000)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    # Avoid inheriting active CPU thread pools into the archived fork loader.
    torch.set_num_threads(1);cv2.setNumThreads(1)
    protocol=read_json(ROOT/'protocol.json')
    assert protocol['adaptation_training']['schedule_horizon']==32000
    data_root=EXP/'data/DDR_split_613'
    manifest=read_json(ROOT/'reports/data_identity.json')
    acceptance=read_json(ROOT/'reports/new_training_data_acceptance.json');assert acceptance['status']=='PASS'
    dataset=DDR(data_root,'train',manifest['splits']['train'])
    stream=ROOT/f'reports/new_stream_DDR_seed{args.seed}.jsonl'
    if not stream.exists():
        metadata,positives=catalog(dataset)
        with stream.open('w',encoding='utf-8') as out:
            for u in range(1,20002):
                for slot in range(8):
                    out.write(json.dumps(draw_row(dataset,metadata,positives,args.seed,u,slot,None if slot<4 else slot-4),separators=(',',':'))+'\n')
        write_json(stream.with_suffix('.identity.json'),{'sha256':sha(stream),'seed':args.seed,'rows':20001*8,'schedule_horizon':32000})
    stream_identity=read_json(stream.with_suffix('.identity.json'));assert sha(stream)==stream_identity['sha256']
    rows=[json.loads(line) for line in stream.read_text().splitlines()]
    assert len(rows)==20001*8
    parent_path=ROOT/'F12_parent_model_exact.pth';receipt=read_json(ROOT/'reports/training_parent_export.json')
    assert sha(parent_path)==receipt['compact_sha256'] and receipt['model_tensors_bitwise_equal']
    seed_all(args.seed)
    from dinov3.hub.backbones import dinov3_vitb16
    parent=Q(dinov3_vitb16(pretrained=False))
    saved=torch.load(parent_path,map_location='cpu',weights_only=False)
    assert saved['source_full_sha256']==protocol['common_parent']['full_checkpoint_sha256'] and saved['completed']==12000
    parent.load_state_dict(saved['model'],strict=True)
    del saved
    if args.arm=='S':
        from kernel import load_kernel
        load_kernel()
    model=FinalModel(parent,args.arm,args.seed,backend='cuda').cuda().train()
    # Verify inherited prediction identity before any optimizer update.
    parent=parent.cuda().eval();model.eval()
    x=torch.from_numpy(rgb_window(read_rgb(dataset.images[dataset.ids[0]]),Tile(0,0,768,768),1.,1.)).unsqueeze(0).cuda()
    with torch.no_grad():initial_delta=float((parent(x)-model(x)).abs().max())
    assert initial_delta<=1e-5,initial_delta
    model.train();del parent,x
    optimizer,groups=optimizer_for(model);scheduler=Schedule(optimizer);ema=EMA(model)
    source_paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.py','.h','.cuh','.cu','.cpp') and 'msda_build' not in p.parts)
    source_identity={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
    identity={'campaign':'supplement_20261023','arm':args.arm,'seed':args.seed,'endpoint':20000,'schedule_horizon':32000,
              'parent_full_sha256':protocol['common_parent']['full_checkpoint_sha256'],
              'parent_compact_sha256':receipt['compact_sha256'],'stream_sha256':stream_identity['sha256'],
              'protocol_sha256':sha(ROOT/'protocol.json'),'data_manifest_sha256':sha(ROOT/'reports/data_identity.json'),
              'source_files':source_identity,'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0)}
    run=ROOT/f'runs/{args.arm}_seed{args.seed}';run.mkdir(parents=True,exist_ok=True)
    next_rows=lambda n:{'DDR':rows[n*8:n*8+8],'TJDR':[]}
    completed=0
    if args.resume:completed=load(run/'last.pth',model,optimizer,scheduler,ema,identity,next_rows)
    else:assert not (run/'last.pth').exists(),'Existing run requires explicit resume'
    write_json(run/'identity.json',identity);write_json(run/'parameter_groups.json',groups)
    write_json(run/'initialization_acceptance.json',{'status':'PASS','full_parent_strictly_loaded':True,
               'FP32_initial_logits_max_abs_difference':initial_delta,'tolerance':1e-5,'training_image':dataset.ids[0]})
    cached,cache_receipt=cache_sources({'DDR':dataset},80*1024**3,
        lambda r:print(json.dumps({'stage':'CACHING',**r}),flush=True))
    write_json(run/'cache.json',cache_receipt)
    resources={'workers':8,'source_workers':0,'prefetch_factor':2}
    loaders={'DDR':loader_for(cached,rows,'DDR',completed,args.stop_at,resources,args.seed)}
    start=time.monotonic();times=[];audits={}
    try:
        for u in range(completed+1,args.stop_at+1):
            stat=update(model,optimizer,scheduler,ema,loaders,args.seed,u-1,audits)
            completed=u;times.append(stat['seconds'])
            if u%20==0:
                torch.cuda.synchronize()
                with (run/'train.jsonl').open('a') as out:out.write(json.dumps({'update':u,**stat})+'\n')
                progress={'stage':'TRAINING','completed':u,'endpoint':20000,'arm':args.arm,'seed':args.seed,
                          'recent':stat,'updated_at':utc_now(),'peak_GPU_bytes':torch.cuda.max_memory_allocated()}
                write_json(run/'status.json',progress);print(json.dumps(progress),flush=True)
            if u%1000==0 or u==args.stop_at:
                save(run/'last.pth',model,optimizer,scheduler,ema,u,identity,next_rows(u))
        if args.stop_at==100:
            write_json(run/'benchmark.json',{'status':'FORMAL_FIRST100_SAVED','completed':completed,
                'mean_seconds':float(np.mean(times[10:])),'remaining_training_seconds_estimate':float(np.mean(times[10:]))*(20000-100),
                'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'formal_updates_retained':True})
            stage='BENCHMARK_COMPLETE_RESUME_REQUIRED'
        else:
            write_json(run/'completion.json',{'status':'COMPLETE','completed':completed,'schedule_horizon':32000,
                       'EMA_completed':ema.completed,'endpoint_checkpoint_sha256':sha(run/'last.pth')})
            stage='COMPLETE'
        write_json(run/'status.json',{'stage':stage,'completed':completed,'endpoint':20000,'seconds_this_launch':time.monotonic()-start})
    except BaseException as exc:
        write_json(run/'status.json',{'stage':'TECHNICAL_PAUSE','completed_in_memory':completed,'error':repr(exc)})
        raise

if __name__=='__main__':main()
