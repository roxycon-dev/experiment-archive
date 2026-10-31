"""Synthetic, zero-SGD backend diagnosis. Never relaxes parity tolerances."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import json
from pathlib import Path
import sys
import traceback

BASE=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
sys.path.insert(0,str(BASE/'code'))
import torch
from model import NativePyramid
from train import seed_all,runtime_identity,write_json,source_identity
from verify import checkpoint_parity,precision_micro_parity

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('cudnn','strict','perimage'))
    args=parser.parse_args()
    out=BASE/'backend_diagnosis_20261017'/args.mode
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(args.mode=='strict')
    seed_all(42)
    record={'mode':args.mode,'environment':runtime_identity(),'source_sha256':source_identity()[0],
            'formal_updates':0,'real_data_updates':0,'test_accessed':False,
            'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}
    try:
        model=NativePyramid(BASE/'assets/hrnetv2_w48-d2186c55.pth').cuda().train()
        if args.mode=='perimage':
            import types
            original=model.forward
            def per_image(self,image,**kwargs):
                if len(image)==1:return original(image,**kwargs)
                outputs=[original(x,**kwargs) for x in image.split(1)]
                return {key:torch.cat([o[key] for o in outputs],dim=0) for key in outputs[0]}
            model.forward=types.MethodType(per_image,model)
            record['hypothesis_only']='per-image forward kernel schedule; unchanged complete-batch loss'
        record['checkpoint_parity']=checkpoint_parity(model,out)
        record['parity']=precision_micro_parity(model,out)
        record['status']='PASS'
    except BaseException as error:
        record.update(status='FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    write_json(out/'result.json',record)
    print(json.dumps(record),flush=True)
