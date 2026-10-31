"""Numerical core-code check against the Git revision named by the recovered checkpoint.

Synthetic small-geometry diagnostic, same BN mode and fixed RNG; zero optimizer updates.
This does NOT test distributed stochastic training or guarantee a retrained test score.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime_deps/legacy_profile'))
sys.path.insert(0,str(ROOT/'local_baseline_profile_source/baseline_v1/repo'))
from mmcv import Config
from mmseg.models import build_segmentor

def main():
    torch.set_num_threads(2)
    rel='mmseg/models/segmentors/HRDecoder.py'
    original=subprocess.check_output(['git','-C','D:/HRDecoder','show','cfc6c88:'+rel]).decode('utf-8')
    evidence=ROOT/'evidence/baseline_core_code_audit';evidence.mkdir(exist_ok=True)
    (evidence/'HRDecoder_at_checkpoint_revision.py').write_text(original,encoding='utf-8')
    tree=ast.parse(original)
    for node in tree.body:
        if isinstance(node,ast.ClassDef):node.decorator_list=[]
    module=types.ModuleType('mmseg.models.segmentors._original_baseline_audit')
    module.__package__='mmseg.models.segmentors'
    exec(compile(tree,'HRDecoder_at_checkpoint_revision.py','exec'),module.__dict__)
    identity=json.loads((ROOT/'evidence/recovered_original_baseline_identity.json').read_text(encoding='utf-8'))
    cfg=Config.fromfile(identity['inference_config']);settings=copy.deepcopy(cfg.model)
    settings['pretrained']=None;settings['backbone'].pop('pretrained',None)
    settings['hr_settings']['hr_scale']=(128,128)
    torch.manual_seed(42);current=build_segmentor(copy.deepcopy(settings))
    args=copy.deepcopy(settings);args.pop('type')
    torch.manual_seed(42);historical=module.HRDecoder(**args)
    saved=torch.load(identity['compact_checkpoint'],map_location='cpu')
    for model in (current,historical):model.load_state_dict(saved['state_dict'],strict=True);model.train()
    assert current.detail_branch is None and current.detail_head is None
    assert current.decode_head.context is None and current.fuse_mode=='simple' and current.crop_mode=='random'
    assert current.scale_ratio==historical.scale_ratio==(.75,1.25)
    assert current.crop_num==historical.crop_num==4
    torch.manual_seed(123);image=torch.randn(2,3,256,256)
    truth=torch.randint(0,5,(2,1,256,256),dtype=torch.long)
    metadata=[dict(img_shape=(256,256,3),ori_shape=(256,256,3),pad_shape=(256,256,3),flip=False)]*2
    outputs=[];gradients=[]
    for model in (historical,current):
        torch.manual_seed(20261024);np.random.seed(20261024)
        losses=model.forward_train(image,metadata,truth)
        loss=sum(value.mean() for key,value in losses.items() if 'loss' in key)
        assert bool(torch.isfinite(loss));loss.backward()
        outputs.append({k:float(v.detach().mean()) for k,v in losses.items()})
        gradients.append({k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None})
    assert outputs[0].keys()==outputs[1].keys()
    loss_diff={k:abs(outputs[0][k]-outputs[1][k]) for k in outputs[0]}
    assert gradients[0].keys()==gradients[1].keys()
    maxdiff=max(float((gradients[0][k]-gradients[1][k]).abs().max()) for k in gradients[0])
    assert max(loss_diff.values())<1e-6 and maxdiff<1e-6
    result={'status':'PASS_BASELINE_CORE_LOSS_AND_GRADIENT_NUMERICAL_CHECK',
        'reference':'Git cfc6c88 named by original checkpoint metadata; not proof of absence of uncommitted historical edits',
        'reference_file_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'actual_source_file':str(ROOT/'local_baseline_profile_source/baseline_v1/repo'/rel),
        'optional_detail_context_learned_fusion_SE_slide_modules_disabled':True,
        'historical_losses':outputs[0],'current_losses':outputs[1],'absolute_loss_differences':loss_diff,
        'maximum_absolute_gradient_difference':maxdiff,'parameter_gradients_checked':len(gradients[0]),
        'optimizer_updates':0,'GPU_used':False,'geometry':'Synthetic2x256 input,128 HR scale; diagnostic only',
        'normalization':'Both BN for core-code comparison, not distributed SyncBN equivalence',
        'limits':'Does not establish matched stochastic sampling, rank RNG, distributed gradients, or identical full-training test scores'}
    (ROOT/'evidence/baseline_core_equivalence_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
