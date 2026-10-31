"""CPU-only verification of actual pretrained parameters and constructor-consumed settings."""
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import sys
import cv2
import torch
from mmcv import Config

BASE=Path('/home/studentuser001/paper_supplement_20261023/baseline_v1')
sys.path.insert(0,str(BASE/'repo'))
from mmseg.models.builder import build_train_model

torch.set_num_threads(1);cv2.setNumThreads(1);torch.manual_seed(42)
cfg=Config.fromfile(str(BASE/'baseline_reproduction_ddp2.py'))
passed_hr=copy.deepcopy(dict(cfg.model.hr_settings));asset=Path(cfg.model.pretrained)
with (BASE/'initialization_audit_stdout.log').open('w') as handle,contextlib.redirect_stdout(handle),contextlib.redirect_stderr(handle):
    model=build_train_model(cfg,train_cfg=cfg.get('train_cfg'),test_cfg=cfg.get('test_cfg'))
    model.init_weights()
source=torch.load(str(asset),map_location='cpu');source=source.get('state_dict',source)
state=model.backbone.state_dict();missing=[k for k in state if k not in source]
different=[k for k in state if k in source and not torch.equal(state[k],source[k])]
receipt={'status':'PASS' if not missing and not different else 'FAIL',
    'asset_path':str(asset),'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'asset_bytes':asset.stat().st_size,
    'backbone_state_tensors':len(state),'missing_backbone_tensors':missing,
    'different_after_full_model_init':different,'classification_only_source_tensors':len(set(source)-set(state)),
    'passed_hr_settings':passed_hr,'post_constructor_hr_settings':dict(cfg.model.hr_settings),
    'effective_scale_ratio':model.scale_ratio,'effective_crop_num':model.crop_num,
    'all_backbone_tensors_equal_to_pretrained_after_full_init':not missing and not different,
    'training_updates':0,'GPU_used':False}
# Rebuild only to demonstrate why a post-constructor log cannot recover popped settings.
del model,state
cfg2=Config.fromfile(str(BASE/'baseline_reproduction_ddp2.py'))
cfg2.model.hr_settings.update(scale_ratio=(.75,1.25),crop_num=4)
with (BASE/'constructor_audit_stdout.log').open('w') as handle,contextlib.redirect_stdout(handle),contextlib.redirect_stderr(handle):
    restored=build_train_model(cfg2,train_cfg=cfg2.get('train_cfg'),test_cfg=cfg2.get('test_cfg'))
receipt['restored_candidate_scale_ratio']=restored.scale_ratio
receipt['restored_candidate_post_constructor_hr_settings']=dict(cfg2.model.hr_settings)
receipt['scale_ratio_missing_from_both_post_constructor_configs']='scale_ratio' not in cfg.model.hr_settings and 'scale_ratio' not in cfg2.model.hr_settings
path=BASE/'initialization_and_constructor_audit.json';path.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
