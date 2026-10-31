"""Restore constructor-consumed original baseline settings; no test-driven recipe search."""
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

ROOT=Path(__file__).resolve().parents[1]
original_path=Path('D:/dataset/lesion_segmentation/01_板块一_baseline实验配置.py')
saved=runpy.run_path(str(original_path))
settings=saved['model']['hr_settings']
assert settings['scale_ratio']==(.75,1.25) and settings['crop_num']==4
audit=json.loads((ROOT/'evidence/baseline_initialization_and_constructor_audit.json').read_text())
assert audit['all_backbone_tensors_equal_to_pretrained_after_full_init']
assert audit['scale_ratio_missing_from_both_post_constructor_configs']
base=ROOT/'baseline_v2';base.mkdir(exist_ok=True)
source=(ROOT/'baseline_v1/original_config_from_log.py').read_text(encoding='utf-8')
remote='/home/studentuser001/paper_supplement_20261023'
patch=f'''
# Technical correction: constructor pop() removed scale_ratio/crop_num from
# the saved post-construction JSON. Restore them from the original input config.
model['hr_settings']['scale_ratio'] = (0.75, 1.25)
model['hr_settings']['crop_num'] = 4
data['samples_per_gpu'] = 1
data['workers_per_gpu'] = 2
for _split in ('train', 'val', 'test'):
    data[_split]['data_root'] = '{remote}/data/DDR_split_613'
model['pretrained'] = '{remote}/assets/hrnetv2_w48-d2186c55.pth'
model['backbone'].pop('pretrained', None)
model['backbone']['norm_cfg'] = dict(type='SyncBN', requires_grad=True)
model['decode_head']['norm_cfg'] = dict(type='SyncBN', requires_grad=True)
work_dir = '{remote}/baseline_v2/runs/baseline_seed42'
model['train_cfg']['work_dir'] = work_dir
log_config['interval'] = 20
find_unused_parameters = True
'''
config=base/'baseline_reproduction_ddp2.py'
assert not config.exists() or config.read_text(encoding='utf-8')==source+patch
config.write_text(source+patch,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bundle=ROOT/'transfer/baseline_v2_sources.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
    z.write(config,'baseline_v2/baseline_reproduction_ddp2.py')
    z.write(original_path,'baseline_v2/original_saved_input_config.py')
    z.write(ROOT/'baseline_v1/original_config_from_log.py','baseline_v2/post_constructor_logged_config.py')
    for p in sorted((ROOT/'local_baseline_profile_source/baseline_v1/repo').rglob('*.py')):
        z.write(p,'baseline_v2/repo/'+p.relative_to(ROOT/'local_baseline_profile_source/baseline_v1/repo').as_posix())
receipt={'status':'FROZEN_TECHNICAL_CORRECTION_BEFORE_BASELINE_V2_TRAINING',
    'reason':'Baseline_v1 reconstructed from post-constructor log omitted original random HR crop scale (.75,1.25)',
    'original_saved_config':str(original_path),'original_saved_config_sha256':sha(original_path),
    'initialization_audit_sha256':sha(ROOT/'evidence/baseline_initialization_and_constructor_audit.json'),
    'corrected_config_sha256':sha(config),'corrected_source_bundle_sha256':sha(bundle),
    'restored_hr_scale_ratio':[.75,1.25],'restored_crop_num':4,
    'pretraining':'Original ImageNet HRNet-W48;1830backbone tensors bitwise equal after full initialization',
    'training':'Fresh seed42,original40k SGD/poly/1500warmup;all10validationnodes;originalhookbest;no test selection',
    'resource_difference_retained':'Two2080Ti FP32 globalbatch2 SyncBN;not bitwise original single4090 BN2 replay',
    'baseline_v1_status':'CONFIG_MISMATCH_NOT_ACCEPTED_AS_FAITHFUL_ORIGINAL_BASELINE;retain all weights/logs/predictions/costs',
    'training_attempt_budget':'Four adaptation runs plus two baseline40k attempts,including the rejected v1 attempt; disclose both',
    'adaptation_protocol_changed':False,'threshold_or_fusion_search':False,
    'expected_historical_score':'No guaranteed42.20; exact cause of score gap cannot be attributed solely to the restored setting',
    'overall_campaign_complete':False}
(ROOT/'plan/baseline_reconstruction_amendment.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
(ROOT/'evidence/baseline_v2_preparation.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'status':receipt['status'],'config_sha256':receipt['corrected_config_sha256'],'bundle_sha256':sha(bundle)}))
