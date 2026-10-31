"""Recover the original baseline config from its actual training log and package source."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
log=Path('D:/dataset/lesion_segmentation/实验日志_json/01_板块一_baseline.json')
metadata=json.loads(log.read_text(encoding='utf-8-sig').splitlines()[0])
assert metadata['seed']==42 and metadata['exp_name']=='hrdecoder_fcn_hr48_ddr_2048_split613_baseline'
original=metadata['config'];out=ROOT/'baseline_v1';out.mkdir(exist_ok=True)
(out/'original_config_from_log.py').write_text(original,encoding='utf-8')
remote='/home/studentuser001/paper_supplement_20261023'
overrides='''
# Resource amendment: two FP32 GPUs, global batch2; synchronized BN has
# the same global normalization population as the original single-GPU BN.
data['samples_per_gpu'] = 1
data['workers_per_gpu'] = 2
for _split in ('train', 'val', 'test'):
    data[_split]['data_root'] = '{r}/data/DDR_split_613'
model['pretrained'] = '{r}/assets/hrnetv2_w48-d2186c55.pth'
# The log captured backbone.pretrained after constructor mutation. Remove
# that duplicate alias before rebuilding; the segmentor sets the same path.
model['backbone'].pop('pretrained', None)
model['backbone']['norm_cfg'] = dict(type='SyncBN', requires_grad=True)
model['decode_head']['norm_cfg'] = dict(type='SyncBN', requires_grad=True)
work_dir = '{r}/baseline_v1/runs/baseline_seed42'
model['train_cfg']['work_dir'] = work_dir
log_config['interval'] = 20
find_unused_parameters = True
'''.format(r=remote)
(out/'baseline_reproduction_ddp2.py').write_text(original+overrides,encoding='utf-8')
files={p:Path('baseline_v1')/p.relative_to(out) for p in out.glob('*.py')}
repo=Path('D:/HRDecoder')
for p in repo.rglob('*.py'):
    if not set(p.relative_to(repo).parts)&{'data','work_dirs','.git','__pycache__'}:
        files[p]=Path('baseline_v1/repo')/p.relative_to(repo)
p=ROOT.parent/'HRDecoder_FinalDecision_20261005/execution/server_results/reports/data_identity.json'
files[p]=Path('baseline_v1/data_identity.json')
for name in ['baseline_train_entry.py','accept_baseline59_data.py','baseline59_queue.py']:
    p=ROOT/'scripts'/name
    if p.exists():files[p]=Path('baseline_v1')/name
bundle=ROOT/'transfer/baseline_sources.zip'
with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
    for p,name in sorted(files.items()):z.write(p,name.as_posix())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'status':'PREPARED_NOT_TRAINED','original_log':str(log),'log_sha256':sha(log),
         'original_config_sha256':hashlib.sha256(original.encode()).hexdigest(),
         'reproduction_config_sha256':sha(out/'baseline_reproduction_ddp2.py'),
         'bundle_sha256':sha(bundle),'source_files':len(files),
         'differences':['Original40000-step schedule, seed42, global batch2, loss and geometry retained',
                        'Two2080Ti ranks with1image/rank and SyncBN instead of single-GPU BN2',
                        'New training rerun; no promise of historical42.20 score or identical random order'],
         'no_validation_or_test_selection_change':True}
(ROOT/'evidence/baseline_reproduction_preparation.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'bundle':str(bundle),'bytes':bundle.stat().st_size,'source_files':len(files)}),flush=True)
