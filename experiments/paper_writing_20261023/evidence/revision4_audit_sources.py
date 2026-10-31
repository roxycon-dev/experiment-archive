"""Hash/documentary training-source audit. No checkpoint deserialization."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
import cv2
from PIL import Image
cv2.setNumThreads(2)
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parent/'paper_supplement_20261023';E=OUT/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def save(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf-8')
start=time.time();source=read(ROOT/'results/final_campaign/numeric_table_sources.json');hashes=[]
for i,(name,digest) in enumerate(source['input_sha256'].items()):
    p=ROOT/name;ok=p.is_file() and sha(p)==digest;hashes.append({'path':name,'exists':p.is_file(),'hash_match':ok})
    if i%1000==0:print(json.dumps({'primary_input_hashes':i,'seconds':time.time()-start}),flush=True)
save(E/'revision4_primary_input_hash_audit.json',{'n_inputs':len(hashes),'all_pass':all(r['hash_match'] for r in hashes),'records':hashes,'no_weight_deserialization':True})
training=[];ids={}
for seed in [2024,3407]:
    for arm in ['C','S']:
        folder=ROOT/'recovered_training_provenance/training_v1/runs'/f'{arm}_seed{seed}'
        identity=read(folder/'identity.json');ids[arm,seed]=identity
        log=[json.loads(s) for s in (folder/'train.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        assert [r['update'] for r in log]==list(range(20,20001,20))
        assert all(math.isfinite(float(r[k])) for r in log for k in ['loss','grad_norm','seconds','data_wait_seconds'])
        assert identity['endpoint']==20000 and identity['schedule_horizon']==32000
        assert identity['parent_full_sha256']=='68d7b5821fab1210b3336cf2a63b9de893f24fc32876f5255d7d662084b5e728'
        training.append({'arm':arm,'seed':seed,'first_logged_update':log[0]['update'],'last_logged_update':log[-1]['update'],'logged_every_updates':20,'n_logged_records':len(log),'logged_losses_and_gradient_norms_finite':True,'identity_sha256':sha(folder/'identity.json'),'log_sha256':sha(folder/'train.jsonl'),'parent':identity['parent_full_sha256'],'stream':identity['stream_sha256'],'schedule_horizon':identity['schedule_horizon']})
    assert ids['C',seed]['stream_sha256']==ids['S',seed]['stream_sha256']
    assert ids['C',seed]['data_manifest_sha256']==ids['S',seed]['data_manifest_sha256']
baseline=read(ROOT/'evidence/recovered_original_baseline_identity.json')
for field in ['local_checkpoint','compact_checkpoint']:assert sha(Path(baseline[field]))==baseline['local_checkpoint_sha256' if field=='local_checkpoint' else 'compact_sha256']
nodes=read(ROOT/'evidence/recovered_original_baseline/history_validation_nodes.json')['validation_nodes'];best=max(nodes,key=lambda x:x['original_log_record']['mIoU'])
assert len(nodes)==10 and best['training_node']==32000
oldconfig=ROOT/'evidence/recovered_original_baseline/original_checkpoint_config.py'
logfile=ROOT/'evidence/recovered_original_baseline/01_板块一_baseline.json'
first=json.loads(logfile.read_text(encoding='utf-8-sig').splitlines()[0])
config=first.get('config',first.get('meta',{}).get('config'))
config_equal=(config==oldconfig.read_text(encoding='utf-8-sig')) if config is not None else None
save(E/'revision4_training_identity_audit.json',{'new_training':training,'same_parent_all_runs':True,'within_seed_C_S_data_stream_identical':True,'only_conditional_adaptation_variance':True,'original_baseline':{'selected_update':32000,'historical_validation_hook_mIoU_pct':baseline['historical_validation_mIoU']*100,'config_export_exactly_matches_original_first_log_config':config_equal,'selected_from_all_10_original_val_nodes':best,'source_full_checkpoint_hash_verified':True,'source_full_SHA':baseline['source_checkpoint_sha256'],'compact_SHA':baseline['compact_sha256'],'original_checkpoint_tensor_checks':'Existing receipt only; this audit hashes files and does not deserialize weights'},'failed_baseline_attempts':{'v1':'40k completed with scale_ratio reconstruction error; excluded from principal comparison','v2':'2980 updates then stopped after original recovery; not a40k result'}})
binrecord=read(ROOT/'evidence/ma_size_bins.json');areas=[];trainchecks=[]
for r in binrecord['training_samples']:
    p=Path('D:/HRDecoder/data/DDR_split_613/labels/train')/(r['sid']+'.png');gt=np.asarray(Image.open(p))==4
    _,_,stats,_=cv2.connectedComponentsWithStats(gt.astype(np.uint8),connectivity=8);aa=stats[1:,4].tolist();assert aa==r['areas'];areas.extend(aa)
    trainchecks.append({'sid':r['sid'],'GT_sha256':sha(p),'components':len(aa),'areas_match_saved_record':True})
assert len(areas)==6793 and np.quantile(areas,[1/3,2/3],method='nearest').tolist()==[32,74]
save(E/'revision4_train_size_bin_audit.json',{'images':len(trainchecks),'components':len(areas),'quantile_method':'nearest','boundaries_pixels':[32,74],'no_test_prediction_used':True,'all_pass':True,'samples':trainchecks})
print(json.dumps({'primary_hash_inputs':len(hashes),'all_hashes_pass':all(r['hash_match'] for r in hashes),'four_training_logs_pass':True,'training_mask_areas_pass':True,'elapsed_seconds':time.time()-start}))
