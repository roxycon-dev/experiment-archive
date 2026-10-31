"""Verify and preserve the user-recovered historical split613 baseline without training."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('C:/Users/GuanXingRen/Downloads/Compressed/工程文件')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    torch.set_num_threads(2)
    p=SOURCE/'server_checkpoints/01_板块一_baseline_best_iter32000.pth'
    digest=sha(p)
    assert digest=='446fc7b598e9680a1470059d2aba3a6f9c2222b66658a5eccef6059b1362850f'
    x=torch.load(p,map_location='cpu',weights_only=False,mmap=True);m=x['meta']
    assert m['seed']==42 and m['iter']==31999
    assert m['exp_name']=='hrdecoder_fcn_hr48_ddr_2048_split613_baseline'
    assert m['hook_msgs']['best_ckpt'].endswith('/best_mIoU_iter_32000.pth')
    assert abs(m['hook_msgs']['best_score']-.39390182190651146)<1e-14
    log=SOURCE/'实验日志_json/01_板块一_baseline.json'
    old=Path('D:/dataset/lesion_segmentation/实验日志_json/01_板块一_baseline.json')
    assert log.read_bytes()==old.read_bytes()
    first=json.loads(log.read_text(encoding='utf-8').splitlines()[0])
    assert first['config']==m['config']
    input_config=SOURCE/'01_板块一_baseline实验配置.py'
    assert input_config.read_bytes()==Path('D:/dataset/lesion_segmentation/01_板块一_baseline实验配置.py').read_bytes()
    state=x['state_dict'];assert len(state)==1844
    assert all(not t.is_floating_point() or bool(torch.isfinite(t).all()) for t in state.values())
    slots=x['optimizer']['state'];floating_slots=0
    for values in slots.values():
        for tensor in values.values():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                assert bool(torch.isfinite(tensor).all());floating_slots+=1
    out=ROOT/'evidence/recovered_original_baseline';out.mkdir(parents=True,exist_ok=True)
    sources=[]
    for source in [log,input_config]:
        dest=out/source.name
        if dest.exists():assert dest.read_bytes()==source.read_bytes()
        else:shutil.copyfile(source,dest)
        sources.append({'source':str(source),'local':str(dest),'sha256':sha(dest)})
    config=out/'original_checkpoint_config.py'
    config.write_text(m['config'],encoding='utf-8')
    inference_config=out/'inference_config.py'
    inference_config.write_text(m['config']+"\n# Restore fields consumed by the historical constructor; inference uses fixed whole-image processing.\nmodel['hr_settings']['scale_ratio']=(.75,1.25)\nmodel['hr_settings']['crop_num']=4\n",encoding='utf-8')
    full=ROOT/'models/recovered_hrdecoder_baseline_u32000.pth';full.parent.mkdir(exist_ok=True)
    if full.exists():assert sha(full)==digest
    else:shutil.copyfile(p,full)
    assert sha(full)==digest
    compact=ROOT/'models/recovered_hrdecoder_baseline_inference.pth'
    if not compact.exists():
        torch.save({'state_dict':state,'source_checkpoint_sha256':digest,'selected_update':32000},compact)
    check=torch.load(compact,map_location='cpu',weights_only=False,mmap=True)
    assert set(check['state_dict'])==set(state)
    assert all(torch.equal(check['state_dict'][k],v) for k,v in state.items())
    (ROOT/'evidence/recovered_original_baseline_export.json').write_text(json.dumps({
        'status':'PASS','source_checkpoint_sha256':digest,'compact_sha256':sha(compact),
        'compact_bytes':compact.stat().st_size,'all_state_tensors_bitwise_equal':True,'tensor_count':len(state)
    },indent=2),encoding='utf-8')
    receipt={'status':'PASS_RECOVERED_HISTORICAL_BASELINE_IDENTITY_AND_LOCAL_BACKUP',
        'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_path':str(p),'source_checkpoint_sha256':digest,'source_bytes':p.stat().st_size,
        'local_checkpoint':str(full),'local_checkpoint_sha256':sha(full),
        'compact_checkpoint':str(compact),'compact_sha256':sha(compact),
        'selected_update':32000,'historical_validation_mIoU':m['hook_msgs']['best_score'],
        'checkpoint_config_matches_original_log_exactly':True,'state_tensors_finite':len(state),
        'optimizer_floating_slots_finite':floating_slots,'source_files':sources,
        'inference_config':str(inference_config),'inference_config_sha256':sha(inference_config),
        'reported_historical_test_mIoU_pct':42.20,'test_score_verified_by_new_inference':False,
        'retraining_needed_to_recover_baseline':False,
        'historical_E4_parent_edge_established_by_finding_this_file':False,
        'overall_campaign_complete':False}
    (ROOT/'evidence/recovered_original_baseline_identity.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ['status','selected_update','historical_validation_mIoU','source_checkpoint_sha256','state_tensors_finite','optimizer_floating_slots_finite']}))

if __name__=='__main__':main()
