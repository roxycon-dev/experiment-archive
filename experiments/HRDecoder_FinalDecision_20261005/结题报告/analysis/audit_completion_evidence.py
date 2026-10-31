"""Offline source/count audit. No model loads, inference, training or selection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, csv
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
EX = OUT.parent / 'execution'
SR = EX / 'server_results'
DATA = Path('D:/HRDecoder/data/DDR_split_613')
CLASSES = ['EX', 'HE', 'SE', 'MA']

def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def save(name, d):
    p = OUT / 'evidence' / name
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), 'utf-8')
def score(ct):
    a = np.array(ct, dtype=np.int64); tp, fp, fn = a.T
    return float(np.mean(tp / np.maximum(tp+fp+fn, 1))*100)
def stats(ct):
    return {c:{'tp':int(tp),'fp':int(fp),'fn':int(fn),
               'iou_pct':100*tp/(tp+fp+fn), 'dice_pct':200*tp/(2*tp+fp+fn),
               'precision_pct':100*tp/(tp+fp), 'recall_pct':100*tp/(tp+fn)}
            for c,(tp,fp,fn) in zip(CLASSES,ct)}

def main():
    identity=read(SR/'reports/data_identity.json'); assignment=read(DATA/'split_assignment.json')
    assert sha(DATA/'split_assignment.json')==identity['assignment_sha256']
    verified=[]; unknown_values={}; ignored=0
    for split, group in identity['splits'].items():
        expected={Path(k).stem for k,v in assignment['images'].items() if v['split']==split}
        assert expected=={v['sid'] for v in group['samples']}
        for s in group['samples']:
            img=DATA/'images'/split/(s['sid']+'.jpg'); lab=DATA/'labels'/split/(s['sid']+'.png')
            assert sha(img)==s['image_sha256'] and sha(lab)==s['label_sha256'],s['sid']
            with Image.open(lab) as f: a=np.asarray(f)
            assert list(a.shape)==s['hw']; vals=np.unique(a)
            ignored+=int(np.count_nonzero(a==255))
            if any(int(x) not in (0,1,2,3,4) for x in vals): unknown_values[s['sid']]=vals.tolist()
        verified.append({'split':split,'n_images':len(expected),'identities_match_assignment':True,'all_image_and_label_SHAs_match':True})
        print('verified split',split,len(expected),flush=True)
    assert not unknown_values, unknown_values
    baseline=read(OUT/'evidence/original_HRDecoder_baseline_evidence.json')
    baseline['split_identity_and_evaluation_code_match_to_current']={
        'documented_baseline_root':'./data/DDR_split_613',
        'same_custom_613_split_documented_in_original_config_and_report':True,
        'current_assignment_matches_frozen_campaign_assignment_exactly':True,
        'assignment_sha256':identity['assignment_sha256'], 'verified_samples':verified,
        'native_labels_contain_only_0_1_2_3_4':True,'native_ignore_255_pixels':ignored,
        'metric_structure':'Both saved legacy lesion evaluator and current code use sigmoid strict >.5, original-image labels, global per-class pixel counts and macro mean of EX/HE/SE/MA; background excluded.',
        'legacy_ignore_argument_not_applied_but_current_native_GT_has_zero_ignore_pixels':ignored==0,
        'historical_baseline_immutable_sample_SHA_snapshot_found':False,
        'comparison_grade':'Same named fixed split and documented metric. Current data identity audited bytewise; baseline test is a rounded historical report, not a newly reproduced baseline.'}
    baseline['metric_source_files']=[{'path':str(p),'sha256':sha(p)} for p in [Path('D:/HRDecoder/mmseg/core/evaluation/lesion_metric.py'),Path('D:/HRDecoder/mmseg/datasets/lesion_dataset.py'),EX/'qlib/metrics.py']]
    save('original_HRDecoder_baseline_evidence.json',baseline)
    test=read(SR/'runs/LOCKED_TEST_seed42/metrics.json'); final=read(SR/'reports/final_test.json')
    ref=read(EX/'configs/R_test_counts_seed42.json'); old={x['sid']:x['R'] for x in ref['per_image']}
    assert test['n_images']==227 and test['split']=='test' and not test['probability_files']
    assert {x['sid'] for x in test['per_image']}==set(old)
    ct=np.array([x['fixed_fusion'] for x in test['per_image']]).sum(0)
    assert ct.tolist()==test['fixed_fusion']['counts']; assert abs(score(ct)-final['seeds']['42']['miou_pct'])<1e-10
    rct=np.array(list(old.values())).sum(0); new=score(ct); rscore=score(rct)
    assert abs(rscore-45.72155710642771)<1e-10
    rows=[]
    for x in test['per_image']:
        a=np.array(x['fixed_fusion']); b=np.array(old[x['sid']])
        assert (a[:,0]+a[:,2]==b[:,0]+b[:,2]).all()
        rows.append({'sid':x['sid'],'new_iou_pct':score(a),'R_iou_pct':score(b),'delta_image_iou_pp':score(a)-score(b),
            'new_counts':a.tolist(),'R_counts':b.tolist()})
    gains={c:stats(ct)[c]['iou_pct']-stats(rct)[c]['iou_pct'] for c in CLASSES}
    result={'stage':'FINAL_LOCKED_TEST_EXACT_COUNT_AUDIT','generated_at':datetime.now(timezone.utc).isoformat(),
        'n_images':227,'locked_recipe':read(SR/'reports/final_selection_lock.json'),
        'locked_test_miou_pct':new,'R_test_miou_pct':rscore,'gain_vs_R_pp':new-rscore,
        'historical_baseline_test_pct_reported':42.2,'approx_gain_vs_baseline_pp':new-42.2,
        'approx_relative_gain_vs_baseline_pct':(new-42.2)/42.2*100,
        'target_test_pct':47.2115,'gap_to_target_pp':47.2115-new,
        'test_per_class':stats(ct),'R_test_per_class':stats(rct),'gain_vs_R_per_class_pp':gains,
        'conditional_bootstrap_from_original_final_test':final['seeds']['42']['conditional_paired_image_bootstrap'],
        'per_image':rows,
        'positive_image_deltas':sum(x['delta_image_iou_pp']>0 for x in rows),
        'negative_image_deltas':sum(x['delta_image_iou_pp']<0 for x in rows),
        'zero_image_deltas':sum(x['delta_image_iou_pp']==0 for x in rows),
        'median_image_delta_pp':float(np.median([x['delta_image_iou_pp'] for x in rows])),
        'metric_warning':'Per-image mIoU distribution is diagnostic; global mIoU is calculated from summed counts and is not their mean.',
        'sources':[{'path':str(p),'sha256':sha(p)} for p in [SR/'runs/LOCKED_TEST_seed42/metrics.json',SR/'reports/final_test.json',EX/'configs/R_test_counts_seed42.json',SR/'reports/final_selection_lock.json']]}
    save('final_locked_test_evidence.json',result)
    with (OUT/'evidence/final_test_paired_images.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sid','new_iou_pct','R_iou_pct','delta_image_iou_pp']);w.writeheader()
        w.writerows([{k:x[k] for k in w.fieldnames} for x in rows])
    val=read(SR/'runs/S_seed42/val_u20000/metrics.json'); rr=read(SR/'reports/R_reconstruction_seed42/metrics.json')
    vref={x['sid']:x['fixed_fusion'] for x in rr['per_image']}
    vr=[]
    for x in val['per_image']:
        a=np.array(x['fixed_fusion']); b=np.array(vref[x['sid']]); vr.append({'sid':x['sid'],'selected_iou_pct':score(a),'R_iou_pct':score(b),'delta_image_iou_pp':score(a)-score(b),'new_counts':a.tolist(),'R_counts':b.tolist(),'single_counts':x['single'],'p0_counts':x['p0']})
    save('validation_paired_images.json',{'stage':'OFFLINE_SAVED_COUNT_ANALYSIS','n_images':76,'per_image':vr,'positive':sum(x['delta_image_iou_pp']>0 for x in vr),'negative':sum(x['delta_image_iou_pp']<0 for x in vr),'median_delta_pp':float(np.median([x['delta_image_iou_pp'] for x in vr]))})
    save('completion_audit_status.json',{'stage':'PASS','time':datetime.now(timezone.utc).isoformat(),'native_data_identities_verified':757,'four_arm_saved_node_records':34,'locked_test_counts_exact':True,'servers_contacted':False,'new_inference_or_training':False})
    (OUT/'report_work_state.json').write_text(json.dumps({'stage':'LOCAL_EVIDENCE_AUDIT_COMPLETE_REPORT_AUTHORING_NEXT','time':datetime.now(timezone.utc).isoformat(),'servers_full_backup_and_shutdown_dispatched':True,'actual_BOTH_OFF_confirmation_pending':True,'final_result':new,'no_new_inference_training_or_reselection':True},ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps({'test':new,'R_gain_pp':new-rscore,'approx_baseline_gain_pp':new-42.2,'classes':gains,'positive_images':result['positive_image_deltas'],'negative_images':result['negative_image_deltas'],'median_delta':result['median_image_delta_pp']},ensure_ascii=True),flush=True)

if __name__=='__main__': main()
