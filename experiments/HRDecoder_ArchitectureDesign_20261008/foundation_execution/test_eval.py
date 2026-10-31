"""One locked fixed40k endpoint evaluation; never select weights with test."""
import json
import os
from pathlib import Path
import time
import threading
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from PIL import Image
from train import ROOT, HERE, REPO, DATA, W48, WORK, REPORTS, identity, write_json, metrics_from_counts
from baseline import build_baseline, file_sha, install_legacy_bf16_interpolation
from model import SACDStudent, inference_probabilities
from data import evaluation_image

@torch.no_grad()
def main():
    assert torch.__version__=='2.0.1+cu117'
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    install_legacy_bf16_interpolation()
    material,digest=identity()
    marker=json.loads((WORK/'TRAINING_COMPLETE.json').read_text())
    val=json.loads((WORK/'val_40000.json').read_text())
    assert marker['completed']==40000 and marker['identity_SHA']==digest
    assert marker['durable_checkpoint_loaded_and_verified'] and val['images']==76
    assert val['mIoU_percent']>45.32183122738841
    assert not (WORK/'test_40000.json').exists(),'Test already evaluated; no repeated selection'
    rows_by_split=json.loads((ROOT/'metadata/data_identity.json').read_text())['splits']
    ids={name:{r['sid'] for r in rows_by_split[name]['samples']} for name in ('train','val','test')}
    assert (len(ids['train']),len(ids['val']),len(ids['test']))==(454,76,227)
    assert not (ids['train']&ids['test'] or ids['val']&ids['test'] or ids['train']&ids['val'])
    image_hashes={name:{r['image_sha256'] for r in rows_by_split[name]['samples']} for name in ('train','val','test')}
    assert not (image_hashes['train']&image_hashes['test'] or image_hashes['val']&image_hashes['test'] or image_hashes['train']&image_hashes['val'])
    lock={'locked_before_test_unix':time.time(),'checkpoint':'training_runs/foundation/last.pt',
          'completed':40000,'identity_SHA':digest,'checkpoint_SHA256':file_sha(WORK/'last.pt'),
          'selected_by':'pre-registered fixed40k endpoint','test_used_for_selection':False,
          'evaluation_code_SHA256':file_sha(HERE/'test_eval.py'),
          'custom_student_split_SID_disjoint':True,'existing_image_content_hashes_disjoint':True,
          'teacher_official_split_independence_claimed':False}
    lock_path=REPORTS/'LOCKED_TEST_CHECKPOINT.json'
    if lock_path.exists():
        previous=json.loads(lock_path.read_text())
        # A failed/incomplete test may be repaired on this exact locked model;
        # a successful evaluation cannot be repeated by this entry point.
        for key in ('completed','identity_SHA','checkpoint_SHA256','selected_by','evaluation_code_SHA256'):
            assert previous[key]==lock[key],'Locked test identity changed'
        lock=previous
    else:
        with lock_path.open('x') as out:json.dump(lock,out,indent=2)
    state=torch.load(WORK/'last.pt',map_location='cpu')
    assert state['identity_SHA']==digest and state['completed']==40000
    base,_=build_baseline(REPO,W48,verified_weight_sha=material['ImageNet_W48_SHA'])
    model=SACDStudent(base)
    model.load_state_dict(state['model'],strict=True);del state
    model.cuda().eval()
    images={p.stem:p for p in (DATA/'images/test').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
    assert set(images)==ids['test']
    counts=np.zeros((4,3),dtype=np.int64);started=time.monotonic()
    for index,row in enumerate(rows_by_split['test']['samples']):
        image=evaluation_image(images[row['sid']]).unsqueeze(0).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):output=model(image)
        prediction=(inference_probabilities(output,row['hw'])>.5).cpu().numpy()[0]
        # Labels are read only after deployment inference, for metric computation.
        with Image.open(DATA/'labels/test'/(row['sid']+'.png')) as source:label=np.asarray(source)
        valid=label!=255
        for c in range(4):
            gt=label==c+1;p=prediction[c]&valid
            counts[c]+=[int((p&gt).sum()),int((p&~gt).sum()),int((~p&gt&valid).sum())]
        del output,image,prediction
        if (index+1)%25==0:
            write_json(WORK/'test_progress.json',{'completed_images':index+1,'expected':227})
    result=metrics_from_counts(counts)
    result.update(images=227,split='test',completed=40000,seconds=time.monotonic()-started,
                  identity_SHA=digest,checkpoint_lock=lock,inference_teacher_or_GT=False,
                  completed_unix=time.time(),
                  TTA=False,online_H4=False,test_for_selection=False,
                  exceeds_fusion_test=result['mIoU_percent']>45.860383832668,
                  exceeds_fusion_val=True,student_split_SID_disjoint=True,
                  teacher_official_split_independence_claimed=False)
    write_json(WORK/'test_40000.json',result)
    print(json.dumps(result),flush=True)

def stop_at_deadline():
    write_json(WORK/'TEST_DEADLINE_STOP.json',{'status':'TEST_STOPPED_AT_USER_DEADLINE',
        'time_unix':time.time(),'checkpoint_preserved':True,'server_kept_running':True})
    # Terminate only this evaluation process, including a stuck native call.
    os._exit(124)

if __name__=='__main__':
    extension=json.loads((REPORTS/'DEADLINE_EXTENSION_ACTIVATED.json').read_text())
    assert extension['full40k_authorized'] and extension['status']=='ACTIVE'
    main()
