"""Bounded teacher-quality check followed by a transform-consistent pseudo cache.

Only grading train is eligible. Training-image precision is an in-sample
diagnostic, not validation. A class also needs adequate standalone val IoU.
MA has a higher precision requirement; failed classes remain entirely ignore.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.configure_server import sha
from tools.final_eval_locked import load_candidate,candidate_probability

def consensus(a,b,allowed,rng):
    target=np.zeros_like(a,dtype=np.uint8);valid=np.zeros_like(target)
    for c in range(4):
        if not allowed[c]:continue
        positive=(a[c]>=.8)&(b[c]>=.8)
        target[c]=positive;valid[c]=positive
        negative=np.flatnonzero(((a[c]<=.05)&(b[c]<=.05)).ravel())
        cap=max(512,int(positive.sum())*2)
        if len(negative)>cap:negative=rng.choice(negative,cap,replace=False)
        valid[c].ravel()[negative]=1
    return target,valid

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',type=Path,required=True)
    ap.add_argument('--auxiliary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    teacher=json.loads(a.teacher.read_text());member=dict(teacher['spec']['members'][0])
    member.update(checkpoint_sha256=sha(member['checkpoint']),config_sha256=sha(member['config']))
    identity=dict(checkpoint=member['checkpoint_sha256'],config=member['config_sha256'],auxiliary=sha(a.auxiliary))
    complete=a.out/'quality.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        if old['identity']!=identity:raise ValueError('Pseudo-cache teacher/pool identity changed')
        print('Reuse completed pseudo cache');return
    item=load_candidate(member);_,cfg,dc=item;torch.set_num_threads(4)
    rng=np.random.RandomState(42);root=Path(dc['ddr_root'])
    ids=sorted((root/'images/train').glob('*.jpg'),key=lambda p:sha_bytes(p.stem))[:32]
    pc=np.zeros((4,2),np.int64)
    def two(rgb):
        p=candidate_probability(rgb,item)
        q=candidate_probability(np.ascontiguousarray(rgb[:,::-1]),item)[:,:,::-1].copy()
        return p,q
    with torch.no_grad():
        for i,p in enumerate(ids):
            rgb=np.asarray(Image.open(p).convert('RGB'));gt=np.asarray(Image.open(root/'labels/train'/f'{p.stem}.png'))
            x,y=two(rgb)
            for c in range(4):
                positive=(x[c]>=.8)&(y[c]>=.8)&(gt!=255)
                pc[c,0]+=np.count_nonzero(positive&(gt==c+1));pc[c,1]+=positive.sum()
            print(f'teacher quality {i+1}/32',flush=True)
        precision=pc[:,0]/np.maximum(pc[:,1],1)
        allowed=[bool(pc[c,1]>=256 and precision[c]>=(.8 if c==3 else .65) and
                      teacher['alone_iou'][c]>=teacher['base_iou'][c]-.02) for c in range(4)]
        report=dict(identity=identity,training_consensus_precision=precision.tolist(),training_pixel_counts=pc.tolist(),
                    allowed=allowed,quality_note='In-sample train diagnostic; class eligibility also constrained by full val IoU.',
                    confidence_positive=.8,confidence_negative=.05,unknown='ignore',negative_cap='max(512,2*positive) per image/class')
        records=[]
        if any(allowed):
            pool=[json.loads(line) for line in a.auxiliary.read_text().splitlines() if line.strip()]
            for i,rec in enumerate(pool):
                image=a.auxiliary.parent/rec['image']
                if rec['split']!='train' or sha(image)!=rec['sha256']:raise ValueError('Auxiliary identity/split violation')
                rgb=np.asarray(Image.open(image).convert('RGB'));x,y=two(rgb)
                target,valid=consensus(x,y,allowed,rng)
                positives=int(target.sum())
                if positives>=16:
                    mask=a.out/(rec['sample_id']+'.npz')
                    np.savez_compressed(mask,target=target,valid=valid)
                    records.append(dict(sample_id=rec['sample_id'],image=str(image),mask=str(mask),
                                        positive_pixels=positives,image_sha256=rec['sha256'],mask_sha256=sha(mask),split='train'))
                if (i+1)%25==0:print(f'pseudo cache {i+1}/{len(pool)} usable={len(records)}',flush=True)
                if i==199 and (len(records)<40 or sum(r['positive_pixels'] for r in records)<2000):
                    report['stop_reason']='first_200_low_supervision_yield';break
        report.update(usable_images=len(records),qualified=any(allowed) and len(records)>=200,
                      status='complete',positive_pixels=sum(r['positive_pixels'] for r in records))
        manifest=a.out/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in records))
        report['manifest_sha256']=sha(manifest);complete.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))

def sha_bytes(value):
    import hashlib
    return hashlib.sha256(('quality42:'+value).encode()).hexdigest()

if __name__=='__main__':main()
