"""Image duplication audit with fundus cropping and anatomy verification.

Hash-only votes are poor on circular fundus photographs: shared black frames
create many false positives. Compare all image pairs using normalized anatomy
and edge correlation; mark uncertain pairs for review, exclude only very close
matches, and retain an image montage of the nearest cross-dataset pairs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

def descriptor(path):
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB'))
    bright=(rgb.max(axis=2)>20).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(bright,8)
    if n>1:
        x,y,w,h,_=stats[1+np.argmax(stats[1:,4])]
        if w>rgb.shape[1]/3 and h>rgb.shape[0]/3:
            rgb=rgb[y:y+h,x:x+w]
    gray=cv2.cvtColor(cv2.resize(rgb,(96,96)),cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    edge=gray-cv2.GaussianBlur(gray,(0,0),2)
    # Circular border excluded from anatomy correlations.
    yy,xx=np.mgrid[:96,:96]
    inside=((xx-47.5)**2+(yy-47.5)**2)<42**2
    def norm(a):
        a=a[inside]
        a=a-a.mean()
        return a/max(np.linalg.norm(a),1e-9)
    return norm(gray),norm(edge),cv2.resize(rgb,(192,192))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--base-dir',required=True,type=Path)
    ap.add_argument('--ddr-root',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    records=[json.loads(x) for x in a.manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    external=[(r['sample_id'],a.base_dir/r['image']['path']) for r in records]
    ddr=[(str(p.relative_to(a.ddr_root)),p) for p in sorted(a.ddr_root.rglob('*.jpg'))]
    ds={str(p):descriptor(p) for _,p in external+ddr}
    matches=[]
    excluded=set()
    # Compare train auxiliary images to every DDR split, plus each other.
    for i,(sid,p) in enumerate(external):
        g,e,_=ds[str(p)]
        choices=[]
        for name,q in ddr+external[:i]:
            qg,qe,_=ds[str(q)]
            gc=float(g@qg); ec=float(e@qe)
            score=(gc+ec)/2
            choices.append((score,name,q,gc,ec))
            if gc>=.985 and ec>=.96:
                excluded.add(sid)
                matches.append(dict(external_id=sid,other=name,gray_corr=gc,edge_corr=ec,action='exclude'))
        score,name,q,gc,ec=max(choices)
        matches.append(dict(external_id=sid,other=name,gray_corr=gc,edge_corr=ec,
                            action='review' if gc>=.94 and ec>=.85 else 'nearest_only'))
    kept=[r for r in records if r['sample_id'] not in excluded]
    a.out.mkdir(parents=True,exist_ok=True)
    report=dict(ddr_images_compared=len(ddr),external_images=len(external),excluded=sorted(excluded),
        high_similarity_review=[x for x in matches if x['action']=='review'],matches=matches,
        limitation='Image-level audit; unknown patients remain unknown; rotation and arbitrary deformation are not exhaustively searched')
    (a.out/'duplicates.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    clean=a.manifest.with_name('external_train_dedup.jsonl')
    clean.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept),encoding='utf-8')
    tops=sorted([r for r in matches if r['other'] in dict(ddr)],
                key=lambda r:r['gray_corr']+r['edge_corr'],reverse=True)[:12]
    canvas=Image.new('RGB',(800,232*len(tops)),(245,245,245))
    draw=ImageDraw.Draw(canvas)
    path_by_id=dict(external+ddr)
    for i,r in enumerate(tops):
        for j,key in enumerate(['external_id','other']):
            thumb=Image.fromarray(ds[str(path_by_id[r[key]])][2])
            canvas.paste(thumb,(j*400,i*232))
            draw.text((j*400,i*232+195),r[key],fill=(20,20,20))
        draw.text((0,i*232+213),f"gray={r['gray_corr']:.3f}; edges={r['edge_corr']:.3f}",fill=(20,20,20))
    canvas.save(a.out/'nearest_pairs.jpg')
    print(json.dumps(dict(n_ddr=len(ddr),n_external=len(external),n_excluded=len(excluded),
                         n_review=len(report['high_similarity_review']),clean_manifest=str(clean))))

if __name__=='__main__':
    main()
