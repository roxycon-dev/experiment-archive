"""Verify coarse-hash candidates by local scene correspondence, without any lesion labels."""
import concurrent.futures
import json
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/external_acceptance'

def features(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert('L'))
    h,w=a.shape;scale=800/max(h,w)
    a=cv2.resize(a,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    kp,d=cv2.SIFT_create(nfeatures=1600).detectAndCompute(a,None)
    return {'points':np.float32([k.pt for k in kp]),'descriptors':d,'wh':list(a.shape[::-1])}

def main():
    data=json.loads((OUT/'image_dedup.json').read_text())
    paths=sorted({r[k]['image'] for r in data['candidates'] for k in ('a','b')})
    cv2.setNumThreads(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        cache=dict(zip(paths,pool.map(features,paths)))
    suspect=[]
    for i,row in enumerate(data['candidates']):
        a,b=(cache[row[k]['image']] for k in ('a','b'))
        if a['descriptors'] is None or b['descriptors'] is None:
            row['scene_check']={'status':'UNRESOLVED_INSUFFICIENT_FEATURES'};suspect.append(i);continue
        matcher=cv2.BFMatcher(cv2.NORM_L2)
        forward=matcher.knnMatch(a['descriptors'],b['descriptors'],k=2)
        reverse=matcher.knnMatch(b['descriptors'],a['descriptors'],k=2)
        rev={m.queryIdx:m.trainIdx for pair in reverse if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance}
        matches=[m for pair in forward if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance and rev.get(m.trainIdx)==m.queryIdx]
        check={'reciprocal_ratio_matches':len(matches),'status':'REJECTED_NO_GEOMETRIC_SCENE_MATCH'}
        if len(matches)>=8:
            p=np.float32([a['points'][m.queryIdx] for m in matches]);q=np.float32([b['points'][m.trainIdx] for m in matches])
            cv2.setRNGSeed(100303+i)
            H,inliers=cv2.findHomography(p,q,cv2.RANSAC,3.,maxIters=2000,confidence=.995)
            n=int(inliers.sum()) if inliers is not None else 0
            check.update(homography_inliers=n,inlier_fraction=n/len(matches))
            # Broad retinal shape alone is insufficient: repeated local vascular
            # texture must fit one geometry. Keep every borderline case for actual visual review.
            if n>=8 and n/len(matches)>=.35:
                check['status']='REQUIRES_VISUAL_SCENE_REVIEW';suspect.append(i)
        if row['exact_decoded_pixels']:
            check['status']='EXACT_DUPLICATE_REQUIRES_SPLIT_EXCLUSION';
            if i not in suspect:suspect.append(i)
        row['scene_check']=check
    data.update(status='PENDING_FINAL_SCENE_REVIEW' if suspect else 'PASS',
                scene_algorithm='SIFT1600 at aspect-preserved long-side800, reciprocal Lowe0.7; RANSAC3px; >=8 inliers and fraction>=0.35 reviewed; exact duplicates always retained',
                suspect_candidate_indexes=suspect,scene_verification_completed_at=datetime.now(timezone.utc).isoformat())
    (OUT/'image_dedup.scene_verified.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    sheets=[]
    for start in range(0,len(suspect),6):
        batch=suspect[start:start+6];canvas=Image.new('RGB',(1200,len(batch)*250),'black');d=ImageDraw.Draw(canvas)
        for rix,i in enumerate(batch):
            row=data['candidates'][i]
            for j,key in enumerate(('a','b')):
                im=Image.open(row[key]['image']).convert('RGB');im.thumbnail((500,210));canvas.paste(im,(j*600,rix*250+30))
                d.text((j*600,rix*250+3),str(i)+' '+row[key]['source']+' '+row[key]['split']+' '+row[key]['sid'],fill='white')
        path=OUT/f'duplicate_scene_review_{start//6:02d}.png';canvas.save(path);sheets.append(str(path))
    (OUT/'duplicate_scene_review_sheets.json').write_text(json.dumps({'suspect_candidate_indexes':suspect,'sheets':sheets},indent=2))
    print(json.dumps({'coarse_candidates':len(data['candidates']),'suspect_candidates':len(suspect),'sheets':sheets}),flush=True)

if __name__=='__main__':main()
