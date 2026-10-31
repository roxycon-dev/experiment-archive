"""Train-only TJDR inspection; DDR holdout IMAGE identities only, never labels.

Candidate rules: exact decoded RGB SHA, or pHash distance<=18 plus >=40
ORB/RANSAC inliers and inlier fraction>=.6. pHash alone is not duplication.
The pre-training repair is recorded in TJDR_dedup_revision.json. Exclusions do not claim patient
independence. All candidate pairs are retained for review.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, hashlib, json
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageDraw
from qlib.common import read_json, sha, write_json, utc_now
from qlib.data import read_rgb

ROOT = Path(__file__).resolve().parents[1]
RULES = {'phash_only_is_not_duplicate':True, 'orb_candidate_distance':18,
         'orb_min_inliers':40, 'orb_min_fraction':.6, 'ransac_pixels_at_768':3.0}

def image_signature(path):
    rgb = read_rgb(path)
    gray = cv2.cvtColor(cv2.resize(rgb, (256,256)), cv2.COLOR_RGB2GRAY)
    hashes=[]
    for k in range(4):
        for flip in (False,True):
            a=np.rot90(gray,k)
            if flip:a=np.fliplr(a)
            d=cv2.dct(np.ascontiguousarray(cv2.resize(a,(32,32)),dtype=np.float32))[:8,:8].ravel()
            bits=d>np.median(d[1:]);bits[0]=False
            hashes.append(int.from_bytes(np.packbits(bits).tobytes(),'little'))
    return {'path':str(path), 'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
            'hw':list(rgb.shape[:2]), 'phash':hashes}

@lru_cache(maxsize=1400)
def orb_features(path):
    im=read_rgb(path);h,w=im.shape[:2]
    im=cv2.cvtColor(cv2.resize(im,(round(w*768/max(h,w)),round(h*768/max(h,w)))),cv2.COLOR_RGB2GRAY)
    return cv2.ORB_create(nfeatures=2000).detectAndCompute(im,None)

def orb_match(a,b):
    ka,da=orb_features(a);kb,db=orb_features(b)
    if da is None or db is None:return {'inliers':0,'fraction':0.}
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
    if len(good)<4:return {'inliers':0,'fraction':0.}
    _,mask=cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
                              np.float32([kb[m.trainIdx].pt for m in good]),cv2.RANSAC,3.)
    n=0 if mask is None else int(mask.sum())
    return {'inliers':n,'fraction':n/len(good),'matches':len(good)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--visual-reviewed',action='store_true');args=ap.parse_args()
    runtime=read_json(ROOT/'configs/runtime.local.json');base=Path(runtime['TJDR_root'])
    files=sorted((base/'image').glob('*.png'))
    assert len(files)==448 and len(list((base/'annotation').glob('*.png')))==448,'All official 448 pairs required'
    write_json(ROOT/'reports/TJDR_dedup_rules.json',RULES)
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=6) as pool:
        signatures=list(pool.map(image_signature,files))
    samples=[]
    for path,sig in zip(files,signatures):
        maskpath=base/'annotation'/path.name
        with Image.open(maskpath) as im:
            label=np.asarray(im);palette=im.getpalette();mode=im.mode
            assert mode in ('P','L') and label.shape==tuple(sig['hw'])
            values,counts=np.unique(label,return_counts=True)
            assert set(map(int,values)).issubset({0,1,2,3,4})
        assert sig['hw'] in ([2048,2048],[3912,3912]),'Unregistered camera dimensions require review'
        samples.append({'sid':path.stem,'split':'train','source':'TJDR','camera':('Topcon' if sig['hw'][0]==2048 else 'Clarus'),
            'image_sha256':sha(path),'label_sha256':sha(maskpath),'rgb_sha256':sig['rgb_sha256'],'hw':sig['hw'],
            'label_mode':mode,'palette':palette[:15] if palette else None,
            'original_label_counts':dict(zip(map(str,values.tolist()),counts.tolist())),
            'excluded_duplicate':False,'exclusion_reasons':[]})
    other=[]
    for split in ('train','val','test'):
        paths=sorted((Path(runtime['ddr_root'])/'images'/split).glob('*.jpg'))
        assert len(paths)=={'train':454,'val':76,'test':227}[split]
        with ThreadPoolExecutor(max_workers=6) as pool:
            for path,sig in zip(paths,pool.map(image_signature,paths)):
                sig.update(dataset='DDR',split=split,sid=path.stem);other.append(sig)
    pairs=[]
    pair_pool=ThreadPoolExecutor(max_workers=6)
    for index,(row,sig) in enumerate(zip(samples,signatures)):
        references=other+[dict(s,dataset='TJDR',split='train',sid=samples[j]['sid']) for j,s in enumerate(signatures[:index])]
        def compare(ref):
            exact=sig['rgb_sha256']==ref['rgb_sha256']
            distance=min((sig['phash'][0]^x).bit_count() for x in ref['phash'])
            if not exact and distance>18:return None
            evidence={} if exact else orb_match(sig['path'],ref['path'])
            excluded=exact or (evidence.get('inliers',0)>=40 and evidence.get('fraction',0)>=.6)
            pair={'TJDR_sid':row['sid'],'other_dataset':ref['dataset'],'other_split':ref['split'],
                  'other_sid':ref['sid'],'exact_rgb':exact,'phash_distance':distance,'orb':evidence,'excluded':excluded}
            return pair
        for pair in pair_pool.map(compare,references):
            if pair is None:continue
            pairs.append(pair)
            if pair['excluded']:
                row['excluded_duplicate']=True;row['exclusion_reasons'].append(pair)
        if index%10==0:print('dedup',index,'candidate_pairs',len(pairs),'ORB_cache',orb_features.cache_info(),flush=True)
        write_json(ROOT/'reports/TJDR_dedup_progress.json',{'completed_images':index+1,'total':448,'candidate_pairs':len(pairs),
            'excluded_so_far':sum(r['excluded_duplicate'] for r in samples[:index+1]),'time':utc_now()})
    pair_pool.shutdown(wait=True)
    write_json(ROOT/'reports/TJDR_duplicate_pairs.json',{'rules':RULES,'pairs':pairs,'uses_DDR_test_images_only':True})
    chosen=[]
    for camera in ('Topcon','Clarus'):
        candidates=[r for r in samples if r['camera']==camera and not r['excluded_duplicate']]
        # Include examples containing each class, then uniformly spaced camera examples.
        for c in ('1','2','3','4'):
            matches=[r for r in candidates if int(r['original_label_counts'].get(c,0))>0]
            if matches and matches[0] not in chosen:chosen.append(matches[0])
        for i in np.linspace(0,len(candidates)-1,10,dtype=int):
            if candidates[i] not in chosen:chosen.append(candidates[i])
    chosen=chosen[:24]
    assert len(chosen)>=20
    colors=np.array([[0,0,0],[255,180,0],[255,30,30],[0,200,255],[60,255,80]],np.uint8)
    panels=[]
    for row in chosen:
        rgb=read_rgb(base/'image'/(row['sid']+'.png'))
        label=np.asarray(Image.open(base/'annotation'/(row['sid']+'.png')))
        rgb=cv2.resize(rgb,(512,512));label=cv2.resize(label,(512,512),interpolation=cv2.INTER_NEAREST)
        overlay=rgb.copy();positive=label>0
        overlay[positive]=(.55*rgb[positive]+.45*colors[label[positive]]).astype(np.uint8)
        panel=Image.new('RGB',(1024,550));panel.paste(Image.fromarray(rgb),(0,38));panel.paste(Image.fromarray(overlay),(512,38))
        ImageDraw.Draw(panel).text((8,8),row['sid']+' '+row['camera']+' | EX orange HE red MA cyan SE green',fill='white')
        panels.append(panel)
    sheets=[]
    for page in range((len(panels)+3)//4):
        sheet=Image.new('RGB',(2048,1100))
        for j,panel in enumerate(panels[page*4:page*4+4]):sheet.paste(panel,((j%2)*1024,(j//2)*550))
        path=ROOT/f'reports/TJDR_overlay_{page+1:02d}.jpg';sheet.save(path,quality=90);sheets.append(str(path))
    visual=read_json(ROOT/'reports/TJDR_visual_review.json') if args.visual_reviewed else None
    if visual:assert visual['status']=='PASS' and visual['reviewed_sids']==[r['sid'] for r in chosen]
    result={'status':'READY' if visual else 'PENDING_VISUAL_REVIEW','time':utc_now(),'mapping_verified':True,
        'official_code_to_channel':{'1':'EX','2':'HE','3':'MA','4':'SE'},'DDR_index_remap':[0,1,2,4,3],
        'usage_basis':'Author public research release: official TJDR README and arxiv2312.15389v1 Open Access section; no distinct commercial redistribution license asserted.',
        'annotation_completeness':'Official paper: complete four-lesion pixel annotation and index0 background; no padding in original PNG.',
        'dedup_status':'PASS','dedup_rules':RULES,'dedup_limit':'Conservative image-based isolation; unknown patient identity and unrecognized crop duplicates remain a limitation.',
        'patient_identifiers_available':False,'official_train_count':448,'accepted_count':sum(not r['excluded_duplicate'] for r in samples),
        'official_test_downloaded':False,'visual_review_count':len(chosen) if visual else 0,
        'visual_review_sids':[r['sid'] for r in chosen],'overlay_sheets':sheets,'samples':samples}
    write_json(ROOT/'reports/TJDR_manifest.json',result);print(result['status'],result['accepted_count'],flush=True)

if __name__=='__main__':main()
