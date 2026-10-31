"""CPU-only external supervision acceptance; visual review is a separate human/agent gate."""
import concurrent.futures
import hashlib
import io
import json
import pathlib
import time
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = ('EX', 'HE', 'SE', 'MA')
MASK_NAMES = dict(EX='Exudates', HE='Hemorrhages', SE='CottonWoolSpots', MA='Microaneurysms')
MASK_ROOT = ROOT/'assets/external/MAPLES_native_masks'
OUT = ROOT/'reports/external_acceptance'
DERIVED = ROOT/'assets/external_derived/IDRiD_best_segmentation_pixel_masks'
COLORS = [(255,220,0), (255,60,60), (70,230,255), (255,80,230)]

def sha(p):
    h = hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()

def write(p, x):
    p = pathlib.Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix('.tmp'); q.write_text(json.dumps(x, indent=2), encoding='utf-8'); q.replace(p)

def progress(stage, **kw):
    write(ROOT/'reports/external_acceptance_progress.json', {'stage':stage, 'updated_at':datetime.now(timezone.utc).isoformat(), **kw})

def fingerprint(row):
    with Image.open(row['image']) as im:
        rgb = np.asarray(im.convert('RGB'))
    pixel = hashlib.sha256(str(rgb.shape).encode()+rgb.tobytes()).hexdigest()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ys, xs = np.nonzero(gray > 8)
    crop = gray[ys.min():ys.max()+1, xs.min():xs.max()+1] if len(ys) else gray
    def hashes(a):
        small = cv2.resize(a, (32,32), interpolation=cv2.INTER_AREA).astype(np.float32)
        low = cv2.dct(small)[:8,:8].flatten()[1:]
        bits = low > np.median(low)
        ph = sum(int(b)<<i for i,b in enumerate(bits))
        small = cv2.resize(a, (9,8), interpolation=cv2.INTER_AREA)
        dh = sum(int(b)<<i for i,b in enumerate((small[:,1:] > small[:,:-1]).flatten()))
        return [ph,dh]
    return {**row, 'pixel_sha256':pixel, 'full_hash':hashes(gray), 'black_bbox_hash':hashes(crop), 'dimensions':list(rgb.shape[:2][::-1])}

def distance(a,b):
    options=[]
    for key in ('full_hash','black_bbox_hash'):
        p=(a[key][0]^b[key][0]).bit_count(); d=(a[key][1]^b[key][1]).bit_count()
        options.append((p,d,key))
    return min(options)

def overlay_sheet(samples, name):
    sheet=Image.new('RGB',(1280,5*330),(20,20,20)); draw=ImageDraw.Draw(sheet)
    for k,row in enumerate(samples):
        rgb=np.array(Image.open(row['image']).convert('RGB'))
        mask=[np.asarray(Image.open(row['classes'][c]['mask']))>0 for c in CLASSES]
        colored=rgb.copy()
        for a,color in zip(mask,COLORS):
            edge=cv2.morphologyEx(a.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))>0
            colored[edge]=color
        uncertain=np.zeros(mask[0].shape,bool)
        if row['source']=='MAPLES':
            for key in ('RedUncertains','BrightUncertains'):
                uncertain |= np.asarray(Image.open(row['uncertain'][key]))>0
            colored[uncertain]=((colored[uncertain].astype(np.uint16)+[150,150,150])//2).astype(np.uint8)
        c=k%4
        interest=mask[c] if mask[c].any() else (uncertain if uncertain.any() else np.logical_or.reduce(mask))
        yy,xx=np.nonzero(interest)
        cy,cx=(int(np.median(yy)),int(np.median(xx))) if len(yy) else (rgb.shape[0]//2,rgb.shape[1]//2)
        y=max(0,min(cy-256,rgb.shape[0]-512)); x=max(0,min(cx-256,rgb.shape[1]-512))
        full=Image.fromarray(colored); full.thumbnail((305,280))
        detail=Image.fromarray(colored[y:y+512,x:x+512]); detail.thumbnail((280,280))
        sx=(k%2)*640; sy=(k//2)*330
        sheet.paste(full,(sx+5,sy+27)); sheet.paste(detail,(sx+325,sy+27))
        draw.text((sx+5,sy+5),row['source']+' '+row['sid']+'  crop:'+str([x,y]),fill='white')
        draw.text((sx+5,sy+310),'EX yellow HE red SE cyan MA magenta; uncertain gray',fill='white')
    dest=OUT/name; sheet.save(dest)
    return {'sheet':str(dest), 'sha256':sha(dest), 'case_ids':[{ 'source':r['source'],'sid':r['sid']} for r in samples]}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    progress('EXTRACTING_NATIVE_TRAIN_MASKS')
    maples_zip=ROOT.parent/'HRDecoder_Ready_20261002/assets/MAPLES-DR.zip'
    with zipfile.ZipFile(maples_zip) as z:
        for name in z.namelist():
            parts=name.split('/')
            if len(parts)==3 and parts[0]=='train' and parts[1] in (*MASK_NAMES.values(),'RedUncertains','BrightUncertains') and name.endswith('.png'):
                dest=MASK_ROOT/name; dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(name))
    derived=json.loads((DERIVED/'manifest.json').read_text(encoding='utf-8'))
    samples=[]
    for r in derived['samples']:
        if r['source_split']!='training': continue
        assert sha(r['image_path'])==r['source_image_sha256']
        classes={}
        for c in CLASSES:
            m=r['masks'][c]; p=DERIVED/m['path']; assert sha(p)==m['sha256']
            classes[c]={'mask':str(p), 'sha256':m['sha256'], 'review_status':'pixel_mask' if m['positive_pixels'] else 'unknown',
                        'completeness_basis':'User-authorized polygon annotation; empty class not proven reviewed', 'positive_pixels':m['positive_pixels']}
        samples.append({'source':'IDRiD','sid':r['sid'],'split':'train','image':r['image_path'], 'image_sha256':r['source_image_sha256'],
                        'classes':classes,'excluded_duplicate':False,'source_quality':'user_authorized_stored_geometry_polygon_labels'})
    # Image-only fingerprints of all DDR splits and both user IDRiD splits; no DDR test GT is opened.
    runtime=json.loads((ROOT/'configs/runtime.local.json').read_text(encoding='utf-8'))
    ddr=[]
    for split in ('train','val','test'):
        ddr.extend({'source':'DDR','sid':p.stem,'split':split,'image':str(p)} for p in (pathlib.Path(runtime['ddr_root'])/'images'/split).glob('*.jpg'))
    assert len(ddr)==757
    idrid_image_rows=[{'source':'IDRiD','sid':r['sid'],'split':'train' if r['source_split']=='training' else 'test','image':r['image_path']} for r in derived['samples']]
    progress('DDR_AND_IDRID_IMAGE_ONLY_DEDUP_FINGERPRINTS')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        ddr_f=list(pool.map(fingerprint,ddr)); ext_f=list(pool.map(fingerprint,idrid_image_rows))
    progress('WAITING_MESSIDOR_DOWNLOAD', ddr_fingerprints=757, idrid_fingerprints=81)
    for _ in range(180):
        p=ROOT/'reports/messidor_download_progress.json'
        s=json.loads(p.read_text()) if p.exists() else {}
        if s.get('stage')=='COMPLETE': break
        if s.get('stage')=='FAILED': raise RuntimeError('MESSIDOR acquisition failed; no automatic acceptance')
        time.sleep(5)
    else: raise TimeoutError('Bounded acquisition wait expired; preserve fingerprints and retry after root-cause review')
    receipt=json.loads((ROOT/'reports/messidor_download_receipt.json').read_text())
    assert receipt['stage']=='COMPLETE' and len(receipt['images'])==198
    image_rows=[]
    progress('MAPLES_SHAPES_VALIDITY_AND_IMAGE_ONLY_DEDUP')
    for r in receipt['images']:
        image_rows.append({'source':'MAPLES','sid':r['id'],'split':r['split'],'image':r['image']})
        if r['split']!='train': continue
        classes={}; arrays={}
        for c,folder in MASK_NAMES.items():
            p=MASK_ROOT/'train'/folder/(r['id']+'.png')
            with Image.open(p) as im: a=np.asarray(im)>0; assert list(im.size)==r['dimensions']
            arrays[c]=a
            classes[c]={'mask':str(p),'sha256':sha(p),'review_status':'pixel_mask',
                        'completeness_basis':'Official MAPLES class mask present, including explicit reviewed zero PNG', 'positive_pixels':int(a.sum())}
        uncertain={}; ignore={}
        for key in ('RedUncertains','BrightUncertains'):
            p=MASK_ROOT/'train'/key/(r['id']+'.png')
            with Image.open(p) as im: a=np.asarray(im)>0; assert list(im.size)==r['dimensions']
            uncertain[key]=str(p); ignore[key]=a
        for c in CLASSES:
            a=ignore['RedUncertains' if c in ('HE','MA') else 'BrightUncertains']
            classes[c].update(valid_pixels=int(a.size-a.sum()),ignore_pixels=int(a.sum()),legal_positive_pixels=int((arrays[c]&~a).sum()))
        samples.append({'source':'MAPLES','sid':r['id'],'split':'train','image':r['image'],'image_sha256':r['sha256'],
            'classes':classes,'uncertain':uncertain,'uncertain_sha256':{k:sha(p) for k,p in uncertain.items()},
            'uncertain_confirmed_absent':{k:False for k in uncertain},'excluded_duplicate':False,'source_quality':'official_native_MESSIDOR_MAPLES_pixel_GT'})
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: ext_f.extend(pool.map(fingerprint,image_rows))
    candidates=[]
    def compare(a,b):
        raw=sha(a['image'])==sha(b['image']) if a['pixel_sha256']==b['pixel_sha256'] else False
        p,d,view=distance(a,b)
        if raw or a['pixel_sha256']==b['pixel_sha256'] or (p<=6 and d<=12):
            return {'a':{k:a[k] for k in ('source','sid','split','image')},'b':{k:b[k] for k in ('source','sid','split','image')},
                    'exact_decoded_pixels':a['pixel_sha256']==b['pixel_sha256'],'exact_file':raw,'phash_distance':p,'dhash_distance':d,'hash_view':view}
    for a in ext_f:
        for b in ddr_f:
            hit=compare(a,b)
            if hit: candidates.append(hit)
    for i,a in enumerate(ext_f):
        for b in ext_f[i+1:]:
            hit=compare(a,b)
            if hit: candidates.append(hit)
    write(OUT/'image_dedup.json', {'status':'PENDING_CANDIDATE_REVIEW' if candidates else 'PASS', 'DDR_images':757,'external_images':len(ext_f),
        'DDR_test_labels_read':False,'algorithm':'Decoded RGB SHA256 plus full/black-bbox grayscale pHash63<=6 AND dHash64<=12; thresholds fixed before inspection',
        'patient_identity_independence_proven':False,'candidates':candidates})
    write(OUT/'image_fingerprints.json',{'DDR':ddr_f,'external':ext_f})
    stats={}
    for source in ('IDRiD','MAPLES'):
        rows=[r for r in samples if r['source']==source]
        stats[source]={'train_images':len(rows),'classes':{}}
        for c in CLASSES:
            ss=[r['classes'][c] for r in rows]
            stats[source]['classes'][c]={'positive_images':sum(x['positive_pixels']>0 for x in ss),
                'unknown_images':sum(x['review_status']=='unknown' for x in ss),'annotated_positive_pixels':sum(x['positive_pixels'] for x in ss)}
        assert all(stats[source]['classes'][c]['positive_images']>0 for c in CLASSES)
    for r in samples:
        if r['source']=='IDRiD':
            dimensions=next(x['size_wh'] for x in derived['samples'] if x['sid']==r['sid']); area=dimensions[0]*dimensions[1]
            for c in CLASSES:
                unknown=r['classes'][c]['review_status']=='unknown'
                r['classes'][c].update(valid_pixels=0 if unknown else area,ignore_pixels=area if unknown else 0,legal_positive_pixels=r['classes'][c]['positive_pixels'])
    for source in ('IDRiD','MAPLES'):
        for c in CLASSES:
            ss=[r['classes'][c] for r in samples if r['source']==source]
            stats[source]['classes'][c].update(valid_pixels=sum(x['valid_pixels'] for x in ss),ignore_pixels=sum(x['ignore_pixels'] for x in ss))
    reviews=[]
    for source in ('IDRiD','MAPLES'):
        rows=sorted([r for r in samples if r['source']==source],key=lambda r:r['sid']); chosen=[]
        def add(row):
            if row not in chosen: chosen.append(row)
        for c in CLASSES: add(max(rows,key=lambda r:r['classes'][c]['legal_positive_pixels']))
        add(min(rows,key=lambda r:sum(r['classes'][c]['positive_pixels'] for c in CLASSES)))
        if source=='MAPLES': add(max(rows,key=lambda r:r['classes']['HE']['ignore_pixels']+r['classes']['EX']['ignore_pixels']))
        for row in rows:
            if len(chosen)>=10: break
            add(row)
        assert len(chosen)==10
        reviews.append(overlay_sheet(chosen,source+'_review_10.png'))
    for i,hit in enumerate(candidates):
        canvas=Image.new('RGB',(1000,420),'black'); draw=ImageDraw.Draw(canvas)
        for j,key in enumerate(('a','b')):
            im=Image.open(hit[key]['image']).convert('RGB'); im.thumbnail((490,360)); canvas.paste(im,(j*500,40))
            draw.text((j*500+3,5),str({k:v for k,v in hit[key].items() if k!='image'}),fill='white')
        canvas.save(OUT/f'duplicate_candidate_{i:03d}.png')
    manifest={'status':'PENDING_VISUAL_AND_DEDUP_REVIEW','dedup_status':'PASS' if not candidates else 'PENDING',
        'visual_review_count':0,'source_override':'reports/external_source_user_override.json','samples':samples,
        'source_counts':stats,'quarantined_test_images':{'IDRiD':27,'MAPLES':60},'test_GT_used_for_training':False,
        'visual_review_sheets':reviews,'created_at':datetime.now(timezone.utc).isoformat()}
    write(OUT/'manifest.local.pending.json',manifest)
    progress('PENDING_VISUAL_REVIEW',training_images=len(samples),dedup_candidates=len(candidates),review_sheets=reviews,source_counts=stats)
    print('PENDING_VISUAL_REVIEW:',len(samples),'train images;',len(candidates),'dedup candidates',flush=True)

if __name__=='__main__': main()
