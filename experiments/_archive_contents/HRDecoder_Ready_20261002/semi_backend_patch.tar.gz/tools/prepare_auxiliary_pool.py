"""Audit an existing curated grading TRAIN pool; never include grading val/test.

The preparation script proves folder provenance, but its old source text lists
are no longer present. Record that limit. Exclude filenames and conservative
anatomy matches against all 757 curated segmentation images before transfer.
"""
import hashlib,json,sys,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_duplicates import descriptor
from tools.configure_server import sha

def main():
    cv2.setNumThreads(1)
    src=Path(r'D:\dataset\YOLO_Grading_CLS');ddr=Path(r'D:\HRDecoder\data\DDR_split_613')
    ddrs=sorted((ddr/'images').rglob('*.jpg'))
    if len(ddrs)!=757:raise ValueError('Need every curated segmentation image for overlap audit')
    forbidden={p.stem for p in ddrs}
    forbidden|={p.stem for s in ['val','test'] for p in (src/s).rglob('*.jpg')}
    alltrain=sorted((src/'train').rglob('*.jpg'))
    eligible=[p for p in alltrain if p.stem not in forbidden]
    eligible.sort(key=lambda p:hashlib.sha256(('aux42:'+p.stem).encode()).hexdigest())
    audited=eligible[:1300]
    def get(p):
        g,e,_=descriptor(p);return g,e
    with ThreadPoolExecutor(max_workers=4) as ex:
        gd,ed=map(np.stack,zip(*ex.map(get,ddrs)))
        ga,ea=map(np.stack,zip(*ex.map(get,audited)))
    gray=ga@gd.T;edge=ea@ed.T
    cross=(gray>=.94)&(edge>=.85)
    # This deliberately excludes uncertain near matches as well as exact
    # matches; grading images have CLAHE and median filtering applied.
    gg=ga@ga.T;ee=ea@ea.T;within=(gg>=.94)&(ee>=.85)
    keep=[];removed=[]
    for i,p in enumerate(audited):
        near=np.flatnonzero(cross[i]);prev=[j for j in keep if within[i,j]]
        if len(near) or prev:
            removed.append(dict(image=str(p),matches=[str(ddrs[j]) for j in near],
                                auxiliary_matches=[str(audited[j]) for j in prev]))
        else:keep.append(i)
    keep=keep[:1000]
    out=ROOT/'assets/auxiliary';(out/'images').mkdir(parents=True,exist_ok=True)
    records=[]
    for i in keep:
        p=audited[i];target=out/'images'/p.name;shutil.copy2(p,target)
        records.append(dict(sample_id=p.stem,image='images/'+p.name,sha256=sha(p),
                            original=str(p),grading_class=p.parent.name,split='train',source='DDR_curated_grading'))
    (out/'train_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    report=dict(source=str(src/'train'),source_script=str(Path(r'D:\dataset\dr_pipeline\run_efficient_training.py')),
        source_script_sha256=sha(Path(r'D:\dataset\dr_pipeline\run_efficient_training.py')),
        transform_script_sha256=sha(Path(r'D:\dataset\dr_pipeline\preprocess_all.py')),
        transform='CLAHE green clip2 grid8; median filter3; no resize in preprocessing script',
        source_train_images=len(alltrain),filename_excluded=len(alltrain)-len(eligible),
        segmentation_audited=len(ddrs),anatomy_audited=len(audited),anatomy_excluded=removed,
        kept=len(records),grading_classes={str(c):sum(r['grading_class']==str(c) for r in records) for c in range(5)},
        threshold=dict(gray=.94,edge=.85),manifest_sha256=sha(out/'train_manifest.jsonl'),
        limitation='Old grading cleaned label lists are absent; provenance uses saved copy script and train folders. Patient IDs unknown; arbitrary rotations not exhaustively searched.')
    (ROOT/'reports/auxiliary_pool_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['source_train_images','filename_excluded','anatomy_audited','kept','grading_classes']}))

if __name__=='__main__':main()
