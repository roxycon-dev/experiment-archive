"""Saved paired DDR stream and camera-balanced, separately seeded TJDR stream."""
import json
from pathlib import Path
import numpy as np
from qlib.common import sha, read_json, write_json
from qlib.data import DDR, catalog, draw_row
from qlib.geometry import transform_points
from inputs import TJDR
ROOT=Path(__file__).resolve().parent

def source_row(dataset, metadata, positives, seed, update, slot, c):
    rng=np.random.default_rng(np.random.SeedSequence([seed,update,slot,3]))
    camera=('Topcon','Clarus')[int(rng.integers(2))]
    ids=[sid for sid in dataset.ids if dataset.rows[sid]['camera']==camera]
    assert ids
    attempts=0;fallback=None
    while True:
        sid=ids[int(rng.integers(len(ids)))];attempts+=1
        if c is None or metadata[sid]['components'][c]:break
        if attempts==64:
            choices=[sid for sid in ids if metadata[sid]['components'][c]]
            if not choices:
                choices=positives[c];fallback='selected_camera_has_no_class_use_other_camera'
            else:fallback='64_rejections_use_uniform_positive_in_selected_camera'
            sid=choices[int(rng.integers(len(choices)))];break
    meta=metadata[sid]
    row={'sid':sid,'source':'TJDR','update':update,'slot':slot,'camera_draw':camera,
         'camera_actual':dataset.rows[sid]['camera'],'image_draw_attempts':attempts,'fallback':fallback,
         'rotation':int(rng.integers(4)),'flip_h':bool(rng.integers(2)),'flip_v':bool(rng.integers(2)),
         'foreground_class':c,'brightness':1.,'contrast':1.}
    h,w=meta['hw']
    if c is not None:
        entries=meta['components'][c];comp=entries[int(rng.integers(len(entries)))];y,x,bh,bw=comp['bbox']
        points,(h,w)=transform_points(np.array([[y,x],[y+bh-1,x+bw-1]]),meta['hw'],row)
        lo,hi=points.min(0),points.max(0)
        if np.all(hi-lo+1<=768):
            ylo,yhi=max(0,int(hi[0])-767),min(int(lo[0]),max(0,h-768))
            xlo,xhi=max(0,int(hi[1])-767),min(int(lo[1]),max(0,w-768))
        else:
            anchor=comp['points'][int(rng.integers(len(comp['points'])))];anchor,_=transform_points(anchor[None],meta['hw'],row)
            ay,ax=map(int,anchor[0]);ylo,yhi=max(0,ay-767),min(ay,max(0,h-768));xlo,xhi=max(0,ax-767),min(ax,max(0,w-768))
        y,x=int(rng.integers(ylo,yhi+1)),int(rng.integers(xlo,xhi+1))
    else:
        if row['rotation']%2:h,w=w,h
        y,x=int(rng.integers(max(0,h-768)+1)),int(rng.integers(max(0,w-768)+1))
    row.update(y=y,x=x,hw=[h,w])
    if rng.random()<.5:row.update(brightness=float(rng.uniform(.9,1.1)),contrast=float(rng.uniform(.9,1.1)))
    return row

def prepare(runtime):
    ddr=DDR(runtime['ddr_root'],'train',read_json(ROOT/'reports/data_identity.json')['splits']['train'])
    tjdr=TJDR(runtime['TJDR_root'],read_json(ROOT/'reports/TJDR_manifest.json'))
    catalogs={}
    for name,dataset in (('DDR',ddr),('TJDR',tjdr)):
        print('catalog',name,flush=True);catalogs[name]=catalog(dataset)
    receipt={}
    for seed in (42,2024):
        for name,dataset,updates,slots in (('DDR',ddr,32000,8),('TJDR',tjdr,20000,4)):
            path=ROOT/f'reports/stream_{name}_seed{seed}.jsonl';meta,positive=catalogs[name]
            with path.open('w',encoding='utf-8') as f:
                for u in range(1,updates+1):
                    for slot in range(slots):
                        if name=='DDR':row=draw_row(dataset,meta,positive,seed,u,slot,None if slot<4 else slot-4)
                        else:row=source_row(dataset,meta,positive,seed,u,slot,((u-1)*2+slot)%4 if slot<2 else None)
                        f.write(json.dumps(row,separators=(',',':'))+'\n')
            receipt[path.name]={'sha256':sha(path),'rows':updates*slots,'seed':seed,'source':name,
                'DDR_reuses_same_seed_update_slot_scheme':name=='DDR',
                'interpretation':'Repeated DDR exposure; not disjoint new training windows' if name=='DDR' else 'Independent source stream, shared exactly by D/SD'}
    receipt['stream_PARENT_seed2024.jsonl']={'sha256':sha(ROOT/'reports/stream_PARENT_seed2024.jsonl'),'rows':96000,'seed':2024,'source':'DDR','historical':True}
    write_json(ROOT/'reports/stream_identity.json',receipt)
    return receipt

if __name__=='__main__':prepare(read_json(ROOT/'configs/runtime.local.json'))
