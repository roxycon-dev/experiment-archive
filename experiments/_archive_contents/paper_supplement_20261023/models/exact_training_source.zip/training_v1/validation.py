"""Original-pixel counts; bounded background compression and mechanism audit."""
import os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
from qlib.common import sha,write_json
from qlib.geometry import rgb_window,center_logits,tiles
from qlib.metrics import counts,summarize
from qlib.evaluate import load_p0

def compress(path,q):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,prob=q)
    os.replace(temporary,path)
    return {'sid':path.stem,'sha256':sha(path),'bytes':path.stat().st_size,'dtype':'float32','shape':list(q.shape)}

def mechanism(rgb,q,g,v,sid,output):
    result={}
    for c,name in ((1,'HE'),(3,'MA')):
        prediction=(q[c]>.5)&(v[c]>0);gt=(g[c]>0)&(v[c]>0)
        n,components,stats,_=cv2.connectedComponentsWithStats(gt.astype(np.uint8),connectivity=8)
        touched=np.unique(components[prediction& gt]);hit=set(map(int,touched))-{0}
        missed=[k for k in range(1,n) if k not in hit]
        fp=prediction & ~gt
        m,fpids,fpstats,_=cv2.connectedComponentsWithStats(fp.astype(np.uint8),connectivity=8)
        result[name]={'gt_lesions':n-1,'missed_lesions':len(missed),'new_false_positive_regions':m-1,
                      'definition':'8-connected valid GT components missed when zero predicted overlap; FP components are predicted valid pixels outside GT'}
        # Deterministic largest examples, not outcome-dependent primary selection.
        for kind,ids,table,labels in (('miss',missed,stats,components),('fp',list(range(1,m)),fpstats,fpids)):
            for rank,k in enumerate(sorted(ids,key=lambda k:(-int(table[k,4]),k))[:1]):
                x,y,w,h,_=map(int,table[k]);pad=64;y0=max(0,y-pad);x0=max(0,x-pad);y1=min(len(rgb),y+h+pad);x1=min(rgb.shape[1],x+w+pad)
                crop=rgb[y0:y1,x0:x1].copy();mask=labels[y0:y1,x0:x1]==k
                crop[mask]=(.5*crop[mask]+.5*np.array([255,30,30])).astype(np.uint8)
                from PIL import Image
                path=output/'mechanism_examples'/f'{sid}_{name}_{kind}_{rank}.jpg';path.parent.mkdir(exist_ok=True)
                image=Image.fromarray(crop);image.thumbnail((768,768));image.save(path,quality=88)
    return result

@torch.no_grad()
def evaluate(model,dataset,runtime,output,update,keep_probability=False,locked_formula=None,p0_ram=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if dataset.split=='test':assert runtime.get('final_test_authorized') and locked_formula in ('single','fixed_fusion','half_fusion')
    training=model.training;model.eval();device=next(model.parameters()).device
    rows=[];hashes=[];pending=[];started=time.perf_counter();pool=ThreadPoolExecutor(max_workers=2)
    try:
        for sid in dataset.ids:
            rgb,g,v=dataset.load(sid);hw=rgb.shape[:2];q=np.empty((4,*hw),np.float32);before=time.perf_counter()
            for tile in tiles(*hw):
                x=torch.from_numpy(rgb_window(rgb,tile)).unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):z=center_logits(model(x))
                q[:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]=z.float().sigmoid()[0,:,:tile.height,:tile.width].cpu().numpy()
            gt,valid=g.transpose(2,0,1),v.transpose(2,0,1)
            if dataset.split=='test' and locked_formula=='single':predictions={'single':q}
            else:
                p0=p0_ram[sid] if p0_ram is not None else load_p0(runtime,sid,hw)
                predictions={'single':q,'fixed_fusion':.75*p0+.25*q,'half_fusion':.5*p0+.5*q,'p0':p0}
            formulas=(locked_formula,) if dataset.split=='test' else tuple(predictions)
            row={'sid':sid,'q_inference_seconds':time.perf_counter()-before,
                 **{formula:counts(predictions[formula],gt,valid).tolist() for formula in formulas}}
            if dataset.split=='val':row['mechanism']=mechanism(rgb,q,gt,valid,sid,output)
            rows.append(row)
            if keep_probability:
                if len(pending)==2:hashes.append(pending.pop(0).result())
                pending.append(pool.submit(compress,output/(sid+'.npz'),q))
            write_json(output/'progress.json',{'completed_images':len(rows),'expected':len(dataset.ids),'update':update})
        hashes.extend(f.result() for f in pending)
    finally:pool.shutdown(wait=True);model.train(training)
    result={'update':update,'split':dataset.split,'n_images':len(rows),'per_image':rows,'seconds':time.perf_counter()-started,
        'threshold':.5,'strict_greater':True,'coordinates':'original_image','probability_dtype':'float32','probability_files':hashes}
    for formula in formulas:result[formula]=summarize(np.asarray([r[formula] for r in rows]).sum(axis=0))
    if dataset.split=='test':result['locked_formula']=locked_formula
    write_json(output/'metrics.json',result)
    return result

def cache_p0(dataset,runtime,limit_bytes=32*1024**3):
    cache={};total=0
    for sid in dataset.ids:
        from PIL import Image
        with Image.open(dataset.images[sid]) as im:w,h=im.size
        total+=16*h*w
        assert total<=limit_bytes,'P0 RAM cache exceeded explicit budget'
        value=load_p0(runtime,sid,(h,w));value.flags.writeable=False;cache[sid]=value
    return cache,{'bytes':total,'images':len(cache),'float32_original_probability':True}
