"""Recover GT-defined lossless FP32 crops of saved val probabilities only."""
import json,shlex,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.remote import connect,command
from qlib.common import read_json,write_json,sha,utc_now
from qlib.metrics import counts

def main():
 plan=read_json(ROOT/'reports/publication_server_evidence_plan.json');runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root']
 assert plan['no_test_inference_or_training'] and plan['no_endpoint_quantization']
 code="""from pathlib import Path
import json,hashlib,os
import numpy as np
from PIL import Image
r=Path(%r);plan=json.loads(%r);runtime=json.loads((r/'configs/runtime.server.json').read_text())
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
 return d.hexdigest()
files=[];checks=[]
for case in plan['cases']:
 sid=case['sid'];sources=[];prob=[];labelpath=Path(runtime['ddr_root'])/'labels/val'/(sid+'.png')
 assert h(labelpath)==case['label_sha256']
 with Image.open(labelpath) as im:label=np.asarray(im)
 gt=np.stack([label==c for c in range(1,5)]);valid=np.broadcast_to(label!=255,gt.shape)
 for component,prefix in plan['p0_source_directories'].items():
  p=Path(prefix)/(sid+'.npz');sources.append({'component':component,'path':str(p),'bytes':p.stat().st_size,'sha256':h(p)})
  with np.load(p,allow_pickle=False) as f:value=f['prob']
  assert value.dtype==np.float32 and value.shape==gt.shape and np.isfinite(value).all()
  prob.append(value)
 p0=.75*prob[0]+.25*prob[1];pred=(p0>.5)&valid;g=gt&valid
 original_counts=np.stack([(pred&g).sum((1,2)),(pred&~g&valid).sum((1,2)),(~pred&g).sum((1,2))],axis=1).tolist()
 assert original_counts==case['p0_expected_global_counts'],sid
 x0,y0,x1,y1=case['crop_xyxy'];crop=p0[:,y0:y1,x0:x1].copy()
 relative='reports/publication_server_evidence/val_p0_crops/'+sid+'.npz';target=r/relative;target.parent.mkdir(parents=True,exist_ok=True)
 with target.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,prob=crop)
 os.replace(target.with_suffix('.tmp'),target)
 files.append({'path':relative,'sha256':h(target),'bytes':target.stat().st_size,'dtype':'float32','shape':list(crop.shape),'sid':sid,'crop_xyxy':case['crop_xyxy'],'source_files':sources})
 checks.append({'sid':sid,'exact_original_p0_counts':original_counts,'label_sha256':case['label_sha256']})
assert sum(x['bytes'] for x in files)<100*1024**2
result={'stage':'READ_ONLY_SAVED_VAL_FP32_CROPS_DERIVED','files':files,'checks':checks,'plan_sha256':%r,'no_inference_training_or_quantization':True}
(r/'reports/publication_server_evidence/derivation.json').write_text(json.dumps(result))
print(json.dumps(result))
"""%(remote,json.dumps(plan),sha(ROOT/'reports/publication_server_evidence_plan.json'))
 with connect() as client:
  derivation=json.loads(command(client,shlex.quote(runtime['server_python'])+' -c '+shlex.quote(code),180))
  write_json(ROOT/'reports/publication_server_evidence_inventory.json',derivation)
  with client.open_sftp() as sftp:
   for i,row in enumerate(derivation['files']):
    target=(ROOT/'server_results'/row['path']).resolve();assert target.is_relative_to((ROOT/'server_results/reports/publication_server_evidence').resolve());target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or sha(target)!=row['sha256']:
     tmp=target.with_suffix('.npz.download');sftp.get(remote+'/'+row['path'],str(tmp),prefetch=True,max_concurrent_prefetch_requests=64);assert sha(tmp)==row['sha256'];tmp.replace(target)
    row['local_path']=str(target)
    write_json(ROOT/'reports/publication_server_evidence_transfer.json',{'stage':'READ_ONLY_FP32_CROP_CUSTODY','verified_files':i+1,'expected_files':len(derivation['files']),'time':utc_now()})
 checks=[];cases={c['sid']:c for c in plan['cases']}
 for row in derivation['files']:
  sid=row['sid'];case=cases[sid];x0,y0,x1,y1=case['crop_xyxy'];labelpath=Path(runtime['ddr_root'])/'labels/val'/(sid+'.png');assert sha(labelpath)==case['label_sha256']
  with Image.open(labelpath) as im:label=np.asarray(im)[y0:y1,x0:x1]
  gt=np.stack([label==c for c in range(1,5)]);valid=np.broadcast_to(label!=255,gt.shape)
  with np.load(row['local_path'],allow_pickle=False) as f:p0=f['prob']
  assert p0.dtype==np.float32 and list(p0.shape)==row['shape']
  qpath=ROOT/'server_results/runs/S_seed42/val_u20000'/(sid+'.npz')
  with np.load(qpath,allow_pickle=False) as f:q=f['prob'][:,y0:y1,x0:x1]
  checks.append({'sid':sid,'q_original_file_sha256':sha(qpath),'p0_crop_counts':counts(p0,gt,valid).tolist(),'locked_formula_crop_counts':counts(.75*p0+.25*q,gt,valid).tolist(),'crop_xyxy':case['crop_xyxy']})
 receipt={'stage':'REQUIRED_SAVED_VALIDATION_EVIDENCE_SHA_AND_COUNTS_VERIFIED','time':utc_now(),'plan_sha256':sha(ROOT/'reports/publication_server_evidence_plan.json'),'selection_lock_sha256':plan['lock_sha256'],'files':derivation['files'],'remote_original_global_count_checks':derivation['checks'],'local_crop_checks':checks,'all_outputs_recovered_with_SHA':True,'no_inference_training_threshold_scan_or_selection':True,'no_required_FP32_endpoint_quantized_or_deleted':True,'no_test_images_used':True}
 write_json(ROOT/'reports/publication_server_evidence_recovery.json',receipt)
 print(json.dumps({'stage':receipt['stage'],'files':len(derivation['files']),'bytes':sum(x['bytes'] for x in derivation['files'])}),flush=True)

if __name__=='__main__':main()
