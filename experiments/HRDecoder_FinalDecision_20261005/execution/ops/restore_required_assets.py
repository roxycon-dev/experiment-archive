"""Restore only this protocol's missing inputs; original local files remain authoritative."""
import sys,json,shlex,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,utc_now
from ops.remote import connect,command

def main():
    rt=read_json(ROOT/'configs/runtime.local.json');remote=rt['server']['remote_root']
    q=Path(rt['old_root']);oldq='/root/autodl-tmp/HRDecoder_Q_20261003';f=Path(rt['parent_f12_execution'])
    rows=[]
    def add(p,target,kind='required'):
        assert p.is_file(),str(p)
        rows.append({'local':str(p),'server':target,'bytes':p.stat().st_size,'sha256':sha(p),'kind':kind})
    for sub in ('qlib','vendor/dinov3'):
        for p in (q/sub).rglob('*'):
            if p.is_file() and not any(part in ('.git','__pycache__') for part in p.relative_to(q).parts):add(p,oldq+'/'+p.relative_to(q).as_posix())
    for rel in ('assets/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth','configs/p0_original_lock.json','configs/p0_identity.json',
                'reports/final_evaluation.py','reports/p0_historical_test_reference.json','reports/data_identity.json'):
        add(q/rel,oldq+'/'+rel)
    for seed in (42,2024):
        base=q/f'server_results/runs/Q_DDR_seed{seed}';dest=oldq+f'/runs/Q_DDR_seed{seed}'
        add(base/'checkpoint_u16000.pth',dest+'/checkpoint_u16000.pth','protected_Q16k_start')
        for p in (base/'val_u16000').iterdir():
            if p.suffix=='.json':add(p,dest+'/val_u16000/'+p.name,'control_metrics')
    for rel in ('checkpoint_u12000.pth','val_u12000/metrics.json'):
        add(f/'server_results/runs/F12_seed42'/rel,'/root/autodl-tmp/HRDecoder_Finetune_20261004/runs/F12_seed42/'+rel,'protected_F12_parent')
    m=read_json(ROOT/'reports/TJDR_manifest.json');assert m['status']=='READY'
    for row in m['samples']:
        for sub,key in (('image','image_sha256'),('annotation','label_sha256')):
            p=ROOT/f'inputs/TJDR/train/{sub}/{row["sid"]}.png';add(p,remote+'/'+p.relative_to(ROOT).as_posix(),'TJDR_train')
            assert rows[-1]['sha256']==row[key]
    write_json(ROOT/'reports/required_assets_inventory.json',{'time':utc_now(),'files':rows,'only_official_TJDR_train':True})
    code="from pathlib import Path;import json,hashlib\nrows=json.loads(%r)\nfor row in rows:\n p=Path(row['server']);p.parent.mkdir(parents=True,exist_ok=True)\n row['existing_sha256']=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() and p.stat().st_size==row['bytes'] else None\n tmp=Path(str(p)+'.incoming');row['incoming_bytes']=tmp.stat().st_size if tmp.exists() else 0;row['incoming_sha256']=hashlib.sha256(tmp.read_bytes()).hexdigest() if tmp.exists() else None\nprint(json.dumps(rows))"%json.dumps(rows)
    with connect() as c:before=json.loads(command(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",300))
    pending=[r for r in before if r['existing_sha256']!=r['sha256']]
    locals_=threading.local();connections=[];connlock=threading.Lock()
    def upload(row):
        if not hasattr(locals_,'sftp'):
            c=connect();locals_.sftp=c.open_sftp()
            with connlock:connections.append(c)
        s=locals_.sftp;tmp=row['server']+'.incoming';offset=row['incoming_bytes'];assert offset<=row['bytes']
        if offset:
            import hashlib
            h=hashlib.sha256()
            with Path(row['local']).open('rb') as f:
                left=offset
                while left:
                    block=f.read(min(left,8*1024**2));h.update(block);left-=len(block)
            assert h.hexdigest()==row['incoming_sha256'],'Partial upload prefix differs: preserve and diagnose'
            with Path(row['local']).open('rb') as f,s.open(tmp,'r+b') as out:
                f.seek(offset);out.seek(offset);out.set_pipelined(True)
                for block in iter(lambda:f.read(256*1024),b''):out.write(block)
        else:s.put(row['local'],tmp)
        try:s.posix_rename(tmp,row['server'])
        except IOError:s.rename(tmp,row['server'])
        return row
    verified_count=0;uploaded_bytes=0
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(upload,r) for r in pending]
            for z in as_completed(futures):
                row=z.result();verified_count+=1;uploaded_bytes+=row['bytes']
                if verified_count%10==0 or row['bytes']>100*1024**2:
                    progress={'stage':'UPLOADING','completed':verified_count,'expected':len(pending),'uploaded_bytes':uploaded_bytes,'time':utc_now()}
                    write_json(ROOT/'reports/required_assets_progress.json',progress);print(progress,flush=True)
    finally:
        for c in connections:c.close()
    verify="from pathlib import Path;import json,hashlib\nrows=json.loads(%r)\nfor row in rows:\n p=Path(row['server']);assert p.stat().st_size==row['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)\nprint(json.dumps({'stage':'ALL_REQUIRED_ASSETS_SHA_VERIFIED','files':len(rows),'bytes':sum(r['bytes'] for r in rows)}))"%json.dumps(rows)
    with connect() as c:result=json.loads(command(c,"/root/miniconda3/bin/python - <<'PY'\n"+verify+"\nPY",300))
    result.update(time=utc_now(),inventory_sha256=sha(ROOT/'reports/required_assets_inventory.json'),protected_Q_starts_and_parent=True)
    write_json(ROOT/'reports/required_assets_receipt.json',result);print(result,flush=True)

if __name__=='__main__':main()
