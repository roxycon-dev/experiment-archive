"""Exercise real custody/unlink guards against corruption, active last and absent endpoints."""
import sys,tempfile,io,contextlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,digest,utc_now
import storage
from ops import cold_archive

def main():
    original=(storage.ROOT,cold_archive.ROOT,cold_archive.connect,cold_archive.command)
    with tempfile.TemporaryDirectory(prefix='storage_gate_',dir=ROOT/'reports') as directory:
        base=Path(directory);assert base.resolve().is_relative_to((ROOT/'reports').resolve())
        local=base/'local';remote=base/'remote';output=local/'server_results';run='C_seed42';u=32000
        (remote/'runs'/run/f'val_u{u}').mkdir(parents=True);(local/'configs').mkdir(parents=True);(output/'reports/storage_requests').mkdir(parents=True)
        write_json(local/'configs/runtime.local.json',{'server':{'remote_root':str(remote),'instance_id':'TEST_ONLY'},'server_python':'TEST_ONLY'})
        files=[]
        for name in ['checkpoint_u32000.pth',*[f'val_u{u}/image{i:03d}.npz' for i in range(76)]]:
            p=remote/'runs'/run/name;p.write_bytes(('full precision immutable '+name).encode())
            rel=p.relative_to(remote).as_posix();q=output/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            files.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
        last=remote/'runs'/run/'last.pth';shutil.copy2(remote/files[0]['path'],last)
        write_json(remote/'runs'/run/'status.json',{'stage':'TRAINING','completed':32000})
        request={'run':run,'update':u,'files':files,'metrics_sha256':'test','identity_sha256':'test','time':utc_now(),'mutable_last_checkpoint_excluded':True}
        request['request_sha256']=digest(request)
        for r in (remote,output):write_json(r/f'reports/storage_requests/{run}_u{u}.json',request)
        class Client:
            def __enter__(self):return self
            def __exit__(self,*args):pass
        def execute(client,command,timeout=60):
            code=command.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0];out=io.StringIO()
            with contextlib.redirect_stdout(out):exec(compile(code,'actual_remote_guard','exec'),{})
            return out.getvalue()
        storage.ROOT=remote;cold_archive.ROOT=local;cold_archive.connect=lambda:Client();cold_archive.command=execute
        # One corrupted local NPZ must block ALL remote deletion, including the checkpoint.
        corrupt=output/files[-1]['path'];corrupt.write_bytes(b'corrupted transfer')
        assert cold_archive.release_verified()==[] and all((remote/r['path']).exists() for r in files)
        shutil.copy2(remote/files[-1]['path'],corrupt)
        receipts=cold_archive.release_verified();assert len(receipts)==1
        assert not any((remote/r['path']).exists() for r in files) and last.exists(),'Active rolling state must survive'
        assert storage.checkpoint_sha(files[0]['path'])==files[0]['sha256'],'Offloaded identity must remain usable by selection'
        write_json(remote/'runs'/run/'status.json',{'stage':'COMPLETE','completed':32000})
        cold_archive.release_verified();assert not last.exists(),'Complete identical redundant last should be reclaimed'
        assert all(sha(output/r['path'])==r['sha256'] for r in files)
        invalid=dict(request,run='UNREGISTERED_seed42');invalid['request_sha256']=digest({k:v for k,v in invalid.items() if k!='request_sha256'})
        write_json(output/'reports/storage_requests/invalid.json',invalid)
        try:cold_archive.release_verified()
        except AssertionError:pass
        else:raise AssertionError('Unregistered endpoint release accepted')
        (output/'reports/storage_requests/invalid.json').unlink()
        # The2024 fixed parent stays available to initialize ALL second-seed
        # arms even after its unrelated-to-training endpoint probabilities move.
        pr=remote/'runs/PARENT_F12_seed2024';(pr/'val_u12000').mkdir(parents=True)
        protected=pr/'checkpoint_u12000.pth';protected.write_bytes(b'protected parent full learned state')
        write_json(pr/'identity.json',{'seed':2024});write_json(pr/'status.json',{'stage':'COMPLETE','completed':12000})
        pf=[]
        for i in range(76):
            p=pr/f'val_u12000/image{i:03d}.npz';p.write_bytes(b'FP32_parent_probability');q=output/p.relative_to(remote);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            pf.append({'sid':p.stem,'bytes':p.stat().st_size,'sha256':sha(p)})
        write_json(pr/'val_u12000/metrics.json',{'probability_files':pf})
        storage.publish_parent_probabilities(pr)
        source=remote/'reports/storage_requests/PARENT_F12_seed2024_u12000.json';shutil.copy2(source,output/'reports/storage_requests'/source.name)
        cold_archive.release_verified();assert protected.exists() and not list((pr/'val_u12000').glob('*.npz'))
    storage.ROOT,cold_archive.ROOT,cold_archive.connect,cold_archive.command=original
    write_json(ROOT/'reports/storage_acceptance.json',{'status':'PASS','time':utc_now(),'checks':['Corrupted local NPZ blocks entire release','All77 exact-SHA artifacts preserved locally before remote unlink','Active last preserved','Complete identical redundant last reclaimed','Missing checkpoint identity resolved from verified custody','Unregistered endpoint rejected','2024 initializer checkpoint protected while parent probabilities archived']})
    print('STORAGE_GUARDS_PASS')

if __name__=='__main__':main()
