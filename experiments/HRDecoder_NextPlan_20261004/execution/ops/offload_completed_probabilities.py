"""Free server space only after completed-run probability files are verified locally."""
import argparse
import json
from pathlib import Path
import shlex
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,sha,utc_now,write_json
from ops.remote import connect,command


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    allowed={f'{arm}_seed{s}' for arm in ('E_NEG','F4','F12') for s in (42,2024)}
    assert args.run in allowed
    runtime=read_json(ROOT/'configs/runtime.local.json');remote=runtime['server']['remote_root']
    with connect() as client:
        with client.open_sftp() as sftp:
            with sftp.open(remote+'/runs/'+args.run+'/status.json') as f:status=json.loads(f.read().decode())
            assert status['stage']=='COMPLETE'
            endpoint=status['completed']
            relative=f'runs/{args.run}/val_u{endpoint}'
            with sftp.open(remote+'/'+relative+'/metrics.json') as f:metrics=json.loads(f.read().decode())
            assert metrics['n_images']==76
            records=[]
            for item in metrics['probability_files']:
                assert '/' not in item['sid'] and '\\' not in item['sid']
                rel=relative+'/'+item['sid']+'.npz'
                local=ROOT/'server_results'/rel;local.parent.mkdir(parents=True,exist_ok=True)
                if not local.exists() or sha(local)!=item['sha256']:
                    partial=local.with_suffix('.npz.download')
                    sftp.get(remote+'/'+rel,str(partial))
                    assert sha(partial)==item['sha256'],'Local transfer SHA differs'
                    partial.replace(local)
                records.append({'path':rel,'sha256':item['sha256'],'bytes':local.stat().st_size})
        code='''
from pathlib import Path
import hashlib,json
r=Path(%r);items=%r;deleted=0
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
 return d.hexdigest()
assert json.loads((r/'runs'/%r/'status.json').read_text())['stage']=='COMPLETE'
for x in items:
 p=r/x['path'];assert p.resolve().is_relative_to(r/'runs'/%r)
 assert p.suffix=='.npz' and not p.is_symlink()
 if p.exists():
  assert h(p)==x['sha256'];p.unlink();deleted+=x['bytes']
print(json.dumps({'deleted_bytes_after_verified_local_backup':deleted,'files':len(items)}))
'''%(remote,records,args.run,args.run)
        result=json.loads(command(client,shlex.quote(runtime['server_python'])+" - <<'PY'\n"+code+"\nPY",timeout=120))
    receipt={'stage':'COMPLETED_PROBABILITIES_BACKED_UP_AND_SERVER_SPACE_FREED','run':args.run,
             'files':records,'result':result,'scripts_logs_models_rng_and_metrics_untouched':True,'time':utc_now()}
    write_json(ROOT/f'reports/probability_offload_{args.run}.json',receipt)
    print(json.dumps({'stage':receipt['stage'],'run':args.run,**result}))


if __name__=='__main__':main()
