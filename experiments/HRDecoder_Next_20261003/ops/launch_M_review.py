"""Small CPU-only review from saved paired best probabilities, no inference."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,REMOTE,connect,remote_exec,write_json
CODE=r'''
from pathlib import Path
import subprocess,os,json,time
r=Path('/root/autodl-tmp/HRDecoder_Next_20261003')
selection=json.loads((r/'work/M_seed42/selection.json').read_text())
assert selection['adaptive_stop']['status']=='PAIRED_COST_STOP_AT_2000'
hist={arm:json.loads((r/'work/M_seed42'/arm/'history.json').read_text()) for arm in ['M0','M1']}
assert all([x['update'] for x in h]==[0,1000,2000] for h in hist.values())
assert all(len(list((r/'work/M_seed42'/arm/'val_u1000').glob('*.npz')))==76 for arm in hist)
def cpu_limit():os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
with (r/'reports/M_error_review.log').open('ab') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','ops/error_review.py',
 '--cache',str(r/'work/M_seed42/M1/val_u1000'),
 '--baseline',str(r/'work/M_seed42/M0/val_u1000'),
 '--out',str(r/'reports/M_error_gallery'),'--top','1'],cwd=r,
 stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=cpu_limit,
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
print(json.dumps(dict(pid=p.pid,time=time.time(),scope='Saved 1000-step M1 versus M0; CPU only; 12 count-selected crops')))
'''
def main():
 c=connect()
 try:
  with c.open_sftp() as sftp:sftp.put(str(ROOT/'ops/error_review.py'),REMOTE+'/ops/error_review.py')
  data=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+CODE+"\nPY",20))
  write_json(ROOT/'reports/M_error_review_pid.json',data);print(json.dumps(data))
 finally:c.close()
if __name__=='__main__':main()
