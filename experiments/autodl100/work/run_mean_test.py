import json,subprocess,sys,time,re,os
from pathlib import Path
r=Path('/root/autodl-tmp/hr100');os.chdir(r/'code');os.environ['PYTHONPATH']=str(r/'code');os.environ['OMP_NUM_THREADS']='4'
(r/'mean_test_state.json').write_text(json.dumps({'phase':'test'}))
log=r/'mean_test.log'
with log.open('wb') as out:
 rc=subprocess.call([sys.executable,'tools/test.py','configs/lesion/exp/hr100_mean_test.py',str(r/'trial_b/best_mIoU_iter_3000.pth'),'--eval','mIoU'],stdout=out,stderr=out)
if rc:raise RuntimeError('validation failed')
v=float(re.findall(r'^global\s+([0-9.]+)',log.read_text(),re.M)[-1])
(r/'mean_test_state.json').write_text(json.dumps({'phase':'backup','test_mIoU':v,'test_evaluated':True}))
until=time.time()+90
while time.time()<until and not (r/'mean_test_backup_ack').exists():time.sleep(3)
