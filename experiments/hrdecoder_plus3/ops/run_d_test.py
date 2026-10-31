import json,subprocess,sys,time,re,os
from pathlib import Path
r=Path('/root/autodl-tmp/hrplus');os.chdir(r/'repo');os.environ['PYTHONPATH']=str(r/'repo');os.environ['OMP_NUM_THREADS']='4'
(r/'d_test_state.json').write_text(json.dumps({'phase':'test'}))
log=r/'d_test.log'
with log.open('wb') as out:
 rc=subprocess.call([sys.executable,'tools/test.py','configs/lesion/exp/hrplus_patch_dice.py',str(r/'D_0913_retry/best_mIoU_iter_2000.pth'),'--eval','mIoU'],stdout=out,stderr=out)
if rc:raise RuntimeError('test failed')
v=float(re.findall(r'^global\s+([0-9.]+)',log.read_text(),re.M)[-1])
(r/'d_test_state.json').write_text(json.dumps({'phase':'backup','test_mIoU':v,'test_evaluated':True}))
until=time.time()+90
while time.time()<until and not (r/'d_test_backup_ack').exists():time.sleep(3)
