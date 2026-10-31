import json,subprocess,sys,time,re,os
from pathlib import Path
r=Path('/root/autodl-tmp/hr100');os.chdir(r/'code')
os.environ['PYTHONPATH']=str(r/'code');os.environ['OMP_NUM_THREADS']='4'
ck=str(r/'trial_b/best_mIoU_iter_3000.pth')
for split in ['val','test']:
 (r/'ensemble_state.json').write_text(json.dumps({'phase':split}))
 log=r/('ensemble_'+split+'.log')
 with log.open('wb') as out:
  rc=subprocess.call([sys.executable,'tools/test.py','configs/lesion/exp/hr100_ensemble_'+split+'.py',ck,'--eval','mIoU'],stdout=out,stderr=out)
 if rc:raise RuntimeError('evaluation failed '+split)
 matches=re.findall(r'^global\s+([0-9.]+)',log.read_text(),re.M)
 assert matches, 'missing evaluation summary'
 if split=='val':assert abs(float(matches[-1])-39.9025)<0.03, 'routing validation inconsistent'
(r/'ensemble_state.json').write_text(json.dumps({'phase':'backup','test_mIoU':float(matches[-1])}))
until=time.time()+90
while time.time()<until and not (r/'ensemble_backup_ack').exists():time.sleep(3)
