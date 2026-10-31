"""Read-only preparation audit, using the two already authorized task connections."""
import json
from pathlib import Path
import shlex
from training_remote_control import connect, execute, ROOT, REMOTE, LAN

ROOT_CODE = r'''
from pathlib import Path
import json,shutil,subprocess,importlib.metadata
exp=Path('/root/autodl-tmp/paper_supplement_20261023_run')
paths=[exp/'remote_seed.zip',exp/'A_exact.zip',exp/'ddr_split613.tar']
weights=exp/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights'
paths+=list(weights.glob('*.pth'))
print(json.dumps({'files':[{'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None} for p in paths],
'disk':shutil.disk_usage(exp)._asdict(),
'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)}))
'''
LAN_CODE = r'''
from pathlib import Path
import json,shutil,subprocess
r=Path('/home/studentuser001/paper_supplement_20261023')
print(json.dumps({'disk':shutil.disk_usage(r)._asdict(),
'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
'processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),
'task_directories':[p.name for p in r.iterdir() if p.is_dir()],
'baseline_workdir':[str(p) for p in (r/'baseline_v1').rglob('*.log.json')],
'conda':str(r/'legacy_env/bin/conda')}))
'''

with connect() as client:
    result={}
    for target,code,python in [('root',ROOT_CODE,'/root/miniconda3/bin/python'),('59',LAN_CODE,'/usr/bin/python3')]:
        r=execute(client,python+' -c '+shlex.quote(code),target)
        assert r['exit_code']==0,r
        result[target]=json.loads(r['stdout'])
    (ROOT/'results/training_resource_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
