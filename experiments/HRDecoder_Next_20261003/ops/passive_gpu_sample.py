"""A bounded read-only GPU trace to inform the next experiment; no training changes."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,connect,remote_exec,write_json
CODE=r'''
import pathlib,json,subprocess,time
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Next_20261003');rows=[]
def progress():
 return {p.parent.name:json.loads(p.read_text()).get('update') for p in (r/'work/M_seed42').glob('M*/progress.json')}
start=progress()
for _ in range(30):
 text=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
  '--format=csv,noheader,nounits'],text=True).strip().split(',')
 rows.append(dict(time=time.time(),gpu_percent=float(text[0]),memory_utilization=float(text[1]),
  memory_MiB=float(text[2]),power_W=float(text[3])))
 time.sleep(1)
print(json.dumps(dict(rows=rows,progress_start=start,progress_end=progress(),
 stage=json.loads((r/'reports/server_stage.json').read_text()),mode='read_only_nvidia_smi',
 M_log_tail=(r/'reports/M.log').read_text()[-600:])))
'''
def main():
 c=connect()
 try:result=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+CODE+"\nPY",45))
 finally:c.close()
 values=[r['gpu_percent'] for r in result['rows']]
 result['summary']=dict(samples=len(values),mean_percent=sum(values)/len(values),
     min_percent=min(values),max_percent=max(values),below_50_percent=sum(v<50 for v in values)/len(values),
     caution='A short passive trace locates variability, not an isolated bottleneck or full-run efficiency.')
 write_json(ROOT/'reports/next_gpu_passive_trace.json',result)
 print(json.dumps(result['summary']))
if __name__=='__main__':main()
