"""Bounded read-only snapshot using the task's existing key and known host."""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from mirror_remote import ROOT,REMOTE,connect,remote_exec,write_json
CODE=r'''
import pathlib,json,subprocess,time
r=pathlib.Path('/root/autodl-tmp/HRDecoder_Next_20261003')
def read(p):
 return json.loads(p.read_text()) if p.exists() else None
def command(args):
 p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 return p.stdout
progress={}
for p in (r/'work').glob('M_seed*/M*/progress.json'):progress[str(p.parent.relative_to(r))]=read(p)
histories={}
for p in (r/'work').glob('M_seed*/M*/history.json'):
 histories[str(p.parent.relative_to(r))]=[dict(update=row['update'],
  solo=row['solo']['mIoU_lesion4'],fixed_half=row['fixed_half']['mIoU_lesion4'],
  solo_counts=row['solo']['counts'],half_counts=row['fixed_half']['counts']) for row in read(p)]
memory={}
for line in pathlib.Path('/sys/fs/cgroup/memory.stat').read_text().splitlines():
 k,v=line.split()
 if k in ['anon','file','shmem','file_mapped','inactive_file','active_file']:memory[k]=int(v)
test_progress=read(r/'work/test_progress.json')
test_progress_summary=dict(n_images=len(test_progress['images']),lock_sha256=test_progress['lock_sha256']) if test_progress else None
gallery=r/'reports/M_error_gallery'
review_summary=dict(images=len(list(gallery.glob('*.png'))),done=(gallery/'index.html').exists(),
 log=(r/'reports/M_error_review.log').read_text()[-1000:] if (r/'reports/M_error_review.log').exists() else '')
print(json.dumps(dict(time=time.time(),stage=read(r/'reports/server_stage.json'),progress=progress,histories=histories,
 test_progress_summary=test_progress_summary,M_review=review_summary,
 M_log=(r/'reports/M.log').read_text()[-1800:] if (r/'reports/M.log').exists() else '',
 completion=read(r/'reports/campaign_completion.json'),
 adaptive_budget_decision=read(r/'reports/adaptive_budget_decision.json'),
 adaptive_budget_guard_error=read(r/'reports/adaptive_budget_guard_error.json'),
 adaptive_budget_guard_log=(r/'reports/adaptive_budget_guard.log').read_text()[-1800:] if (r/'reports/adaptive_budget_guard.log').exists() else '',
 final_test=read(r/'work/final_test.json'),
 memory_current=int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text()),memory=memory,
 memory_events=pathlib.Path('/sys/fs/cgroup/memory.events').read_text(),
 disk=command(['df','-h','/dev/shm','/root/autodl-tmp']),
 gpu=command(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader']),
 processes=command(['ps','-eo','pid,ppid,etime,pcpu,rss,stat,args']))))
'''
def main():
 c=connect()
 try:s=json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+CODE+"\nPY",30))
 finally:c.close()
 s['processes']='\n'.join(line for line in s['processes'].splitlines() if any(x in line for x in ['run_campaign.py','continue_campaign.py','final_test.py','ops/final_g.py']))
 write_json(ROOT/'reports/latest_direct_probe.json',s)
 print(json.dumps(s,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
