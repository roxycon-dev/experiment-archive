"""Only experiment input paths and data-disk inventory; no hardware scan."""
import sys,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.remote import connect,command
from qlib.common import read_json,write_json,utc_now

def main():
    runtime=read_json(ROOT/'configs/runtime.local.json')
    code="""
import os,json,shutil
from pathlib import Path
base=Path('/root/autodl-tmp');usage=shutil.disk_usage(base)
paths=%r
inventory=[]
for root in sorted(base.iterdir()):
 total=0;protected_count=0
 if root.is_file():total=root.stat().st_size
 elif root.is_dir():
  for directory,_,names in os.walk(root):
   for name in names:
    p=Path(directory)/name
    if p.is_symlink():continue
    try:total+=p.stat().st_size
    except FileNotFoundError:continue
    protected_count+=p.suffix in ('.py','.sh','.log','.json','.jsonl','.md')
 inventory.append({'path':str(root),'bytes':total,'scripts_logs_reports_count':protected_count})
print(json.dumps({'disk':{'total':usage.total,'free':usage.free},'required_paths':{p:Path(p).exists() for p in paths},'top_level_inventory':inventory}))
"""%[runtime['server_python'],'/root/miniconda3/bin/python','/root/HRDecoder/data/DDR_split_613',
         '/root/autodl-tmp/HRDecoder_Q_20261003','/root/autodl-tmp/HRDecoder_Finetune_20261004','/root/autodl-tmp/HRDecoder_Ready_20261002']
    with connect() as client:
        report=json.loads(command(client,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",120))
    report['time']=utc_now();write_json(ROOT/'reports/assigned_server_inventory.json',report);print(json.dumps(report))

if __name__=='__main__':main()
