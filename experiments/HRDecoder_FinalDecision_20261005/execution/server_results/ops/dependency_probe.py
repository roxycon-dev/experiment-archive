"""Required Python/Torch availability and space for isolated dependencies only."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.remote import connect,command
from qlib.common import write_json
code="""
import torch,sys,shutil,json
print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,dependency_disk_free=shutil.disk_usage('/root/miniconda3').free)))
"""
with connect() as client:result=json.loads(command(client,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",60))
write_json(ROOT/'reports/required_dependency_state.json',result);print(result)
