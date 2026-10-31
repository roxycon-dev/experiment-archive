"""Upload scheduling/timing helpers, start detached controller, archive initial smoke."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"local_preparation"))
from remote import connect
ROOT = "/root/autodl-tmp/HRDecoder_v3_20261008"
client = connect()
with client.open_sftp() as ftp:
    for name in ("train_performance.py", "gpu_campaign_controller.py", "ram_loader.py", "channels_last_runtime.py"):
        ftp.put(str(HERE/name), ROOT+"/training_execution/"+name)
    destination = HERE.parent/"cache_execution/server_receipts/training_runs/v3"
    destination.mkdir(parents=True, exist_ok=True)
    ftp.get(ROOT+"/training_runs/v3/GPU_SMOKE_PASS.json", str(destination/"GPU_SMOKE_PASS.json"))
command = """cd /root/autodl-tmp/HRDecoder_v3_20261008/training_execution
rm -f ../reports/shutdown_hold.enabled
/root/miniconda3/bin/python -c 'import subprocess; f=open("../reports/gpu_controller.log", "ab", buffering=0); p=subprocess.Popen(["/root/miniconda3/bin/python", "-u", "gpu_campaign_controller.py"], stdin=subprocess.DEVNULL, stdout=f, stderr=f, start_new_session=True); print(p.pid)'
"""
_, stdout, stderr = client.exec_command(command)
print(stdout.read().decode(), end="")
error = stderr.read().decode()
assert stdout.channel.recv_exit_status() == 0, error
print(error, end="")
client.close()
