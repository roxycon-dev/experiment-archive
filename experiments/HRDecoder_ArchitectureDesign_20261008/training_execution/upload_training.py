"""Upload prepared code, CPU evidence and a pinned source manifest."""
import hashlib
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"local_preparation"))
from remote import connect

names = ["model.py", "private_modules.py", "objectives.py", "data.py", "train.py",
    "check_server_init.py", "check_server_route.py", "check_optimizer_resume.py", "verify_server_preparation.py", "launch_gpu.sh", "resume_gpu.sh",
    "CPU_PREPARATION_PASS.json", "REAL_CACHE_PIPELINE_PASS.json", "OPTIMIZER_RESUME_PASS.json", "review_notes.md"]
for name in names: assert (HERE/name).is_file(), name
manifest = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
(HERE/"execution_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
client = connect()
try:
    _, stdout, stderr = client.exec_command("mkdir -p /root/autodl-tmp/HRDecoder_v3_20261008/training_execution")
    assert stdout.channel.recv_exit_status() == 0, stderr.read().decode()
    with client.open_sftp() as ftp:
        for name in names+["execution_manifest.json"]:
            ftp.put(str(HERE/name), "/root/autodl-tmp/HRDecoder_v3_20261008/training_execution/"+name)
finally: client.close()
print(json.dumps({"files": len(names)+1, "uploaded_and_sha_manifest_created": True}))
