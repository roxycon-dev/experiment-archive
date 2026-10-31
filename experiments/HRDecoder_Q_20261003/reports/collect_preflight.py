from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from ops.remote import connect
from qlib.common import sha, utc_now, write_json

remote_root = '/root/autodl-tmp/HRDecoder_Q_20261003'
files = ['reports/environment_freeze.txt', 'reports/legacy_cache_cleanup.json',
         'reports/gpu_preflight.json', 'reports/learnability.json',
         'reports/pretraining_identity.json', 'reports/formal_readiness.json',
         'reports/preflight_full_val/metrics.json', 'reports/preparation_status.json']
received = []
with connect() as client:
    with client.open_sftp() as sftp:
        for rel in files:
            remote = remote_root+'/'+rel
            try:
                before = sftp.stat(remote)
            except FileNotFoundError:
                continue
            target = root/'server_results'/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, str(target))
            _, stdout, _ = client.exec_command("sha256sum '"+remote+"'", timeout=20)
            remote_sha = stdout.read().decode().split()[0]
            assert sha(target) == remote_sha
            received.append({'path': rel, 'bytes': before.st_size, 'sha256': remote_sha})
write_json(root/'reports/preflight_backup_receipt.json', {'files': received, 'verified_at': utc_now()})
print(json.dumps({'verified_files': len(received)}))
