"""Copy immutable completed-node evidence; no training changes or deletion."""
from pathlib import Path
import argparse
import json
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from ops.remote import connect
from qlib.common import read_json, sha, utc_now, write_json

ap = argparse.ArgumentParser()
ap.add_argument('--arm', choices=['Q_DDR', 'Q_MIX'], default='Q_DDR')
ap.add_argument('--seed', choices=[42, 2024], type=int, default=42)
ap.add_argument('--checkpoints', action='store_true')
args = ap.parse_args()
remote_root = '/root/autodl-tmp/HRDecoder_Q_20261003'
rel_run = f'runs/{args.arm}_seed{args.seed}'
receipt_path = root/'reports'/f'node_backup_{args.arm}_seed{args.seed}.json'
prior = read_json(receipt_path)['files'] if receipt_path.exists() else []
known = {r['path']: r for r in prior}
with connect() as client:
    with client.open_sftp() as sftp:
        names = sftp.listdir(remote_root+'/'+rel_run)
        files = [rel_run+'/'+name for name in ('identity.json', 'initialization.json') if name in names]
        for name in sorted(names):
            if name.startswith('val_u') and name[5:].isdigit():
                path = rel_run+'/'+name+'/metrics.json'
                try:
                    sftp.stat(remote_root+'/'+path)
                except FileNotFoundError:
                    continue
                files.append(path)
            elif args.checkpoints and name.startswith('checkpoint_u') and name.endswith('.pth'):
                files.append(rel_run+'/'+name)
        for rel in files:
            target = root/'server_results'/rel
            if rel in known and target.exists() and sha(target) == known[rel]['sha256']:
                continue
            remote = remote_root+'/'+rel
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix+'.download')
            sftp.get(remote, str(temporary))
            _, stdout, _ = client.exec_command("sha256sum '"+remote+"'", timeout=20)
            remote_sha = stdout.read().decode().split()[0]
            assert sha(temporary) == remote_sha, rel
            temporary.replace(target)
            known[rel] = {'path': rel, 'bytes': target.stat().st_size, 'sha256': remote_sha}
            write_json(receipt_path, {'files': list(known.values()), 'verified_at': utc_now()})
write_json(receipt_path, {'files': list(known.values()), 'verified_at': utc_now()})
print(json.dumps({'verified_files': len(known), 'receipt': str(receipt_path)}, ensure_ascii=False))
