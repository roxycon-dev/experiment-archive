"""Two extra read-only receivers for immutable final validation probabilities."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.remote import connect, command
from ops.transfer_acceleration import install
from qlib.common import read_json, sha, utc_now, write_json

install()
OUTPUT = ROOT / 'server_results'
RECEIPT = ROOT / 'reports/parallel_probability_backup.json'
LOCK = threading.Lock()
STATE = {}


def official_backup_finished():
    p = ROOT / 'reports/final_backup_receipt.json'
    return p.exists() and read_json(p).get('stage') == 'FULL_BACKUP_VERIFIED'


def save(**updates):
    with LOCK:
        STATE.update(updates)
        STATE['updated_at'] = utc_now()
        write_json(RECEIPT, STATE)


def receiver(worker, items, remote):
    verified = []
    deferred = []
    with connect() as client, client.open_sftp() as sftp:
        for item in items:
            if official_backup_finished():
                deferred.extend(x['path'] for x in items if x['path'] not in verified)
                break
            target = (OUTPUT / item['path']).resolve()
            assert target.is_relative_to(OUTPUT.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == item['sha256'], 'Existing immutable file differs'
            elif target.with_suffix(target.suffix + '.download').exists():
                # The original attendant owns this file; give it sole responsibility.
                deferred.append(item['path'])
                continue
            else:
                temporary = target.with_suffix(target.suffix + f'.parallel{worker}.download')
                last_saved = [0.0]

                def progress(received, total):
                    now = time.monotonic()
                    if now - last_saved[0] < 5:
                        return
                    last_saved[0] = now
                    with LOCK:
                        STATE['active'][str(worker)] = {
                            'path': item['path'], 'received': received, 'bytes': total}
                        STATE['updated_at'] = utc_now()
                        write_json(RECEIPT, STATE)

                sftp.get(remote + '/' + item['path'], str(temporary), callback=progress)
                assert temporary.stat().st_size == item['bytes']
                assert sha(temporary) == item['sha256'], 'Received probability SHA differs'
                if target.exists():
                    assert sha(target) == item['sha256'], 'Concurrent completed file differs'
                    # Keep the independently verified duplicate temporary as evidence.
                else:
                    temporary.replace(target)
            verified.append(item['path'])
            with LOCK:
                STATE['verified'].append(item['path'])
                STATE['active'].pop(str(worker), None)
                STATE['updated_at'] = utc_now()
                write_json(RECEIPT, STATE)
                print(len(STATE['verified']), '/', len(STATE['files']), item['path'], flush=True)
    return verified, deferred


def main():
    assert not RECEIPT.exists(), 'Inspect existing receiver receipt before starting another'
    runtime = read_json(ROOT / 'configs/runtime.local.json')
    remote = runtime['server']['remote_root']
    code = '''
from pathlib import Path
import json,hashlib
r=Path(%r);run=r/'runs/F12_seed42'
assert json.loads((run/'status.json').read_text())['stage']=='COMPLETE'
files=[]
for p in run.rglob('*.npz'):
 if not p.is_file() or p.is_symlink():continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 files.append({'path':p.relative_to(r).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(files))
''' % remote
    with connect() as client:
        files = json.loads(command(client, shlex.quote(runtime['server_python']) +
                                   ' -c ' + shlex.quote(code), 180))
    assert files, 'No final probabilities found'
    assert shutil.disk_usage(OUTPUT).free > sum(x['bytes'] for x in files) + 2 * 1024**3
    save(stage='RECOVERING', started_at=utc_now(), workers=2, files=files,
         verified=[], active={}, remote_files_deleted=False, remote_source_changed=False,
         primary_receiver='ops/attendant_transfer.py',
         purpose='Receive completed F12 probabilities alongside the primary checkpoint transfer')
    # Receive from the opposite end to minimize overlap with the primary mirror.
    ordered = sorted(files, key=lambda x: x['path'], reverse=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(receiver, n, ordered[n::2], remote) for n in range(2)]
        results = [f.result() for f in futures]
    deferred = sorted(set(x for _, paths in results for x in paths))
    save(stage='PARALLEL_PROBABILITIES_VERIFIED', completed_at=utc_now(), deferred=deferred,
         note='The primary attendant retains responsibility for full inventory SHA verification and shutdown.')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if official_backup_finished():
            save(stage='SUPERSEDED_BY_FULL_VERIFIED_BACKUP', closed_at=utc_now())
        else:
            save(stage='RECEIVER_ERROR', reason=str(error), automatic_retry=False)
            raise
