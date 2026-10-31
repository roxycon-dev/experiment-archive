"""One finite online late-node export for the final registered paired run.

Start only after the formal MIX2024 identity is known. No training is launched
or retried here. Remote deletion is limited to completed, double-SHA-backed
probability NPZ; all models and count evidence remain available.
"""
import argparse
import concurrent.futures
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reports'))
import handoff_after_qmix16 as h
from qlib.common import write_json, utc_now

h.RUN = 'Q_MIX_seed2024'
h.STATE = ROOT / 'reports/mix2024_late_offload_state.json'
h.DELETED = ROOT / 'reports/mix2024_late_cache_offload_receipt.json'
h.ALLOWED_CACHE_UPDATES = (12000, 16000, 20000, 24000)
h.PLAN_PREFIX = 'mix2024_late_cache_offload_plan'
owns_state = False


def wait_node(update, timeout_minutes, final=False):
    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        x = h.execute(f'''
import pathlib,json,shutil
r=pathlib.Path({h.REMOTE!r});run=r/'runs/{h.RUN}'
s=json.loads((run/'status.json').read_text())
assert s['identity_sha256']=={h.IDENTITY!r} and s['endpoint']==24000
receipt=json.loads((r/'reports/qmix_formal_launch_receipt.json').read_text())
assert receipt['arm']=='Q_MIX' and receipt['seed']==2024 and receipt['endpoint']==24000
pid=(r/'reports/formal.pid').read_text().strip();cmd=pathlib.Path('/proc')/pid/'cmdline'
alive=cmd.exists() and b'train_run.py' in cmd.read_bytes()
m=run/'val_u{update}/metrics.json'
ready=False
if m.exists():
 v=json.loads(m.read_text())
 ready=v['update']=={update} and v['n_images']==76 and len(v['probability_files'])==76
print(json.dumps(dict(status=s,node_ready=ready,formal_alive=alive,free_bytes=shutil.disk_usage(r).free)))
''')
        status = x['status']
        if status['stage'] in ('TECHNICAL_PAUSE', 'SERIOUS_FAILURE_STOP'):
            raise RuntimeError('Registered training paused; export continuation stopped: ' + str(status))
        if x['node_ready'] and (not final or (status['stage'] == 'COMPLETE' and not x['formal_alive'])):
            return x
        if not x['formal_alive'] and status['stage'] != 'COMPLETE':
            raise RuntimeError('Formal training process exited without completed evidence')
        if time.monotonic() > deadline:
            raise TimeoutError('Finite late-node wait expired; no automatic retry')
        h.state('WAITING_FOR_REGISTERED_LATE_NODE', update=update,
                formal_status=status, free_bytes=x['free_bytes'])
        time.sleep(30)


def freeze_final_inventory():
    return h.execute(f'''
import pathlib,json
from qlib.common import sha,utc_now
r=pathlib.Path({h.REMOTE!r});run=r/'runs/{h.RUN}'
s=json.loads((run/'status.json').read_text())
assert s['stage']=='COMPLETE' and s['completed']==24000 and s['endpoint']==24000
assert s['identity_sha256']=={h.IDENTITY!r}
files=[]
for p in sorted(run.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():
  assert not p.name.endswith(('.tmp','.download'))
  files.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
print(json.dumps(dict(run={h.RUN!r},status=s,files=files,created_at=utc_now())))
''')


def main():
    global owns_state
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity', required=True)
    args = parser.parse_args()
    assert len(args.identity) == 64 and all(c in '0123456789abcdef' for c in args.identity)
    h.IDENTITY = args.identity
    assert not h.STATE.exists(), 'Inspect the existing worker; never duplicate or automatically retry it'
    owns_state = True
    h.state('REGISTERED_FINITE_LATE_NODE_EXPORT', run=h.RUN,
            identity_sha256=h.IDENTITY, first_node_timeout_minutes=180,
            later_node_timeout_minutes=75)
    for update in (12000, 16000, 20000):
        wait_node(update, 180 if update == 12000 else 75)
        h.offload(h.cache_plan(update), str(update))
    wait_node(24000, 75, final=True)
    inventory = freeze_final_inventory()
    merged = {row['path']: row for row in inventory['files']}
    for path, row in h.verified.items():
        if path in merged:
            assert merged[path] == row
        merged[path] = row
    plan = {**inventory, 'files': sorted(merged.values(), key=lambda row: row['path']),
            'earlier_online_offloaded_files_included': True}
    write_json(ROOT / f'reports/backup_plan_{h.RUN}_24k.json', plan)
    h.offload(h.cache_plan(24000), '24000')
    write_json(h.DELETED, dict(status='COMPLETE', deleted_files=list(h.deleted.values()),
        deleted_bytes=sum(row['bytes'] for row in h.deleted.values()),
        each_local_and_remote_SHA_verified=True, local_files_deleted=False,
        completed_at=utc_now()))
    pending = [row for row in plan['files'] if row['path'] not in h.verified]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(h.copy_file, row) for row in pending]
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result()
            except BaseException:
                for task in futures:
                    task.cancel()
                raise
            h.verified[row['path']] = row
            h.state('FINAL_COMPLETE_RUN_BACKUP', run=h.RUN)
    assert set(h.verified) == set(merged)
    receipt = dict(complete=True, run=h.RUN, endpoint=24000,
        identity_sha256=h.IDENTITY, verified_at=utc_now(),
        files=sorted(h.verified.values(), key=lambda row: row['path']),
        total_bytes=sum(row['bytes'] for row in h.verified.values()),
        remote_deletion_scope='Only completed double-SHA-backed 12k/16k/20k/24k probability NPZ')
    write_json(ROOT / f'reports/run_backup_{h.RUN}_24k.json', receipt)
    write_json(ROOT / f'reports/run_backup_{h.RUN}.json', receipt)
    h.state('COMPLETE_RUN_BACKUP_SHA_VERIFIED', run=h.RUN,
            identity_sha256=h.IDENTITY, total_bytes=receipt['total_bytes'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if owns_state:
            h.state('TECHNICAL_PAUSE_LATE_EXPORT', reason=f'{type(error).__name__}: {error}')
        raise
