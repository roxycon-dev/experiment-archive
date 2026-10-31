"""One bounded handoff: verified cache offload, registered next run, full backup.

Only completed validation caches may be removed; every removed byte first has
an independently SHA-verified local copy. No checkpoint or evidence is deleted.
"""
import concurrent.futures
import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.remote import connect
from qlib.common import sha, utc_now, write_json

REMOTE = '/root/autodl-tmp/HRDecoder_Q_20261003'
PYTHON = '/root/miniconda3/envs/hrdecoder_q_20261003/bin/python'
RUN = 'Q_MIX_seed42'
IDENTITY = '9803ad79aecd9c3d7a426fda0053ae0e43f1afc663933e30d35364be38800a3e'
STATE = ROOT/'reports/qmix16_handoff_state.json'
DELETED = ROOT/'reports/qmix16_cache_offload_receipt.json'
ALLOWED_CACHE_UPDATES = (12000, 16000)
PLAN_PREFIX = 'qmix16_cache_offload_plan'
verified = {}
deleted = {}
launched = None
started = time.monotonic()


def execute(code, timeout=120):
    with connect() as client:
        _, out, err = client.exec_command('cd '+REMOTE+' && '+PYTHON+" - <<'PY'\n"+code+"\nPY", timeout=timeout)
        data, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error[-3000:])
    return json.loads(data)


def state(stage, **kwargs):
    write_json(STATE, dict(stage=stage, updated_at=utc_now(), automatic_retry=False,
        verified_files=len(verified), offloaded_files=len(deleted), next_launch=launched,
        seconds=time.monotonic()-started, **kwargs))


def cache_plan(update):
    return execute(f'''
import json,pathlib
from qlib.common import sha
r=pathlib.Path({REMOTE!r});run=r/'runs/{RUN}'
status=json.loads((run/'status.json').read_text())
assert status['identity_sha256']=={IDENTITY!r}
directory=run/'val_u{update}'
metrics=json.loads((directory/'metrics.json').read_text())
assert metrics['update']=={update} and metrics['n_images']==76
assert len(metrics['probability_files'])==76
files=[]
for row in metrics['probability_files']:
 p=directory/(row['sid']+'.npz')
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(directory)
 assert row['dtype']=='float32' and sha(p)==row['sha256']
 files.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=row['sha256'],
  reason='Completed formal validation FP32 cache; local SHA verified before remote-only offload'))
print(json.dumps(files))
''')


def copy_file(row):
    rel = pathlib.PurePosixPath(row['path'])
    assert rel.parts[:2] == ('runs', RUN) and '..' not in rel.parts
    base = ROOT/'server_results/runs'/RUN
    base.mkdir(parents=True, exist_ok=True)
    parent = ROOT/'server_results'/rel.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Resolve after creating the containing directory. In particular, do not
    # compare concurrently changing, non-existent Windows directory paths.
    allowed = base.resolve(strict=True)
    target = (parent/rel.name).resolve()
    assert allowed.is_relative_to(ROOT.resolve(strict=True))
    assert target.is_relative_to(allowed), f'Backup path escapes allowed root: {target!s}; allowed={allowed!s}'
    if not target.exists() or target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
        temporary = target.with_suffix(target.suffix+'.download')
        with connect() as client:
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(120)
                sftp.get(REMOTE+'/'+row['path'], str(temporary))
        assert temporary.stat().st_size == row['bytes'] and sha(temporary) == row['sha256'], 'Backup SHA mismatch'
        temporary.replace(target)
    return row


def offload(rows, label):
    write_json(ROOT/f'reports/{PLAN_PREFIX}_{label}.json', dict(
        created_at=utc_now(), remote_root=REMOTE, delete_files=rows,
        local_files_deleted=False, protected='All models, checkpoints, last, data, P0, logs, counts and unknown assets'))
    # Establish the validated destination before starting parallel transfers.
    for row in rows:
        rel=pathlib.PurePosixPath(row['path'])
        assert rel.parts[:2]==('runs',RUN) and '..' not in rel.parts
        (ROOT/'server_results'/rel.parent).mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(copy_file, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result()
            except BaseException as error:
                for pending in futures:pending.cancel()
                state('TECHNICAL_PAUSE_HANDOFF',reason=f'{type(error).__name__}: {error}')
                raise
            # Recheck both copies at the deletion boundary. The remote allowlist
            # is limited to the two completed, immutable validation directories.
            target = ROOT/'server_results'/row['path']
            assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
            result = execute(f'''
import pathlib,json,shutil
from qlib.common import sha
r=pathlib.Path({REMOTE!r}).resolve();row=json.loads({json.dumps(row)!r})
rel=pathlib.PurePosixPath(row['path']);p=(r/rel).resolve()
assert rel.parts[:2]==('runs',{RUN!r}) and rel.suffix=='.npz'
assert rel.parent.name in {tuple('val_u'+str(u) for u in ALLOWED_CACHE_UPDATES)!r} and p.is_relative_to(r/'runs/{RUN}')
assert p.is_file() and not p.is_symlink() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
m=json.loads((p.parent/'metrics.json').read_text())
assert m['n_images']==76 and any(x['sid']==p.stem and x['sha256']==row['sha256'] for x in m['probability_files'])
p.unlink()
print(json.dumps(dict(deleted_path=row['path'],free_bytes=shutil.disk_usage(r).free)))
''')
            verified[row['path']] = row
            deleted[row['path']] = row
            write_json(DELETED, dict(status='IN_PROGRESS', deleted_files=list(deleted.values()),
                deleted_bytes=sum(x['bytes'] for x in deleted.values()),
                local_files_deleted=False, each_local_and_remote_SHA_verified=True, updated_at=utc_now()))
            state('BACKING_UP_AND_OFFLOADING_COMPLETED_CACHE', last_file=row['path'], free_bytes=result['free_bytes'])
    print(label, 'cache verified and offloaded', len(rows), flush=True)


def final_inventory():
    return execute(f'''
import pathlib,json
from qlib.common import sha,utc_now
r=pathlib.Path({REMOTE!r});run=r/'runs/{RUN}'
status=json.loads((run/'status.json').read_text())
assert status['stage']=='COMPLETE' and status['completed']==16000 and status['identity_sha256']=={IDENTITY!r}
files=[]
for p in sorted(run.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():
  assert not p.name.endswith(('.tmp','.download'))
  files.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
print(json.dumps(dict(run={RUN!r},status=status,files=files,created_at=utc_now())))
''')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--resume-after-path-fix',action='store_true')
    args=ap.parse_args()
    if STATE.exists():
        prior=json.loads(STATE.read_text(encoding='utf-8'))
        assert args.resume_after_path_fix and prior['stage']=='TECHNICAL_PAUSE_HANDOFF'
        assert prior['verified_files']==0 and prior['offloaded_files']==0 and prior['next_launch'] is None
        evidence=ROOT/'reports/qmix16_handoff_path_guard_failure.json'
        assert not evidence.exists(), 'Only one explicit repaired handoff resume is permitted'
        write_json(evidence,dict(prior=prior,preserved_at=utc_now(),
            repair='Create destination directories serially, then resolve canonical existing root; preserve all containment checks',
            deletion_receipt_absent=not DELETED.exists(),original_worker_training_unmodified=True))
    else:
        assert not args.resume_after_path_fix
    state('BACKING_UP_COMPLETED_12K_CACHE_WHILE_TRAINING_CONTINUES')
    offload(cache_plan(12000), '12000')
    # This is a one-shot dependency wait, not recurring training inspection.
    # It expires with evidence and never retries training on a technical error.
    deadline = time.monotonic()+20*60
    while True:
        snapshot = execute(f'''
import pathlib,json
r=pathlib.Path({REMOTE!r});s=json.loads((r/'runs/{RUN}/status.json').read_text())
pid=(r/'reports/formal.pid').read_text().strip();p=pathlib.Path('/proc')/pid/'cmdline'
alive=p.exists() and b'train_run.py' in p.read_bytes()
print(json.dumps(dict(status=s,formal_alive=alive)))
''')
        status = snapshot['status']
        assert status['identity_sha256'] == IDENTITY
        if status['stage'] in ('TECHNICAL_PAUSE', 'SERIOUS_FAILURE_STOP'):
            raise RuntimeError('Training paused; no next run: '+json.dumps(status))
        if status['stage'] == 'COMPLETE' and not snapshot['formal_alive']:
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Bounded completion wait expired; no next run launched')
        state('WAITING_FOR_COMPLETE_16K_NODE', status=status)
        time.sleep(30)
    # Freeze the remaining run files before any continuation. Previously
    # offloaded 12k files are part of the same complete local backup plan.
    plan = final_inventory()
    plan['files'] = sorted([*verified.values(), *plan['files']], key=lambda x:x['path'])
    assert len({x['path'] for x in plan['files']}) == len(plan['files'])
    write_json(ROOT/f'reports/backup_plan_{RUN}.json', plan)
    offload(cache_plan(16000), '16000')
    write_json(DELETED, dict(status='COMPLETE', deleted_files=list(deleted.values()),
        deleted_bytes=sum(x['bytes'] for x in deleted.values()), local_files_deleted=False,
        each_local_and_remote_SHA_verified=True, completed_at=utc_now()))
    global launched
    launched = execute(f'''
import pathlib,json,subprocess,os,shutil
r=pathlib.Path({REMOTE!r});os.chdir(r)
py={PYTHON!r}
subprocess.run([py,'reports/advance_qmix_campaign.py'],check=True,stdout=subprocess.DEVNULL)
c=json.loads((r/'reports/campaign_state.json').read_text())
assert not c['architecture_stopped']
endpoint=c['common_endpoint']
if endpoint==24000:
 arm,seed,resume='Q_DDR',42,True
 assert c['statuses']['Q_DDR_seed42']['completed']==16000
else:
 assert endpoint==16000 and c['discovery_complete'] and c['review_seed_authorized']
 arm,seed,resume='Q_MIX',2024,False
 assert not (r/'runs/Q_MIX_seed2024').exists()
assert shutil.disk_usage(r).free>=9*1024**3, 'Insufficient reserved capacity for the next two late nodes'
for name in ('formal.pid','qmix_launcher.pid'):
 p=r/'reports'/name
 if p.exists():
  cmd=pathlib.Path('/proc')/p.read_text().strip()/'cmdline'
  if cmd.exists():
   assert b'train_run.py' not in cmd.read_bytes() and b'start_qmix_remote.py' not in cmd.read_bytes()
command=[py,'-u','reports/start_qmix_remote.py','--arm',arm,'--seed',str(seed),'--endpoint',str(endpoint)]
if resume:command.append('--resume')
with (r/'reports/qmix_next_launcher.log').open('wb') as log:
 p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=r,
  env={{**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}},start_new_session=True)
(r/'reports/qmix_launcher.pid').write_text(str(p.pid)+chr(10))
receipt=dict(stage='REGISTERED_NEXT_RUN_QUEUED',launcher_pid=p.pid,arm=arm,seed=seed,endpoint=endpoint,
 resume=resume,free_bytes=shutil.disk_usage(r).free,campaign=c)
(r/'reports/qmix16_next_run_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''')
    write_json(ROOT/'reports/qmix16_next_run_receipt.json', launched)
    state('NEXT_RUN_QUEUED_FULL_EVIDENCE_BACKUP_CONTINUES')
    pending = [x for x in plan['files'] if x['path'] not in verified]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(copy_file, row) for row in pending]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            verified[row['path']] = row
            state('NEXT_RUN_QUEUED_FULL_EVIDENCE_BACKUP_CONTINUES', last_file=row['path'])
    assert set(verified) == {x['path'] for x in plan['files']}
    write_json(ROOT/f'reports/run_backup_{RUN}.json', dict(complete=True, run=RUN,
        files=sorted(verified.values(),key=lambda x:x['path']), verified_at=utc_now(),
        total_bytes=sum(x['bytes'] for x in verified.values()), seconds=time.monotonic()-started,
        remote_deletion_performed=True, deletion_scope='Only SHA-backed late probability NPZ; every model and evidence file retained'))
    state('COMPLETE_BACKUP_VERIFIED_NEXT_RUN_QUEUED')
    print('Complete SHA backup and one registered handoff queued', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state('TECHNICAL_PAUSE_HANDOFF', reason=f'{type(error).__name__}: {error}')
        raise
