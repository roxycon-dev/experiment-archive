"""One finite DDR42->MIX42 handoff at the registered common 24k endpoint."""
import concurrent.futures
import json
import pathlib
import shutil
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'reports'))
import handoff_after_qmix16 as h
from qlib.common import read_json,write_json,sha,utc_now

h.RUN='Q_DDR_seed42'
h.IDENTITY='b3e4cea88c5eeda8923a9c3950c0f67e74f02c28057730f2dd2d19154c500396'
h.STATE=ROOT/'reports/ddr42_24k_handoff_state.json'
h.DELETED=ROOT/'reports/ddr42_24k_cache_offload_receipt.json'
h.ALLOWED_CACHE_UPDATES=(20000,24000)
h.PLAN_PREFIX='ddr42_24k_cache_offload_plan'
PREFIX='ddr42_24k'
TARGET_ARM='Q_MIX'
TARGET_SEED=42
TARGET_RESUME=True
MIN_FREE_BYTES=9*1024**3
TARGET_LOG='mix42_24k_launcher.log'


def wait_node(update, final=False):
    deadline=time.monotonic()+75*60
    while True:
        x=h.execute(f'''
import pathlib,json
r=pathlib.Path({h.REMOTE!r});run=r/'runs/{h.RUN}'
s=json.loads((run/'status.json').read_text());assert s['identity_sha256']=={h.IDENTITY!r}
pid=(r/'reports/formal.pid').read_text().strip();cmd=pathlib.Path('/proc')/pid/'cmdline'
alive=cmd.exists() and b'train_run.py' in cmd.read_bytes()
m=run/'val_u{update}/metrics.json'
ready=m.exists() and json.loads(m.read_text())['n_images']==76
print(json.dumps(dict(status=s,node_ready=ready,formal_alive=alive)))
''')
        if x['status']['stage'] in ('TECHNICAL_PAUSE','SERIOUS_FAILURE_STOP'):
            raise RuntimeError('Formal run paused; no handoff: '+json.dumps(x))
        if x['node_ready'] and (not final or (x['status']['stage']=='COMPLETE' and not x['formal_alive'])):
            return x
        if time.monotonic()>deadline:raise TimeoutError('Finite node dependency wait expired; no launch')
        h.state('WAITING_FOR_REGISTERED_NODE',update=update,formal_status=x['status'])
        time.sleep(30)


def preserve_prior_mutable(plan, prior):
    # Preserve the exact older files before replacing local aliases with 24k
    # state. Existing 16k checkpoints and its final-test lock never change.
    current={x['path']:x for x in plan['files']}
    mapping=[]
    for row in prior['files']:
        if row['path'] in current and current[row['path']]['sha256']!=row['sha256']:
            source=ROOT/'server_results'/row['path']
            assert source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
            base=ROOT/'server_results/stage_backups'/f'{h.RUN}_16000'
            base.mkdir(parents=True,exist_ok=True)
            allowed=base.resolve(strict=True)
            assert allowed.is_relative_to(ROOT.resolve(strict=True))
            target=base/row['path']
            target.parent.mkdir(parents=True,exist_ok=True)
            target=target.resolve()
            assert target.is_relative_to(allowed)
            shutil.copy2(source,target)
            assert sha(target)==row['sha256']
            mapping.append(dict(original_path=row['path'],preserved_local_path=str(target),sha256=row['sha256']))
    write_json(ROOT/f'reports/{h.RUN}_16k_mutable_preservation.json',dict(complete=True,files=mapping,verified_at=utc_now()))


def main():
    assert not h.STATE.exists(), 'An existing handoff must be inspected, not dispatched again'
    prior=read_json(ROOT/f'reports/run_backup_{h.RUN}.json')
    assert prior['complete'] and prior['run']==h.RUN
    frozen=ROOT/f'reports/run_backup_{h.RUN}_16k_frozen.json'
    if not frozen.exists():write_json(frozen,prior)
    wait_node(20000)
    h.offload(h.cache_plan(20000),'20000')
    wait_node(24000,final=True)
    inventory=h.execute(f'''
import pathlib,json
from qlib.common import sha,utc_now
r=pathlib.Path({h.REMOTE!r});run=r/'runs/{h.RUN}';s=json.loads((run/'status.json').read_text())
assert s['stage']=='COMPLETE' and s['completed']==24000 and s['identity_sha256']=={h.IDENTITY!r}
files=[]
for p in sorted(run.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():
  assert not p.name.endswith(('.tmp','.download'))
  files.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
print(json.dumps(dict(run={h.RUN!r},status=s,files=files,created_at=utc_now())))
''')
    merged={x['path']:x for x in prior['files']}
    merged.update(h.verified)
    merged.update({x['path']:x for x in inventory['files']})
    plan={**inventory,'files':sorted(merged.values(),key=lambda x:x['path']),
          'earlier_offloaded_files_included_from_verified_16k_backup':True}
    write_json(ROOT/f'reports/backup_plan_{h.RUN}_24k.json',plan)
    preserve_prior_mutable(plan,prior)
    h.offload(h.cache_plan(24000),'24000')
    write_json(h.DELETED,dict(status='COMPLETE',deleted_files=list(h.deleted.values()),
        deleted_bytes=sum(x['bytes'] for x in h.deleted.values()),each_local_and_remote_SHA_verified=True,
        local_files_deleted=False,completed_at=utc_now()))
    # The accepted upload archive is redundant after extraction. Its exact
    # local and remote SHA are checked before freeing this one transport file.
    transport=read_json(ROOT/'reports/qmix_transport_manifest.json')
    archive=ROOT/'assets/Q_MIX_accepted_assets.tar.gz'
    assert archive.stat().st_size==transport['archive_bytes'] and sha(archive)==transport['archive_sha256']
    launch=h.execute(f'''
import pathlib,json,subprocess,os,shutil
from qlib.common import sha
r=pathlib.Path({h.REMOTE!r}).resolve();os.chdir(r);py={h.PYTHON!r}
archive=r/'assets/Q_MIX_accepted_assets.tar.gz';transport=json.loads((r/'reports/qmix_transport_manifest.json').read_text())
asset=json.loads((r/'reports/qmix_remote_asset_receipt.json').read_text())
assert asset['status']=='ALL_REMOTE_ASSET_SHA_VERIFIED'
if archive.exists():
 assert not archive.is_symlink() and archive.resolve().is_relative_to(r/'assets')
 assert sha(archive)=={transport['archive_sha256']!r} and archive.stat().st_size=={transport['archive_bytes']}
 archive.unlink()
 (r/'reports/qmix_transport_offload_receipt.json').write_text(json.dumps(dict(path='assets/Q_MIX_accepted_assets.tar.gz',
  sha256={transport['archive_sha256']!r},bytes={transport['archive_bytes']},local_SHA_verified=True,remote_SHA_verified=True,
  reason='Redundant accepted transport archive, local original retained; extracted training data retained')))
subprocess.run([py,'reports/advance_qmix_campaign.py'],check=True,stdout=subprocess.DEVNULL)
c=json.loads((r/'reports/campaign_state.json').read_text())
assert c['common_endpoint']==24000 and not c['architecture_stopped']
assert c['statuses'][{h.RUN!r}]['stage']=='COMPLETE' and c['statuses'][{h.RUN!r}]['completed']==24000
target={f'{TARGET_ARM}_seed{TARGET_SEED}'!r}
if {TARGET_RESUME!r}:
 assert c['statuses'][target]['stage']=='COMPLETE' and c['statuses'][target]['completed']==16000
else:
 assert target not in c['statuses'] and not (r/'runs'/target).exists(), 'Fresh registered run already exists'
if {TARGET_SEED}==2024:assert c['discovery_complete'] and c['review_seed_authorized']
assert shutil.disk_usage(r).free>={MIN_FREE_BYTES}, 'Insufficient reserved late-node capacity'
for name in ('formal.pid','qmix_launcher.pid'):
 p=r/'reports'/name
 if p.exists():
  cmd=pathlib.Path('/proc')/p.read_text().strip()/'cmdline'
  if cmd.exists():assert b'train_run.py' not in cmd.read_bytes() and b'start_qmix_remote.py' not in cmd.read_bytes()
with (r/'reports'/{TARGET_LOG!r}).open('wb') as log:
 command=[py,'-u','reports/start_qmix_remote.py','--arm',{TARGET_ARM!r},'--seed',{str(TARGET_SEED)!r},'--endpoint','24000']
 if {TARGET_RESUME!r}:command.append('--resume')
 p=subprocess.Popen(command,
  cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
  env={{**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}})
(r/'reports/qmix_launcher.pid').write_text(str(p.pid)+chr(10))
receipt=dict(stage='REGISTERED_24K_RESUME_QUEUED',launcher_pid=p.pid,arm={TARGET_ARM!r},seed={TARGET_SEED},endpoint=24000,
 resume={TARGET_RESUME!r},free_bytes=shutil.disk_usage(r).free,campaign=c)
(r/'reports'/{(PREFIX+'_next_run_receipt.json')!r}).write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''')
    h.launched=launch
    write_json(ROOT/f'reports/{PREFIX}_next_run_receipt.json',launch)
    h.state('NEXT_RUN_QUEUED_FULL_BACKUP_CONTINUES')
    pending=[x for x in plan['files'] if x['path'] not in h.verified]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(h.copy_file,row) for row in pending]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();h.verified[row['path']]=row;h.state('NEXT_RUN_QUEUED_FULL_BACKUP_CONTINUES')
    assert set(h.verified)==set(merged)
    receipt=dict(complete=True,run=h.RUN,endpoint=24000,files=sorted(h.verified.values(),key=lambda x:x['path']),
        verified_at=utc_now(),total_bytes=sum(x['bytes'] for x in h.verified.values()),
        original_16k_mutable_evidence_preserved=f'reports/{h.RUN}_16k_mutable_preservation.json',
        remote_deletion_scope='Only locally SHA-backed 20k/24k probability NPZ and redundant accepted transport archive')
    write_json(ROOT/f'reports/run_backup_{h.RUN}_24k.json',receipt)
    write_json(ROOT/f'reports/run_backup_{h.RUN}.json',receipt)
    h.state('COMPLETE_BACKUP_VERIFIED_NEXT_RUN_QUEUED')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        h.state('TECHNICAL_PAUSE_HANDOFF',reason=f'{type(error).__name__}: {error}')
        raise
