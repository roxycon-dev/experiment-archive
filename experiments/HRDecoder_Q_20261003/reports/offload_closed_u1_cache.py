"""Preserve one unrelated closed U1 cache locally, then free its remote NPZ only."""
import concurrent.futures
import json
import pathlib
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports'))
from ops.remote import connect
from qlib.common import read_json,write_json,sha,utc_now
from handoff_after_qmix16 import execute

REMOTE='/root/autodl-tmp/HRDecoder_Next_20261003/cold_storage/G/U1_native_whole'
LOCAL=pathlib.Path('D:/dataset/experiment/HRDecoder_Next_20261003/server_results/cold_storage/G/U1_native_whole')
STATE=ROOT/'reports/closed_u1_cache_offload_state.json'
verified=[]


def copy(row):
    assert pathlib.PurePosixPath(row['name']).name==row['name'] and row['name'].endswith('.npz')
    target=(LOCAL/row['name']).resolve()
    assert target.is_relative_to(LOCAL.resolve(strict=True))
    if not target.exists() or target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:
        temporary=target.with_suffix('.npz.download')
        with connect() as client:
            with client.open_sftp() as sftp:
                sftp.get_channel().settimeout(120)
                sftp.get(REMOTE+'/'+row['name'],str(temporary))
        assert temporary.stat().st_size==row['bytes'] and sha(temporary)==row['sha256']
        temporary.replace(target)
    return row


def main():
    assert not STATE.exists(), 'Existing offload must be inspected; never blindly repeat'
    LOCAL.mkdir(parents=True,exist_ok=True)
    assert LOCAL.resolve(strict=True).is_relative_to(pathlib.Path('D:/dataset/experiment/HRDecoder_Next_20261003').resolve(strict=True))
    plan=execute(f'''
import pathlib,json
from qlib.common import sha
q=pathlib.Path('/root/autodl-tmp/HRDecoder_Q_20261003');r=pathlib.Path({REMOTE!r}).resolve()
assert r==pathlib.Path({REMOTE!r}) and not r.is_symlink()
for runtime in ('runtime.server.json','runtime.qmix.server.json'):
 x=json.loads((q/'configs'/runtime).read_text())
 assert pathlib.Path(x['p0_c_cache']).resolve()!=r and pathlib.Path(x['p0_u_cache']).resolve()!=r
files=[]
for p in sorted(r.glob('*.npz')):
 assert not p.is_symlink() and p.resolve().parent==r
 files.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)))
assert len(files)==76
print(json.dumps(dict(files=files,remote_root=str(r),reason='Closed previous U1 validation cache; no Q training, P0, accepted external data or final inference dependency; preserve models and original counts/metadata')))
''')
    write_json(ROOT/'reports/closed_u1_cache_offload_plan.json',plan)
    write_json(STATE,dict(stage='COPYING_EXACT_CLOSED_CACHE',verified_files=0,expected_files=76,
        local_deletion_performed=False,updated_at=utc_now(),automatic_retry=False))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(copy,row) for row in plan['files']]
        for future in concurrent.futures.as_completed(futures):
            try:row=future.result()
            except BaseException:
                for pending in futures:pending.cancel()
                raise
            verified.append(row)
            write_json(STATE,dict(stage='COPYING_EXACT_CLOSED_CACHE',verified_files=len(verified),expected_files=76,
                verified_bytes=sum(x['bytes'] for x in verified),local_deletion_performed=False,
                updated_at=utc_now(),automatic_retry=False))
    for row in verified:
        p=LOCAL/row['name'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    receipt=execute(f'''
import pathlib,json,shutil
from qlib.common import sha,utc_now
r=pathlib.Path({REMOTE!r}).resolve();plan=json.loads({json.dumps(plan)!r})
assert r==pathlib.Path({REMOTE!r})
checked=[]
for row in plan['files']:
 p=r/row['name'];assert p.resolve().parent==r and p.suffix=='.npz' and not p.is_symlink()
 assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'];checked.append(p)
before=shutil.disk_usage(r).free
for p in checked:p.unlink()
receipt=dict(stage='COMPLETE_BACKUP_SHA_VERIFIED_REMOTE_NPZ_ONLY_OFFLOADED',files=plan['files'],
 local_root={str(LOCAL)!r},remote_root=str(r),deleted_bytes=sum(x['bytes'] for x in plan['files']),
 free_bytes_before=before,free_bytes_after=shutil.disk_usage(r).free,
 local_deletion_performed=False,models_and_original_metadata_retained=True,completed_at=utc_now())
q=pathlib.Path('/root/autodl-tmp/HRDecoder_Q_20261003')
(q/'reports/closed_u1_cache_offload_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
''')
    write_json(ROOT/'reports/closed_u1_cache_offload_receipt.json',receipt)
    write_json(STATE,dict(stage=receipt['stage'],verified_files=76,deleted_bytes=receipt['deleted_bytes'],
        free_bytes_after=receipt['free_bytes_after'],local_deletion_performed=False,automatic_retry=False,updated_at=utc_now()))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write_json(STATE,dict(stage='TECHNICAL_PAUSE_STORAGE_OFFLOAD',reason=f'{type(error).__name__}: {error}',
            verified_files=len(verified),automatic_retry=False,updated_at=utc_now()))
        raise
