"""Recover complete continuation evidence into a separate immutable local mirror."""
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.remote import connect
from qlib.common import read_json, sha, utc_now, write_json

REMOTE = '/root/autodl-tmp/HRDecoder_Q_20261003'
DEST = ROOT / 'server_results/continuation_evidence'
STATE = ROOT / 'reports/continuation_evidence_backup_state.json'
RECEIPT = ROOT / 'reports/continuation_final_evidence_backup_receipt.json'
owns_state = False


def main():
    global owns_state
    assert not STATE.exists(), 'Inspect existing evidence recovery; no duplicate or automatic retry'
    owns_state = True
    write_json(STATE, dict(stage='RECOVERING_COMPLETED_FINAL_EVIDENCE', updated_at=utc_now()))
    with connect() as client:
        code = '''
import json,pathlib
from qlib.common import read_json,sha,utc_now
r=pathlib.Path(%r);state=read_json(r/'reports/continuation_final_evaluation_state.json')
lock=r/'reports/continuation_final_selection.json';summary=read_json(r/'reports/continuation_final_test/summary.json')
assert state['stage']=='COMPLETE' and summary['status']=='COMPLETE'
assert state['lock_sha256']==summary['lock_sha256']==sha(lock)
assert sha(r/'reports/final_selection.json')=='26e5852194aca66bb02062ac79b2f0f8026e33652a95133140a3824bffb38905'
pid=read_json(r/'reports/continuation_final_launch_receipt.json')['pid'];cmd=pathlib.Path('/proc')/str(pid)/'cmdline'
assert not cmd.exists() or b'run_continuation_locked_test.py' not in cmd.read_bytes(), 'Final evidence still active'
paths=[r/'reports'/name for name in ('continuation_final_selection.json','continuation_final_evaluation_state.json',
 'continuation_final_evaluation.log','continuation_final_launch_receipt.json','campaign_state.json')]
paths+=sorted(p for p in (r/'reports/continuation_final_test').rglob('*') if p.is_file())
files=[]
for p in paths:
 assert not p.is_symlink() and p.resolve().is_relative_to(r) and not p.name.endswith(('.tmp','.download'))
 files.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
print(json.dumps(dict(files=files,lock_sha256=sha(lock),test_action=read_json(lock)['test_action'],
 final_process_exited=True,summary=summary,created_at=utc_now())))
''' % REMOTE
        _, out, err = client.exec_command('cd '+REMOTE+" && /root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+'\nPY', timeout=120)
        payload = out.read().decode(); error = err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error[-3000:])
        plan = json.loads(payload)
        write_json(ROOT / 'reports/continuation_final_evidence_backup_plan.json', plan)
        with client.open_sftp() as transfer:
            for row in plan['files']:
                relative = PurePosixPath(row['path'])
                assert relative.parts[0] == 'reports' and '..' not in relative.parts and not relative.is_absolute()
                target = DEST.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                assert target.resolve().is_relative_to(DEST.resolve()) and not target.is_symlink()
                if target.exists():
                    assert sha(target) == row['sha256'] and target.stat().st_size == row['bytes']
                else:
                    temporary = target.with_suffix(target.suffix+'.download')
                    transfer.get(REMOTE+'/'+relative.as_posix(), str(temporary))
                    assert sha(temporary) == row['sha256'] and temporary.stat().st_size == row['bytes']
                    temporary.replace(target)
        write_json(RECEIPT, dict(stage='COMPLETE_SHA_VERIFIED', **plan,
            local_mirror=str(DEST), verified_at=utc_now(), verified_files=len(plan['files']),
            total_bytes=sum(x['bytes'] for x in plan['files']), remote_deletion_performed=False))
        write_json(STATE, dict(stage='COMPLETE_SHA_VERIFIED', verified_files=len(plan['files']),
                              lock_sha256=plan['lock_sha256'], test_action=plan['test_action'], updated_at=utc_now()))
        print(json.dumps(dict(stage='COMPLETE_SHA_VERIFIED', test_action=plan['test_action'],
                             summary=plan['summary'], files=len(plan['files']))))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if owns_state:
            write_json(STATE, dict(stage='TECHNICAL_PAUSE_FINAL_EVIDENCE_RECOVERY', reason='%s: %s' % (type(error).__name__,error),
                                  automatic_retry=False, updated_at=utc_now()))
        raise
