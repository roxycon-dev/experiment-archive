"""Recover frozen streams and reproducibility assets; no training or deletion."""
import json
from pathlib import Path, PurePosixPath
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops.remote import connect
from qlib.common import digest, read_json, sha, utc_now, write_json

REMOTE = '/root/autodl-tmp/HRDecoder_Q_20261003'
FILES = [
    'configs/protocol.json', 'configs/runtime.server.json', 'configs/runtime.qmix.server.json',
    'reports/source_identity.json', 'reports/data_identity.json', 'reports/pretraining_identity.json',
    'reports/qmix_continuation_protocol.json', 'reports/external_source_user_override.json',
    'reports/external_acceptance/manifest.server.accepted.json',
    'reports/final_evaluation.py', 'reports/continuation_final_evaluation.py',
    'reports/run_continuation_locked_test.py', 'reports/advance_qmix_campaign.py',
]
for arm in ('Q_DDR', 'Q_MIX'):
    for seed in (42, 2024):
        FILES.extend('reports/stream_%s_seed%s.%s' % (arm, seed, suffix)
                     for suffix in ('jsonl', 'summary.json'))
        FILES.append('runs/%s_seed%s/identity.json' % (arm, seed))
STATE = ROOT / 'reports/immutable_campaign_asset_backup_state.json'
RECEIPT = ROOT / 'reports/immutable_campaign_asset_backup_receipt.json'
owns_state = False


def main():
    global owns_state
    assert not STATE.exists(), 'Review existing backup; no duplicate or automatic retry'
    owns_state = True
    write_json(STATE, dict(stage='RECOVERING_FROZEN_ASSETS', updated_at=utc_now(), automatic_retry=False))
    with connect() as client:
        code = '''
import json,pathlib
from qlib.common import sha
r=pathlib.Path(%r); paths=json.loads(%r); files=[]
for relative in paths:
 p=r/relative
 assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(r)
 files.append(dict(path=relative,bytes=p.stat().st_size,sha256=sha(p)))
print(json.dumps(files))
''' % (REMOTE, json.dumps(FILES))
        command = 'cd '+REMOTE+" && /root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+'\nPY'
        _, out, err = client.exec_command(command, timeout=120)
        payload = out.read().decode(); error = err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error[-3000:])
        plan = json.loads(payload)
        write_json(ROOT / 'reports/immutable_campaign_asset_backup_plan.json', plan)
        mirror = ROOT / 'server_results'
        with client.open_sftp() as transfer:
            for row in plan:
                relative = PurePosixPath(row['path'])
                assert relative.as_posix() in FILES and not relative.is_absolute() and '..' not in relative.parts
                target = mirror.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                assert target.resolve().is_relative_to(mirror.resolve()) and not target.is_symlink()
                if target.exists():
                    assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256'], 'Existing immutable asset differs: '+str(relative)
                else:
                    temporary = target.with_suffix(target.suffix+'.immutable.download')
                    transfer.get(REMOTE+'/'+relative.as_posix(), str(temporary))
                    assert temporary.stat().st_size == row['bytes'] and sha(temporary) == row['sha256']
                    temporary.replace(target)
        source = read_json(mirror / 'reports/source_identity.json')
        local_source = []
        for relative, expected in source['files'].items():
            assert sha(ROOT / relative) == expected, 'Local frozen source differs: '+relative
            local_source.append(dict(path=relative,sha256=expected,bytes=(ROOT/relative).stat().st_size))
        for arm in ('Q_DDR', 'Q_MIX'):
            for seed in (42, 2024):
                stream = mirror / ('reports/stream_%s_seed%s.jsonl' % (arm, seed))
                info = read_json(stream.with_suffix('.summary.json'))
                assert sha(stream) == info['file_sha256'] and info['updates'] == 24000
                identity = read_json(mirror / ('runs/%s_seed%s/identity.json' % (arm, seed)))
                assert identity['stream_sha256'] == info['file_sha256']
        receipt = dict(stage='COMPLETE_SHA_VERIFIED', files=plan,
                       frozen_source_present_locally=local_source, source_sha256=source['sha256'],
                       verified_at=utc_now(), total_recovered_bytes=sum(x['bytes'] for x in plan),
                       remote_deletion_performed=False, private_credentials_copied=False)
        write_json(RECEIPT, receipt)
        write_json(STATE, dict(stage=receipt['stage'], verified_at=receipt['verified_at'],
                              verified_assets=len(plan), verified_source_files=len(local_source), automatic_retry=False))
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('files','frozen_source_present_locally')}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if owns_state:
            write_json(STATE, dict(stage='TECHNICAL_PAUSE_ASSET_BACKUP', reason='%s: %s' % (type(error).__name__,error),
                                  automatic_retry=False, updated_at=utc_now()))
        raise
