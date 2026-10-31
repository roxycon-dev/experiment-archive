"""One lossless upload of the already locked checkpoint, overlapping bulk recovery."""
import hashlib
import json
import os
import shlex
import sys
import time
import traceback
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qlib.common import digest, read_json, sha, utc_now, write_json
from ops.remote import connect, command


def main():
    guard = ROOT / 'reports/priority_selected_restore.lock'
    with guard.open('x', encoding='utf-8') as f:
        json.dump({'pid': os.getpid(), 'time': utc_now()}, f)
    statefile = ROOT / 'reports/priority_selected_restore_state.json'
    try:
        runtime = read_json(ROOT / 'configs/runtime.local.json')
        assert runtime['server']['instance_id'] == 'connect.nmb2.seetacloud.com:36431'
        remote = runtime['server']['remote_root']
        with connect() as client:
            request_code = """from pathlib import Path
import json
r=Path(%r)
print(json.dumps({'lock':json.loads((r/'reports/final_selection_lock.json').read_text()),'decision':json.loads((r/'reports/decision.json').read_text()),'request':json.loads((r/'reports/selected_restore_request.json').read_text())}))
""" % remote
            evidence = json.loads(command(client, shlex.quote(runtime['server_python']) + ' -c ' + shlex.quote(request_code), 45))
            lock = evidence['lock']
            assert digest({k: v for k, v in lock.items() if k != 'lock_sha256'}) == lock['lock_sha256']
            assert lock['lock_sha256'] == '3146ca1edafdffdae7bf4987524bddc502209929e5d6d5b0e7ba49f12b6bd66e'
            assert (lock['arm'], lock['update'], lock['formula'], lock['seed_scope']) == ('S', 20000, 'fixed_fusion', [42])
            assert evidence['decision']['test_authorized'] and evidence['decision']['selection_lock_sha256'] == lock['lock_sha256']
            assert evidence['request']['lock_sha256'] == lock['lock_sha256']
            assert len(evidence['request']['files']) == 1
            row = evidence['request']['files'][0]
            assert row == lock['checkpoints']['42']
            local = (ROOT / 'server_results' / row['path']).resolve()
            assert local.is_relative_to((ROOT / 'server_results/runs/S_seed42').resolve())
            assert sha(local) == row['sha256']
            expected_bytes = local.stat().st_size
            check_code = """from pathlib import Path
import hashlib,json
r=Path(%r);p=r/%r
print(json.dumps({'existing_sha':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None,'ordinary_upload_active':Path(str(p)+'.restore').exists()}))
""" % (remote, row['path'])
            current = json.loads(command(client, shlex.quote(runtime['server_python']) + ' -c ' + shlex.quote(check_code), 90))
            if current['existing_sha']:
                assert current['existing_sha'] == row['sha256']
                write_json(statefile, {'stage': 'ALREADY_RESTORED_WITH_LOCKED_SHA', 'time': utc_now(), 'no_duplicate_upload': True})
                return
            if current['ordinary_upload_active']:
                write_json(statefile, {'stage': 'DEFERRED_TO_EXISTING_UPLOAD_OWNER', 'time': utc_now(), 'no_duplicate_upload': True})
                return
            # Unique task-created scratch never collides with the ordinary owner's .restore.
            temporary = row['path'] + '.priority_20261005.tmp'
            receive_code = """import sys,zlib,hashlib,json,os
from pathlib import Path
r=Path(%r);p=(r/%r).resolve();target=(r/%r).resolve()
assert p.is_relative_to(r/'runs/S_seed42') and target.is_relative_to(r/'runs/S_seed42')
assert not p.exists(), 'Review a previous priority upload before retrying'
assert not Path(str(target)+'.restore').exists(), 'Existing ordinary uploader owns the file'
lock=json.loads((r/'reports/final_selection_lock.json').read_text())
assert lock['lock_sha256']==%r and lock['checkpoints']['42']==%r
d=zlib.decompressobj();h=hashlib.sha256();total=0
try:
 with p.open('xb') as f:
  while True:
   block=sys.stdin.buffer.read(1024*1024)
   if not block:break
   decoded=d.decompress(block);f.write(decoded);h.update(decoded);total+=len(decoded)
  tail=d.flush();f.write(tail);h.update(tail);total+=len(tail);f.flush();os.fsync(f.fileno())
 assert d.eof and total==%r and h.hexdigest()==%r, 'Lossless stream or original SHA mismatch'
 assert not Path(str(target)+'.restore').exists(), 'Ordinary uploader took ownership; defer'
 if target.exists():
  assert hashlib.sha256(target.read_bytes()).hexdigest()==%r
  p.unlink()
 else:os.replace(p,target)
 print(json.dumps({'stage':'ONLY_LOCKED_WEIGHTS_SHA_RESTORED','path':%r,'bytes':total,'sha256':h.hexdigest(),'lock_sha256':%r,'lossless_compression':True,'no_scientific_state_changed':True}),flush=True)
except BaseException:
 if p.exists():p.unlink()
 raise
""" % (remote, temporary, row['path'], lock['lock_sha256'], row,
          expected_bytes, row['sha256'], row['sha256'], row['path'], lock['lock_sha256'])
            channel = client.get_transport().open_session(timeout=30)
            channel.settimeout(180)
            channel.exec_command(shlex.quote(runtime['server_python']) + ' -c ' + shlex.quote(receive_code))
            compressor = zlib.compressobj(1)
            started = time.monotonic()
            consumed = sent = 0
            last_report = -100.0
            with local.open('rb') as f:
                for block in iter(lambda: f.read(4 * 1024**2), b''):
                    payload = compressor.compress(block)
                    if payload:
                        channel.sendall(payload)
                        sent += len(payload)
                    consumed += len(block)
                    elapsed = time.monotonic() - started
                    if elapsed - last_report >= 30:
                        write_json(statefile, {'stage': 'TRANSFERRING_ONLY_LOCKED_CHECKPOINT', 'pid': os.getpid(), 'lock_sha256': lock['lock_sha256'], 'input_bytes': consumed, 'expected_input_bytes': expected_bytes, 'compressed_bytes_sent': sent, 'seconds': elapsed, 'time': utc_now(), 'ordinary_receiver_not_restarted': True})
                        last_report = elapsed
            tail = compressor.flush()
            channel.sendall(tail)
            sent += len(tail)
            channel.shutdown_write()
            stdout = channel.makefile('rb').read().decode('utf-8')
            stderr = channel.makefile_stderr('rb').read().decode('utf-8', 'replace')
            assert channel.recv_exit_status() == 0, stderr[-1500:]
            result = json.loads(stdout)
            assert result['sha256'] == row['sha256'] and result['bytes'] == expected_bytes
            result.update(time=utc_now(), pid=os.getpid(), seconds=time.monotonic() - started,
                          compressed_bytes_sent=sent, original_primary_attendant_preserved=True,
                          diagnostic_cause='The ordinary endpoint mirror was already receiving 4.9GB before restore_selected; only the frozen selected checkpoint was prioritized.')
            write_json(ROOT / 'reports/priority_selected_restore_receipt.json', result)
            write_json(statefile, result)
            print(json.dumps(result), flush=True)
    except BaseException as e:
        write_json(statefile, {'stage': 'PRIORITY_UPLOAD_REVIEW_REQUIRED', 'reason': str(e), 'time': utc_now(), 'automatic_training_or_test_restart': False})
        traceback.print_exc()
        raise
    finally:
        guard.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.stdout = (ROOT / 'reports/priority_selected_restore.log').open('a', encoding='utf-8', buffering=1)
    sys.stderr = (ROOT / 'reports/priority_selected_restore.err.log').open('a', encoding='utf-8', buffering=1)
    main()
