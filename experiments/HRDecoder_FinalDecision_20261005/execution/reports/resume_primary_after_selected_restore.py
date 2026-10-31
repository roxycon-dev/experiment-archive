"""Resume the original receiver after diagnosed competing restore ownership."""
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil
from qlib.common import read_json, write_json, sha, utc_now
from ops.remote import connect, command

STATE = ROOT / 'reports/selected_restore_receiver_recovery_state.json'
LOCK = '3146ca1edafdffdae7bf4987524bddc502209929e5d6d5b0e7ba49f12b6bd66e'
EXPECTED = 'c933256b07556689a7fd199dd80e50d4d0c27eb6b9ea782713d95ec51e6d8d62'


def main():
    write_json(STATE, {'stage': 'WAITING_FOR_SINGLE_EXISTING_PRIORITY_UPLOAD', 'pid': os.getpid(),
                      'time': utc_now(), 'original_receiver_stopped_after_sealed_pool': 30004,
                      'diagnosis': 'The ordinary restore started after its complete D32 pool and competed with the already active priority upload; only the ordinary duplicate owner was canceled.',
                      'no_training_or_test_restarted': True})
    started = time.monotonic()
    receiptfile = ROOT / 'reports/priority_selected_restore_receipt.json'
    while not receiptfile.exists():
        s = read_json(ROOT / 'reports/priority_selected_restore_state.json')
        assert s['stage'] == 'TRANSFERRING_ONLY_LOCKED_CHECKPOINT', s
        p = psutil.Process(28616)
        assert any(x.endswith('priority_selected_restore.py') for x in p.cmdline())
        assert time.monotonic() - started < 5400, 'Diagnose priority transport; do not restart blindly'
        time.sleep(10)
    receipt = read_json(receiptfile)
    assert receipt['stage'] == 'ONLY_LOCKED_WEIGHTS_SHA_RESTORED'
    assert receipt['sha256'] == EXPECTED and receipt['lock_sha256'] == LOCK
    runtime = read_json(ROOT / 'configs/runtime.local.json')
    code = """from pathlib import Path
import hashlib,json
r=Path(%r);p=r/'runs/S_seed42/checkpoint_u20000.pth'
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert h.hexdigest()==%r
assert json.loads((r/'reports/final_selection_lock.json').read_text())['lock_sha256']==%r
print(json.dumps({'sha256':h.hexdigest(),'bytes':p.stat().st_size}))
""" % (runtime['server']['remote_root'], EXPECTED, LOCK)
    with connect() as client:
        actual = json.loads(command(client, shlex.quote(runtime['server_python']) + ' -c ' + shlex.quote(code), 120))
    assert actual['bytes'] == 1621379373
    assert not psutil.pid_exists(30004), 'Original duplicate receiver must have exited'
    for p in psutil.process_iter(['pid', 'cmdline']):
        cmd = p.info['cmdline'] or []
        assert not (any(x.endswith('independent_local_attendant_runner.py') for x in cmd) and cmd[-1:] == ['primary']), 'An original primary receiver is already active'
    mirrorlock = ROOT / 'reports/mirror.lock'
    if mirrorlock.exists():
        assert read_json(mirrorlock)['pid'] == 30004
        mirrorlock.unlink()
    runner = ROOT / 'reports/independent_local_attendant_runner.py'
    assert sha(ROOT / 'reports/primary_parallel_attendant.py') == 'f76acf364bdcd56adc290c73ab7d47c9f6641bb817d1b0262b96483d8831578e'
    executable = Path(sys.executable)
    cmdline = subprocess.list2cmdline([str(executable), '-u', str(runner), 'primary'])
    # Use the same previously verified hidden WMI service launcher, with no app kill-on-close job.
    ps = "$startup = New-CimInstance -Namespace root/cimv2 -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0;CreateFlags=[uint32]16777216}; "
    ps += "$result = Invoke-CimMethod -Namespace root/cimv2 -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='" + cmdline.replace("'", "''") + "';CurrentDirectory='" + str(ROOT).replace("'", "''") + "';ProcessStartupInformation=$startup}; $result | ConvertTo-Json -Compress"
    output = subprocess.check_output(['powershell', '-NoProfile', '-Command', ps], text=True)
    launch = json.loads(output)
    assert launch['ReturnValue'] == 0, launch
    time.sleep(3)
    newpid = int(launch['ProcessId'])
    assert psutil.pid_exists(newpid)
    assert read_json(ROOT / 'reports/attendant_pid.json')['pid'] == newpid
    write_json(STATE, {'stage': 'ORIGINAL_PRIMARY_RECEIVER_RECOVERED_AFTER_LOCKED_SHA', 'time': utc_now(),
                      'pid': newpid, 'canceled_duplicate_owner': 30004, 'locked_sha_verified': actual,
                      'original_helper_sha256': sha(ROOT / 'reports/primary_parallel_attendant.py'),
                      'no_duplicate_receiver': True, 'no_scientific_code_or_selection_changed': True,
                      'cause': 'Concrete competing selected-checkpoint upload; sealed endpoint download pool and all custody receipts preserved.'})


if __name__ == '__main__':
    sys.stdout = (ROOT / 'reports/selected_restore_receiver_recovery.log').open('a', encoding='utf-8', buffering=1)
    sys.stderr = (ROOT / 'reports/selected_restore_receiver_recovery.err.log').open('a', encoding='utf-8', buffering=1)
    try:
        main()
    except BaseException as e:
        write_json(STATE, {'stage': 'SELECTED_RESTORE_RECEIVER_RECOVERY_REVIEW_REQUIRED', 'reason': str(e),
                          'time': utc_now(), 'no_automatic_training_or_test_restart': True})
        traceback.print_exc()
        raise
