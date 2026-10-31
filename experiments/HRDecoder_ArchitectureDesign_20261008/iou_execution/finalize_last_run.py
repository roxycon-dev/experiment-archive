"""One-shot archive/report/shutdown, guarded by this exact run's terminal state.

Does not wait for training, launch/restart training, or change scientific gates.
Use --dry-run to inspect current readiness without a server mutation.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT = ROOT / 'final_report'
REMOTE = '/root/autodl-tmp/HRDecoder_v3_20261008'
sys.path.insert(0, str(ROOT / 'local_preparation'))
from remote import connect, HOST, PORT


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(p)


def run_local(path, *args):
    subprocess.run([sys.executable, '-X', 'utf8', str(path), *args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    a = parser.parse_args()
    policy = read(HERE / 'LAST_RUN_POLICY.json')
    assert policy['last_architecture'] and not policy['new_architecture_or_fresh_run_allowed']
    assert policy['shutdown_authorized_after_terminal_archive']
    observation = REPORT / 'shutdown_observation.json'
    if observation.exists() and read(observation).get('status') == 'SHUTDOWN_REQUESTED_SSH_UNREACHABLE':
        run_local(REPORT / 'build_report.py')
        print(json.dumps({'status': 'ALREADY_FINALIZED_NO_REPEAT_SHUTDOWN'})); return
    run_local(HERE / 'monitor_iou.py', '--force')
    snapshot = read(HERE / 'latest_snapshot.json')
    rt = snapshot['runtime']
    running = rt.get('GPU_CAMPAIGN_RUNNING.json', {})
    assert running.get('identity_SHA') == policy['identity_SHA'], 'Wrong run identity; refuse shutdown'
    outcome = rt.get('GPU_CAMPAIGN_OUTCOME.json')
    error = rt.get('CONTROLLER_ERROR.json') or rt.get('LAUNCH_ERROR.json')
    checks = rt['own_process_checks']
    # A controller may have written outcome just before exiting; do not kill it.
    if not (outcome or error) or any(x['alive'] for x in checks.values()) or rt.get('own_session_live_processes'):
        print(json.dumps({'status': 'NOT_READY_NO_SHUTDOWN', 'training_alive': checks.get('pid', {}).get('alive'),
                          'controller_alive': checks.get('controller_pid', {}).get('alive')})); return
    terminal_status = outcome['status'] if outcome else 'FAILED'
    assert terminal_status in ('COMPLETE', 'SCREEN_REJECTED', 'CRITICAL_CLASS_COLLAPSE', 'FAILED')
    if a.dry_run:
        print(json.dumps({'status': 'READY_DRY_RUN_NO_MUTATION', 'terminal_status': terminal_status})); return
    client = connect()
    try:
        # Only the RAM owner with both exact PID and command identity may be ended.
        owner = rt['own_RAM']
        code = """import json,os,time
from pathlib import Path
r=Path(%r);pid=%d
p=Path('/proc/%%d/cmdline'%%pid)
cmd=p.read_bytes() if p.exists() else b''
if cmd:
 assert str(r/'sacd_execution/ram_cache_owner.py').encode() in cmd,'RAM owner identity mismatch'
 os.kill(pid,15)
print(json.dumps({'own_RAM_pid':pid,'termination_requested':bool(cmd),'time_unix':time.time(),'external_processes_controlled':False}))
""" % (REMOTE, owner['pid'])
        _, out, err = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(code))
        raw, error_text = out.read().decode(), err.read().decode()
        assert out.channel.recv_exit_status() == 0, error_text
        write(REPORT / 'own_ram_cleanup.json', json.loads(raw))
    finally:
        client.close()
    time.sleep(2)
    # Required final archive after cleanup; no checkpoint is downloaded.
    run_local(HERE / 'monitor_iou.py', '--force')
    snapshot = read(HERE / 'latest_snapshot.json'); rt = snapshot['runtime']
    assert rt['server_online'] and not rt['own_RAM']['alive']
    assert not any(x['alive'] for x in rt['own_process_checks'].values())
    assert not rt['own_session_live_processes']
    assert rt['GPU_CAMPAIGN_RUNNING.json']['identity_SHA'] == policy['identity_SHA']
    run_local(HERE / 'audit_final_outcome.py')
    audit = read(HERE / 'FINAL_OUTCOME_AUDIT.json')
    manifest = []
    receipt_root = ROOT / 'cache_execution/server_receipts/iou'
    for p in sorted(receipt_root.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.log'):
            raw = p.read_bytes()
            manifest.append({'path': str(p), 'bytes': len(raw), 'SHA256': hashlib.sha256(raw).hexdigest()})
    closed = {'status': 'TERMINAL_ARCHIVED_BEFORE_AUTHORIZED_SHUTDOWN', 'terminal_status': terminal_status,
              'identity_SHA': policy['identity_SHA'], 'archived_at': snapshot['checked_at'],
              'own_train_controller_workers_RAM_ended': True, 'server_online_before_shutdown': True,
              'scientific_goal_audit_status': audit['status'], 'goal_achieved': audit['goal_achieved'],
              'checkpoint_downloaded': False, 'cache_data_hash_rescanned': False,
              'further_training_authorized': False, 'receipts': manifest}
    write(REPORT / 'closeout_receipt.json', closed)
    run_local(REPORT / 'build_report.py')
    assert (REPORT / 'HRDecoder完整实验报告.md').exists()
    before = datetime.now(timezone(timedelta(hours=8))).isoformat()
    requested = {'status': 'AUTHORIZED_SHUTDOWN_REQUESTED', 'requested_at': before,
                 'identity_SHA': policy['identity_SHA'], 'terminal_status': terminal_status,
                 'local_receipts_archived': True, 'local_report_saved': True,
                 'command': '/bin/bash /usr/bin/shutdown', 'shutdown_command_issued': False,
                 'platform_poweroff_verified': False, 'external_diagnostic_processes_individually_controlled': False}
    # Fail closed if a different run has appeared since the archive.
    client = connect()
    try:
        code = """import json,os
from pathlib import Path
r=Path(%r);run=json.loads((r/'reports/iou/GPU_CAMPAIGN_RUNNING.json').read_text())
assert run['identity_SHA']==%r
for k in ('pid','controller_pid'):
 p=Path('/proc/%%d/cmdline'%%run[k]);assert not(p.exists() and p.read_bytes()),'Own task still alive'
os.sync()
print('FINAL_RUN_ENDED_SYNCED')
""" % (REMOTE, policy['identity_SHA'])
        _, out, err = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(code))
        raw, error_text = out.read().decode(), err.read().decode()
        assert out.channel.recv_exit_status() == 0 and 'FINAL_RUN_ENDED_SYNCED' in raw, error_text
        requested['shutdown_command_issued'] = True
        write(observation, requested)
        # Existing provider shutdown helper. An SSH drop is an observation,
        # not proof of billing or a cloud console power state.
        try:
            _, out, err = client.exec_command('/bin/bash /usr/bin/shutdown', timeout=20)
            channel = out.channel
            deadline = time.monotonic() + 25
            while not channel.exit_status_ready() and client.get_transport() and client.get_transport().is_active() and time.monotonic() < deadline:
                time.sleep(.5)
            requested['SSH_transport_active_after_command'] = bool(client.get_transport() and client.get_transport().is_active())
            if channel.exit_status_ready(): requested['command_exit_code'] = channel.recv_exit_status()
            if channel.recv_stderr_ready(): requested['command_stderr'] = channel.recv_stderr(8192).decode(errors='replace')
        except Exception as error:
            requested['connection_event_after_command'] = type(error).__name__ + ': ' + str(error)
    finally:
        client.close()
    try:
        probe = connect()
    except Exception as error:
        requested.update(status='SHUTDOWN_REQUESTED_SSH_UNREACHABLE', SSH_reconnect_succeeded=False,
                         connection_error=type(error).__name__ + ': ' + str(error))
    else:
        requested.update(status='SHUTDOWN_REQUESTED_SERVER_STILL_REACHABLE', SSH_reconnect_succeeded=True)
        probe.close()
    requested['observed_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    write(observation, requested)
    for rel in ('iterative_execution/iteration_state.json', 'cache_execution/handover_state.json'):
        d = read(ROOT / rel)
        d.update(current_task_training_active=False, further_architecture_revisions_user_authorized=False,
                 status='FINAL_RUN_CLOSED', final_report=str(REPORT/'HRDecoder完整实验报告.md'),
                 final_goal_audit=str(HERE/'FINAL_OUTCOME_AUDIT.json'), goal_achieved=audit['goal_achieved'],
                 final_terminal_receipt=str(REPORT/'closeout_receipt.json'),
                 current_shutdown_observation=requested, heartbeat_check_interval_minutes=45,
                 updated_at=requested['observed_at'])
        write(ROOT / rel, d)
    run_local(REPORT / 'build_report.py')
    print(json.dumps({'status': requested['status'], 'report': str(REPORT/'HRDecoder完整实验报告.md'),
                      'goal_achieved': audit['goal_achieved'], 'shutdown_command_issued': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
