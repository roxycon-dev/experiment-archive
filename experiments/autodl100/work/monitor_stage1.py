"""Local read/backup monitor; credentials supplied in process memory only."""
import json
from pathlib import Path
import time
import paramiko

LOCAL = Path(r'D:\dataset\experiment\autodl100\results')
REMOTE = '/root/autodl-tmp/hr100'


def run(password):
    LOCAL.mkdir(parents=True, exist_ok=True)
    failures = 0
    last_phase = None
    while time.time() < 1789163580:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect('connect.nmb1.seetacloud.com', port=37047, username='root',
                           password=password, timeout=15, auth_timeout=15)
            client.get_transport().set_keepalive(15)
            sftp = client.open_sftp()
            sftp.get_channel().settimeout(20)
            with sftp.file(REMOTE + '/stage1_state.json') as stream:
                state = json.load(stream)
            for name in ['stage1_state.json', 'initial_val_e4_corrected.log',
                         'trial_a.log', 'trial_b.log', 'wrapper.log', 'hard_deadline.json']:
                try:
                    sftp.get(REMOTE + '/' + name, str(LOCAL / name))
                except FileNotFoundError:
                    pass
            for trial in ['a', 'b']:
                remote_dir = REMOTE + '/trial_' + trial
                local_dir = LOCAL / ('trial_' + trial)
                local_dir.mkdir(exist_ok=True)
                try:
                    for item in sftp.listdir_attr(remote_dir):
                        if item.filename.endswith(('.log.json', '.py')):
                            sftp.get(remote_dir + '/' + item.filename, str(local_dir / item.filename))
                except FileNotFoundError:
                    pass
            phase = state['phase']
            if phase != last_phase:
                print('REMOTE_PHASE', phase, state.get('trial'), flush=True)
                last_phase = phase
            if phase == 'awaiting_backup':
                for trial in state['completed']:
                    dest = LOCAL / ('trial_' + trial['trial'])
                    for remote_path in trial['best_checkpoints']:
                        final = dest / Path(remote_path).name
                        partial = final.with_suffix('.pth.part')
                        sftp.get(remote_path, str(partial), prefetch=False)
                        assert partial.stat().st_size == sftp.stat(remote_path).st_size
                        partial.replace(final)
                with sftp.file(REMOTE + '/backup_complete', 'w') as stream:
                    stream.write('Local best checkpoints and logs received.\n')
                (LOCAL / 'backup_complete.json').write_text(json.dumps({'time': time.time(), 'state': state}))
                print('BACKUP_COMPLETE; remote wrapper will shut down', flush=True)
                return
            if phase in ['failed', 'finished']:
                print('TERMINAL_PHASE', phase, flush=True)
                return
            failures = 0
            (LOCAL / 'monitor_status.json').write_text(json.dumps({'updated_unix': time.time(), 'state': state}))
            sftp.close()
        except Exception as exc:
            failures += 1
            (LOCAL / 'monitor_status.json').write_text(json.dumps({'updated_unix': time.time(),
                'connection_failures': failures, 'error_type': type(exc).__name__}))
            print('MONITOR_RECONNECT', failures, type(exc).__name__, flush=True)
            if failures >= 5:
                return
        finally:
            client.close()
        time.sleep(60)

