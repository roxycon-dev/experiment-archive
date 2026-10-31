# -*- coding: utf-8 -*-
"""Check server state after the interrupted smoke; kill leftovers."""
import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
            timeout=30, banner_timeout=30, auth_timeout=30)

for tag, cmd in [
    ('procs', 'ps aux | grep -E "smoke_gpu|python" | grep -v grep || echo NONE'),
    ('gpu', 'nvidia-smi --query-gpu=memory.used --format=csv,noheader'),
]:
    _, o, e = cli.exec_command(cmd, timeout=60)
    print('## %s\n%s' % (tag, o.read().decode('utf-8', 'replace').strip()))

# kill any leftover smoke process (the channel death usually kills it, but
# be thorough -- a zombie holding GPU memory would poison the next run)
_, o, _ = cli.exec_command(
    "pkill -f smoke_gpu.py && echo KILLED || echo NOTHING_TO_KILL", timeout=30)
print('## pkill\n%s' % o.read().decode().strip())
import time
time.sleep(3)
_, o, _ = cli.exec_command('nvidia-smi --query-gpu=memory.used --format=csv,noheader',
                           timeout=30)
print('## gpu_after\n%s' % o.read().decode().strip())
cli.close()
