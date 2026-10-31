"""Register the user-supplied replacement SSH endpoint; never persist a password."""
import getpass
import hashlib
import json
import paramiko
from remote import HOST, PORT, KEY, KNOWN

client = paramiko.SSHClient()
client.load_host_keys(str(KNOWN))
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
arguments = dict(hostname=HOST, port=PORT, username="root", timeout=20,
    auth_timeout=20, look_for_keys=False, allow_agent=False)
try:
    client.connect(key_filename=str(KEY), **arguments)
except paramiko.AuthenticationException:
    client.close()
    client = paramiko.SSHClient()
    client.load_host_keys(str(KNOWN))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    password = getpass.getpass("SSH password (not saved): ")
    client.connect(password=password, **arguments)
    password = None
    with client.open_sftp() as ftp:
        public_key = KEY.with_suffix(".pub").read_text().strip()
        try:
            with ftp.open("/root/.ssh/authorized_keys", "r") as stream:
                current = stream.read().decode()
        except FileNotFoundError:
            current = ""
        if public_key not in current:
            with ftp.open("/root/.ssh/authorized_keys", "a") as stream:
                stream.write("\n"+public_key+"\n")
            ftp.chmod("/root/.ssh/authorized_keys", 0o600)
client.save_host_keys(str(KNOWN))
print(json.dumps({"SSH_connected": True, "host": HOST, "port": PORT,
    "host_key_SHA256_hex": hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).hexdigest()}))
command = """nvidia-smi --query-gpu=name,memory.total,power.limit,utilization.gpu,memory.used --format=csv
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/cpu.max
df -h /root/autodl-tmp /dev/shm
ls /root/autodl-tmp/HRDecoder_v3_20261008/training_runs/v3 2>/dev/null || true
ps -eo pid,args | grep -E 'train.py|cache_audit|launch_gpu' | grep -v grep || true
"""
_, stdout, stderr = client.exec_command(command)
print(stdout.read().decode(), end="")
print(stderr.read().decode(), end="")
client.close()
