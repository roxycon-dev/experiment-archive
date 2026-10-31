"""Independent checked backup using a separate temporary file and SSH connection."""
import argparse
import json
import os
from pathlib import Path
import shlex
import time
import paramiko
from shutdown_after_complete import sha, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--local-root', required=True)
    args = ap.parse_args()
    root = Path(args.local_root).resolve()
    policy = json.loads((root/'shutdown_policy.json').read_text(encoding='utf-8'))
    password = os.environ.pop('HRD_FAST_BACKUP_PASSWORD')
    tag = 'U0_highres'
    rel = 'work_dirs/'+tag+'/ckpt_best.pth'
    remote = policy['remote_root']+'/'+rel
    target = root/'server_results'/rel
    temporary = target.with_suffix('.fast-backup.part')
    status_path = root/'reports/fast_backup_status.json'
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(policy['host'], port=policy['port'], username='root', password=password,
                       allow_agent=False, look_for_keys=False, timeout=20)
        password = None
        sftp = client.open_sftp()
        sftp.get_channel().settimeout(30)
        summary = json.loads(sftp.open(policy['remote_root']+'/work_dirs/'+tag+'/completion_summary.json').read())
        assert summary['status'] in ('complete', 'stopped'), 'Cannot back up an active checkpoint'
        item = sftp.stat(remote)
        code = 'import hashlib,sys;h=hashlib.sha256();f=open(sys.argv[1],"rb");[h.update(b) for b in iter(lambda:f.read(4194304),b"")];print(h.hexdigest())'
        _, out, err = client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code)+' '+shlex.quote(remote),timeout=40)
        expected = out.read().decode().strip()
        assert out.channel.recv_exit_status() == 0 and len(expected) == 64, 'Remote hash failed'
        if not (target.exists() and target.stat().st_size == item.st_size and sha(target) == expected):
            target.parent.mkdir(parents=True, exist_ok=True)
            offset = temporary.stat().st_size if temporary.exists() else 0
            assert offset <= item.st_size, 'Invalid partial backup size'
            started = time.time()
            last_report = 0
            with sftp.open(remote,'rb') as stream, temporary.open('ab') as local:
                stream.seek(offset)
                stream.prefetch(item.st_size,max_concurrent_requests=32)
                while offset < item.st_size:
                    block = stream.read(min(1048576,item.st_size-offset))
                    if not block:
                        raise IOError('Unexpected end of checkpoint')
                    local.write(block)
                    offset += len(block)
                    if time.time()-last_report >= 10:
                        write(status_path,{'status':'copying','transferred_bytes':offset,'total_bytes':item.st_size,'started_at_epoch':started,'updated_at_epoch':time.time(),'pid':os.getpid()})
                        last_report = time.time()
            assert temporary.stat().st_size == item.st_size and sha(temporary) == expected, 'Full backup hash mismatch'
            current = sftp.stat(remote)
            assert (current.st_size,current.st_mtime) == (item.st_size,item.st_mtime), 'Checkpoint changed during backup'
            temporary.replace(target)
        journal_path = root/'reports/verified_backups.json'
        journal = json.loads(journal_path.read_text(encoding='utf-8')) if journal_path.exists() else {}
        journal[tag+'/ckpt_best.pth'] = {'sha256':expected,'bytes':item.st_size,'verified_at_epoch':time.time(),'remote':remote,'local':str(target),'backup_method':'independent_prefetch32'}
        # Distinct temporary file from the ordinary mirror; both entries preserve
        # the same immutable completed-stage content and full SHA requirement.
        write(journal_path,journal)
        write(status_path,{'status':'verified','sha256':expected,'bytes':item.st_size,'local':str(target),'updated_at_epoch':time.time(),'pid':os.getpid()})
        sftp.close()
    except Exception as error:
        write(status_path,{'status':'needs_attention','error':repr(error),'updated_at_epoch':time.time(),'pid':os.getpid()})
        raise
    finally:
        password = None
        client.close()


if __name__ == '__main__':
    main()
