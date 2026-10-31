"""Download stopped-run text artifacts; never transfer checkpoint/cache binaries."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shlex
import shutil
import stat
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'local_preparation'))
from remote import connect

REMOTE = '/root/autodl-tmp/HRDecoder_v3_20261008'
RECEIPTS = ROOT / 'cache_execution/server_receipts'
ALLOWED = {'.json', '.log', '.csv', '.md', '.txt', '.yaml', '.yml', '.py'}

VERIFY = r'''
import datetime,json,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
remaining=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid(): continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  state=(p/'stat').read_text().split(') ',1)[1].split()[0]
  cwd=os.readlink(str(p/'cwd'))
  matched=(str(root) in cmd and any(x in cmd for x in ['gpu_campaign_controller.py','ram_cache_mixed.py'])) or ('train_performance.py' in cmd and cwd==str(root/'training_execution'))
  if matched and state!='Z': remaining.append({'pid':int(p.name),'cmdline':cmd,'state':state})
 except (OSError,ValueError): pass
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
status=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
memory={}
for name in ['/sys/fs/cgroup/memory.current','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory/memory.usage_in_bytes','/sys/fs/cgroup/memory/memory.limit_in_bytes']:
 try: memory[name]=Path(name).read_text().strip()
 except OSError: pass
v={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'server_online':True,'remaining_campaign_processes':remaining,'gpu_compute_processes_csv':gpu.stdout,'gpu_status_csv':status.stdout,'gpu_query_exit_code':gpu.returncode,'cgroup_memory':memory,'shutdown_hold_exists':(root/'reports/shutdown_hold.enabled').exists(),'stop_record':json.loads((root/'reports/GPU_CAMPAIGN_STOPPED.json').read_text())['status'],'training_complete_marker_exists':(root/'training_runs/v3/TRAINING_COMPLETE.json').exists(),'datasets_and_disk_caches_preserved':True}
print(json.dumps(v))
'''

def main():
    client = connect()
    try:
        sftp = client.open_sftp()
        outcome = json.loads(sftp.open(REMOTE + '/reports/GPU_CAMPAIGN_STOPPED.json').read().decode())
        assert outcome['status'] == 'STOPPED_BY_USER'
        stop_time = datetime.fromisoformat(outcome['stopped_at_utc']).astimezone(timezone(timedelta(hours=8)))
        archive = RECEIPTS / ('stopped_' + stop_time.strftime('%Y%m%dT%H%M%S'))
        archive.mkdir(parents=True, exist_ok=True)
        items, skipped = [], []

        def walk(relative, recurse=True):
            for entry in sorted(sftp.listdir_attr(REMOTE + '/' + relative), key=lambda x: x.filename):
                rel = relative + '/' + entry.filename
                if stat.S_ISDIR(entry.st_mode):
                    if recurse:
                        walk(rel)
                    continue
                if not stat.S_ISREG(entry.st_mode) or Path(entry.filename).suffix.lower() not in ALLOWED:
                    skipped.append({'relative_path': rel, 'bytes': entry.st_size, 'reason': 'Non-text artifact excluded; not downloaded.'})
                    continue
                target = archive / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(REMOTE + '/' + rel, str(target))
                after = sftp.stat(REMOTE + '/' + rel)
                assert target.stat().st_size == entry.st_size == after.st_size, rel
                items.append({'relative_path': rel, 'bytes': target.stat().st_size, 'remote_mtime': after.st_mtime})
        walk('reports')
        walk('training_runs/v3')
        walk('training_execution', recurse=False)
        # Also retain campaign-root text receipts, if any.
        for entry in sftp.listdir_attr(REMOTE):
            if stat.S_ISREG(entry.st_mode) and Path(entry.filename).suffix.lower() in ALLOWED:
                target = archive / entry.filename
                sftp.get(REMOTE + '/' + entry.filename, str(target))
                assert target.stat().st_size == entry.st_size
                items.append({'relative_path': entry.filename, 'bytes': target.stat().st_size, 'remote_mtime': entry.st_mtime})
        stdin, stdout, stderr = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(VERIFY))
        raw = stdout.read().decode()
        error = stderr.read().decode()
        code = stdout.channel.recv_exit_status()
        if code:
            raise RuntimeError(error)
        verification = json.loads(raw)
        assert not verification['remaining_campaign_processes'], verification
        assert verification['gpu_query_exit_code'] == 0
        assert len(verification['gpu_compute_processes_csv'].strip().splitlines()) == 1
        assert verification['shutdown_hold_exists']
        assert not verification['training_complete_marker_exists']
        (archive / 'post_stop_verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
        manifest = {'status': 'ARCHIVED_STOPPED_BY_USER', 'archived_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    'remote_root': REMOTE, 'archive_directory': str(archive),
                    'file_count': len(items), 'total_bytes': sum(x['bytes'] for x in items),
                    'validation_steps': sorted(int(p.stem.split('_')[-1]) for p in (archive / 'training_runs/v3').glob('val_*.json')),
                    'checks': 'Each transferred file byte length compared with stable SFTP file size. No data/cache hash scan.',
                    'checkpoint_downloaded': False, 'checkpoint_deleted': False, 'shutdown_executed': False,
                    'files': items, 'excluded_files': skipped}
        (archive / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for rel in ['reports', 'training_runs/v3']:
            for source in (archive / rel).rglob('*'):
                if source.is_file():
                    target = RECEIPTS / source.relative_to(archive)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        loss_log = HERE / 'loss_exports/v3_training_export.log'
        loss_log.parent.mkdir(exist_ok=True)
        shutil.copyfile(archive / 'reports/v3_training.log', loss_log)
        summary = {'archive': str(archive), 'file_count': manifest['file_count'], 'bytes': manifest['total_bytes'],
                   'validation_steps': manifest['validation_steps'], 'verification': verification}
        (HERE / 'stopped_run_archive.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=True))
    finally:
        client.close()

if __name__ == '__main__':
    main()
