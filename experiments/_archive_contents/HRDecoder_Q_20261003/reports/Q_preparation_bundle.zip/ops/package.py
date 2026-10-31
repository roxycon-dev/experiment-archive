"""Freeze and package source plus small acceptance records, excluding credentials/assets."""
from pathlib import Path
import json
import subprocess
import zipfile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib.common import digest, read_json, sha, utc_now, write_json


root = Path(__file__).resolve().parents[1]
source_paths = list(root.glob('*.py')) + list((root/'qlib').glob('*.py')) + list((root/'ops').glob('*.py')) + list((root/'ops').glob('*.sh'))
source_paths += [root/'configs/protocol.json']
source_paths += [p for p in (root/'vendor/dinov3').rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix != '.pyc']
files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(source_paths)}
official_commit = subprocess.check_output(['git', '-C', str(root/'vendor/dinov3'), 'rev-parse', 'HEAD'], text=True).strip()
source = {'files': files, 'sha256': digest(files), 'official_dinov3_commit': official_commit, 'created_at': utc_now()}
source['experiment_code_commit'] = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
write_json(root/'reports/source_identity.json', source)
runtime = read_json(root/'configs/runtime.local.json')
pretrained = read_json(root/'reports/pretraining_download.json')
runtime['pretrained_checkpoint'] = '/root/autodl-tmp/HRDecoder_Q_20261003/assets/'+pretrained['file']
runtime['dinov3_repo'] = '/root/autodl-tmp/HRDecoder_Q_20261003/vendor/dinov3'
runtime['ddr_root'] = '/root/HRDecoder/data/DDR_split_613'
runtime['p0_c_cache'] = '/root/autodl-tmp/HRDecoder_Ready_20261002/cache/val/champion'
runtime['p0_u_cache'] = '/root/autodl-tmp/HRDecoder_Ready_20261002/cache/val/U0'
runtime['license_acceptance_evidence'] = 'reports/pretraining_download.json'
runtime['stage'] = 'GPU_PREFLIGHT_PENDING'
write_json(root/'configs/runtime.server.json', runtime)
target = root/'reports/Q_preparation_bundle.zip'
include = source_paths + list((root/'reports').glob('*.summary.json')) + list((root/'configs').glob('*.json'))
include += [root/'reports'/x for x in ('data_identity.json', 'cpu_acceptance.json', 'p0_acceptance.json', 'source_identity.json', 'pretraining_download.json')]
include += list((root/'reports').glob('stream_*.jsonl')) + [root/'README.md']
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for path in sorted(set(include)):
        z.write(path, path.relative_to(root).as_posix())
write_json(root/'reports/package_identity.json', {'sha256': sha(target), 'bytes': target.stat().st_size,
    'source_sha256': source['sha256'], 'official_dinov3_commit': official_commit, 'signed_urls_or_passwords_included': False})
print('Prepared source bundle', target.stat().st_size, source['sha256'])
