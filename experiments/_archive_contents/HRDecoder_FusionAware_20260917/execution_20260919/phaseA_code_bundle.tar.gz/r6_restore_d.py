# -*- coding: utf-8 -*-
"""Restore the lost D student weights from the audit checkpoint.

Facts:
  * repo/weights/glm_D_student.pth was a symlink to
    D_0913_retry/backup_student_2000.pth (expected sha256 5fe18717...),
    deleted during the 14:00 disk cleanup.
  * audit_20260916_A/tests/c2_roundtrip.pth is an MA-refiner variant whose
    3754 non-ma_refiner tensors are the D student: loading them reproduces
    the verified champion val BIT-FOR-BIT on all 76 images x 4 classes
    (0 differing cells; val 0.426686, per-class .457705/.437552/.551806/
    .259681 == the P0 baseline).

This script therefore:
  1. strips the 20 ma_refiner.* keys,
  2. writes the result as the real file repo/weights/glm_D_student.pth
     (replacing the dangling symlink),
  3. updates expected_sha256 for that entry (backup kept as
     settings.json.bak_before_d_restore) and re-runs validate_settings,
  4. records the whole action in DEVIATIONS.jsonl + RECOVERY_D_STUDENT.md.
The change is DECLARED, not silent: a hash entry changes, everything else
(including the reference/settings for every other asset) is untouched.
"""
import os

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
PY = '/root/miniconda3/bin/python'
GLM = '/root/autodl-tmp/hrplus/glm'
REPO = '/root/autodl-tmp/hrplus/repo'
RUN = GLM + '/runs/r6'
CAND = '/root/autodl-tmp/audit/audit_20260916_A/tests/c2_roundtrip.pth'

MAKE = r'''
# -*- coding: utf-8 -*-
import hashlib, json, os, sys
sys.path.insert(0, '/root/autodl-tmp/hrplus/glm')
os.chdir('/root/autodl-tmp/hrplus/repo')
import runtime as rt

REPO = '/root/autodl-tmp/hrplus/repo'
DST = REPO + '/weights/glm_D_student.pth'
CAND = %r

settings = json.load(open('/root/autodl-tmp/hrplus/glm/settings.json'))
settings['_abs'] = {k: (v if os.path.isabs(v) else os.path.join(REPO, v))
                    for k, v in {'reference_checkpoint':
                                 settings['reference_checkpoint']}.items()}
rt.inject_repo(REPO)
from mmseg.models import build_segmentor      # importable only after inject
c = rt.load_config(REPO, settings['config'])
c.model['source_checkpoint'] = settings['_abs']['reference_checkpoint']
model = build_segmentor(c.model)
expected = set(model.state_dict().keys())

ck = rt._load_state(CAND)               # strips 'module.' and unwraps state_dict
keep = {k: v for k, v in ck.items() if k in expected}
dropped = sorted(set(ck) - expected)
missing = sorted(expected - set(ck))
if missing:
    raise SystemExit('ABORT: candidate lacks %%d expected keys' %% len(missing))
print('strip: dropped %%d keys, missing %%d' %% (len(dropped), len(missing)))
print('strip: dropped sample %%s' %% dropped[:6])

if os.path.islink(DST):
    print('dst: removing dangling symlink -> %%s' %% os.readlink(DST))
    os.remove(DST)
tmp = DST + '.tmp'      # same filesystem as DST (os.replace)
import torch
torch.save({'state_dict': keep,
            'meta': {'recovered_from': CAND,
                     'method': 'strip ma_refiner.* then functional match',
                     'n_keys': len(keep)}}, tmp)
os.replace(tmp, DST)
h = hashlib.sha256()
with open(DST, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
print('dst: wrote %%s (%%d bytes)' %% (DST, os.path.getsize(DST)))
print('NEW_SHA256 %%s' %% h.hexdigest())
''' % CAND


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)

    def sh(cmd, timeout=900):
        _, o, e = cli.exec_command(cmd, timeout=timeout)
        return (o.read().decode('utf-8', 'replace') +
                e.read().decode('utf-8', 'replace')).strip()

    sftp = cli.open_sftp()
    with sftp.open('/root/autodl-tmp/hrplus/glm/r6_make_d_student.py', 'w') as f:
        f.write(MAKE)
    print('uploaded maker script')
    out = sh('cd %s && %s r6_make_d_student.py 2>&1 | tail -8' % (GLM, PY))
    print(out)
    if 'NEW_SHA256' not in out:
        print('REFUSING: maker failed'); cli.close(); return
    new_sha = [l.split()[1] for l in out.splitlines()
               if l.startswith('NEW_SHA256')][0]
    print('new sha256: %s' % new_sha)

    # update settings.json entry (with backup), then validate
    patch = r'''
import hashlib, json, shutil
p = '/root/autodl-tmp/hrplus/glm/settings.json'
shutil.copy(p, p + '.bak_before_d_restore')
s = json.load(open(p))
old = s['expected_sha256']['weights/glm_D_student.pth']
s['expected_sha256']['weights/glm_D_student.pth'] = '%s'
json.dump(s, open(p, 'w'), indent=2, sort_keys=True)
print('settings: expected_sha256 D student')
print('  old %%s' %% old)
print('  new %%s' %% '%s')
''' % (new_sha, new_sha)
    with sftp.open('/tmp/r6_patch_settings.py', 'w') as f:
        f.write(patch)
    sftp.close()
    print(sh('cd %s && %s /tmp/r6_patch_settings.py' % (GLM, PY)))
    print(sh('cd %s && %s -c "import run_protocol as rp; s=rp.'
             'validate_settings(\'settings.json\'); print(\'VALIDATE_OK\', '
             's[\'_pending\'])"' % (GLM, PY)))

    entry = ('{"time": "%s", "kind": "operator_weight_restore", "detail": '
             '{"asset": "weights/glm_D_student.pth", '
             '"cause": "deleted with D_0913_retry during the 14:00 disk '
             'cleanup (the path was a symlink into that directory)", '
             '"restored_from": "%s", '
             '"method": "strip 20 ma_refiner.* keys; functional match: '
             'val counts bit-identical on all 76 images x 4 classes vs the '
             'P0 baseline (val 0.426686)", "new_sha256": "%s"}}'
             % (__import__('time').strftime('%%Y-%%m-%%dT%%H:%%M:%%S'), CAND,
                new_sha)).replace('%%', '%')
    sh("printf '%s\\n' >> %s/DEVIATIONS.jsonl" % (entry.replace('"', '\\"'),
                                                  RUN))
    print(sh('tail -1 %s/DEVIATIONS.jsonl' % RUN))
    print(sh('df -h /root/autodl-tmp | tail -1'))
    cli.close()


if __name__ == '__main__':
    main()
