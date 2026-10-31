# -*- coding: utf-8 -*-
"""Main deployment: upload GLM code + settings to the server, verify
hashes, compile-check on server python 3.8, run unit tests there, run the
GPU smoke, init the budget ledger, and launch run_guarded.sh.

Every stage asserts; any failure aborts BEFORE the launch (never leaves a
half-deployed running state)."""
import hashlib
import json
import os
import posixpath
import sys
import time

import paramiko

HOST, PORT, PASSWORD = '[REDACTED_USE_ENVIRONMENT]', 19057, 'NKgdrekCg6bo'
LOCAL = r'D:\dataset\experiment\HRDecoder_FusionAware_20260917'
PY = '/root/miniconda3/bin/python'
REPO = '/root/autodl-tmp/hrplus/repo'
GLM = '/root/autodl-tmp/hrplus/glm'
RUN = GLM + '/runs/r1'
BACKUP = GLM + '/backups'
CACHE = '/root/autodl-tmp/hrplus/cache/ref_train_v3'

# billing: container PID1 started 2026-09-17 23:48:11 +08:00 (=15:48:11Z);
# total lease 10h -> other_cost = 50 - 1.98*10 = 30.2 yuan
LEDGER_STARTED_AT = '2026-09-17T15:48:11+00:00'
LEDGER_OTHER_COST = 30.2

SETTINGS = {
    'repo': REPO,
    'config': 'configs/lesion/exp/hrplus_patch_dice.py',
    'data_root': REPO + '/data/DDR_split_613',
    'student_checkpoint': 'weights/glm_D_student.pth',
    'reference_checkpoint': 'weights/b_weights.pth',
    'cache_dir': CACHE,
    'champion': {
        'head': 'weights/h4_best.pth',
        'members': [
            {'name': 'D', 'config': 'configs/lesion/exp/hrplus_patch_dice.py',
             'checkpoint': 'weights/glm_D_student.pth',
             'load': 'student_only', 'weight': 3},
            {'name': 'E', 'config': 'configs/lesion/exp/hrplus_mares.py',
             'checkpoint': 'weights/glm_E_1500.pth',
             'load': 'full', 'weight': 2},
            {'name': 'B', 'config': 'configs/lesion/exp/hr100_b.py',
             'checkpoint': 'weights/b_weights.pth',
             'load': 'full', 'weight': 1},
            {'name': 'E4', 'config': 'configs/lesion/exp/e4_detail_split613.py',
             'checkpoint': 'weights/glm_E4_36000.pth',
             'load': 'full', 'weight': 1},
        ],
    },
    'expected_sha256': {
        'configs/lesion/exp/hrplus_patch_dice.py':
            '2d687086e28533d4bb080f31c12c4abb802d5293fc4f2a463f9f0bd8f36452f9',
        'configs/lesion/exp/hrplus_mares.py':
            '591e4786abaa35b8dfc9a6476c45a8c9eb387544c37e0d56b6a499b683e100b4',
        'configs/lesion/exp/hr100_b.py':
            '556e7c2f1557c6658f6233f1fe1601b8d0b1c8382cc6bdaaaf108a712da3ed54',
        'configs/lesion/exp/e4_detail_split613.py':
            '65b2c5eb88db2790c1e00dbcb6e18deaf8fc6abc5bf3f148be9ae70d2f06fc1c',
        'weights/glm_D_student.pth':
            '5fe18717b62d8fc3ab4846a50c046588e2af05abbbfdb3a958f4794018b319ea',
        'weights/b_weights.pth':
            'abd74330b5ccaad291ee1d2a74739aa7901366f0edf40cf151027cdd61e1c84d',
        'weights/glm_E_1500.pth':
            'a18ff92b4da24d870554f17856983d4be042386cacbcaba0d4e13f3f248ed5e5',
        'weights/glm_E4_36000.pth':
            'ef63779928a1267c5b7e9050bf82ddb3ed88b62d830a9275d7018fd90272e112',
        'weights/h4_best.pth':
            '14757e02de49f6f065d4ef82cb8b1c4ee8a6a7f3f66cb1867276993950f80c50',
    },
    'final_reserve_hours': 2.5,
    'gpu_id': 0,
}

CODE_FILES = ['fusion_core.py', 'runtime.py', 'ops.py', 'run_protocol.py',
              'storage_preflight.py', 'smoke_gpu.py']
TEST_FILES = ['test_core.py', 'test_ops.py', 'test_protocol.py',
              'test_runtime.py', 'test_storage.py']

SYMLINKS = {
    REPO + '/weights/glm_D_student.pth':
        '/root/autodl-tmp/hrplus/D_0913_retry/backup_student_2000.pth',
    REPO + '/weights/glm_E_1500.pth':
        '/root/autodl-tmp/hrplus/results/E_0916/best_mIoU_iter_1500.pth',
    REPO + '/weights/glm_E4_36000.pth':
        '/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth',
}


def sh(cli, cmd, timeout=120, check=True):
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', 'replace')
    err = stderr.read().decode('utf-8', 'replace')
    rc = stdout.channel.recv_exit_status()
    if check and rc != 0:
        raise RuntimeError('cmd failed rc=%d: %s\n--stdout--\n%s\n--stderr--\n%s'
                           % (rc, cmd, out[-3000:], err[-3000:]))
    return out, err, rc


def local_sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def upload(cli_sftp, local_path, remote_path, normalize_lf=False):
    data = open(local_path, 'rb').read()
    if normalize_lf:
        data = data.replace(b'\r\n', b'\n')
    with cli_sftp.open(remote_path, 'wb') as f:
        f.write(data)


def stage(name):
    print('\n' + '=' * 66)
    print('== STAGE: %s' % name)
    print('=' * 66, flush=True)


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username='root', password=PASSWORD,
                timeout=30, banner_timeout=30, auth_timeout=30)
    sftp = cli.open_sftp()

    # ---------------- A: dirs + symlinks ----------------
    stage('A dirs + symlinks')
    sh(cli, 'mkdir -p %s %s %s %s %s/tests' % (GLM, RUN, BACKUP, CACHE, GLM))
    for link, target in SYMLINKS.items():
        sh(cli, 'ln -sfn %s %s' % (target, link))
        sh(cli, 'test -e %s' % link)   # target must exist
    print('symlinks ok')

    # ---------------- B: upload ----------------
    stage('B upload code/tests/settings/head')
    for f in CODE_FILES:
        upload(sftp, os.path.join(LOCAL, f), GLM + '/' + f)
    for f in TEST_FILES:
        upload(sftp, os.path.join(LOCAL, 'tests', f), GLM + '/tests/' + f)
    # guard script: force LF line endings (bash chokes on CRLF)
    upload(sftp, os.path.join(LOCAL, 'run_guarded.sh'),
           GLM + '/run_guarded.sh', normalize_lf=True)
    sh(cli, 'chmod +x %s/run_guarded.sh' % GLM)
    # MA correction head -> repo/weights/h4_best.pth (small)
    upload(sftp, os.path.join(
        LOCAL, '..', 'HRDecoder_审查轮_20260916', 'audit_runs',
        'audit_20260916_A', 'h4_best.pth'), REPO + '/weights/h4_best.pth')
    # settings.json
    with sftp.open(GLM + '/settings.json', 'wb') as f:
        f.write(json.dumps(SETTINGS, indent=1, ensure_ascii=True).encode())
    print('uploaded %d code files, %d test files, settings.json, h4_best.pth'
          % (len(CODE_FILES), len(TEST_FILES)))

    # ---------------- C: verify every hash ----------------
    stage('C hash verification (fail-closed)')
    for rel, exp in sorted(SETTINGS['expected_sha256'].items()):
        out, _, _ = sh(cli, 'sha256sum %s' % posixpath.join(REPO, rel))
        got = out.split()[0]
        assert got == exp, 'sha mismatch %s: %s != %s' % (rel, got, exp)
        print('  ok %s' % rel)
    # uploaded code files must match local bytes
    for f in CODE_FILES + ['run_guarded.sh']:
        out, _, _ = sh(cli, 'sha256sum %s/%s' % (GLM, f))
        got = out.split()[0]
        loc = local_sha(os.path.join(LOCAL, f))
        if f == 'run_guarded.sh':      # LF-normalized on upload
            loc = hashlib.sha256(
                open(os.path.join(LOCAL, f), 'rb').read()
                .replace(b'\r\n', b'\n')).hexdigest()
        assert got == loc, 'code upload mismatch %s' % f
    print('code upload verified')

    # ---------------- D: py_compile on server python 3.8 ----------------
    stage('D py_compile (python 3.8.10)')
    out, err, rc = sh(cli, 'cd %s && %s -m py_compile %s' %
                      (GLM, PY, ' '.join(CODE_FILES)), check=False)
    print(out.strip() or '(no output)')
    assert rc == 0, 'py_compile failed:\n%s' % err
    print('py_compile ok')

    # ---------------- E: unit tests on server ----------------
    stage('E unit tests on server')
    out, err, rc = sh(cli, 'cd %s && PYTHONWARNINGS=ignore %s -m unittest '
                      'discover -s tests 2>&1' % (GLM, PY), timeout=300,
                      check=False)
    print(out[-2500:])
    assert rc == 0 and '\nOK' in out, 'server unit tests failed'
    print('server unit tests OK')

    # ---------------- F: GPU smoke ----------------
    stage('F GPU smoke (5 gates)')
    out, err, rc = sh(cli, 'cd %s && %s smoke_gpu.py 2>&1' % (GLM, PY),
                      timeout=1200, check=False)
    print(out)
    if rc != 0:
        print('STDERR:', err[-2000:])
    assert rc == 0 and 'ALL GATES PASS' in out, 'GPU smoke failed'

    # ---------------- G: ledger init ----------------
    stage('G budget ledger init')
    out, _, _ = sh(cli, 'cd %s && %s ops.py init --run-dir %s '
                  '--started-at "%s" --other-cost %s' %
                  (GLM, PY, RUN, LEDGER_STARTED_AT, LEDGER_OTHER_COST))
    print(out.strip())

    # ---------------- H: launch guarded run ----------------
    stage('H launch run_guarded.sh')
    sh(cli, 'cd %s && PYTHON=%s nohup bash run_guarded.sh settings.json '
          '%s %s > guard.log 2>&1 & echo LAUNCHED $!' % (GLM, PY, RUN, BACKUP))
    time.sleep(20)
    out, _, _ = sh(cli, 'ps aux | grep -E "run_protocol|run_guarded|watchdog" '
                  '| grep -v grep || true')
    print(out)
    assert 'run_protocol' in out or 'run_guarded' in out, \
        'guarded run not alive after launch'
    out, _, _ = sh(cli, 'for i in 1 2 3 4 5 6; do sleep 10; '
                  'if [ -s %s/protocol.log ]; then break; fi; done; '
                  'tail -20 %s/protocol.log 2>/dev/null; echo ---GUARD---; '
                  'tail -5 %s/guard.log 2>/dev/null' % (RUN, RUN, GLM))
    print(out)
    print('\nDEPLOYMENT COMPLETE -- guarded protocol is running')
    cli.close()


if __name__ == '__main__':
    main()
