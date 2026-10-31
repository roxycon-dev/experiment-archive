"""Dry-run-only tests for ops/guard.py. Never passes --execute-shutdown."""
import datetime
import json
import os
from pathlib import Path
import shutil
import signal  # explicit import for signal tests
import subprocess
import sys
import tempfile
import time
import unittest

GUARD = str(Path(__file__).resolve().parent.parent / 'ops' / 'guard.py')
IS_LINUX = sys.platform.startswith('linux')


def deadline_in(seconds):
    dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=seconds)
    return dt.isoformat()


def _pid_dead(pid):
    """A pid is considered dead if it no longer exists OR is a zombie
    ('Z' in /proc/<pid>/stat) — a zombie in the reaped group is not running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    try:
        with open('/proc/%d/stat' % pid) as f:
            state = f.read().rsplit(') ', 1)[1].split()[0]
        return state == 'Z'
    except OSError:
        return True


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='guardtest_')
        self.log = os.path.join(self.tmp, 'logs', 'run.log')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_guard(self, extra, timeout=60):
        return subprocess.run(
            [sys.executable, GUARD, '--log', self.log] + extra,
            capture_output=True, text=True, timeout=timeout)

    def assert_dry_run(self, res):
        self.assertIn('DRY_RUN_SHUTDOWN', res.stdout)
        self.assertNotIn('SHUTDOWN_NONZERO_STATUS', res.stdout)

    def read_state(self):
        with open(self.log + '.status.json') as f:
            return json.load(f)

    @unittest.skipUnless(IS_LINUX, 'linux only')
    def test_success(self):
        res = self.run_guard(['--deadline-utc', deadline_in(60),
                              sys.executable, '-c', 'pass'])
        self.assertEqual(res.returncode, 0)
        self.assertEqual(self.read_state()['status'], 'success')
        self.assert_dry_run(res)

    @unittest.skipUnless(IS_LINUX, 'linux only')
    def test_failure_propagates(self):
        res = self.run_guard(['--deadline-utc', deadline_in(60),
                              sys.executable, '-c', 'import sys; sys.exit(3)'])
        self.assertEqual(res.returncode, 3)
        self.assertEqual(self.read_state()['status'], 'failure')
        self.assert_dry_run(res)

    @unittest.skipUnless(IS_LINUX, 'linux only')
    def test_timeout_124(self):
        res = self.run_guard(
            ['--deadline-utc', deadline_in(2), sys.executable, '-c',
             'import time; time.sleep(30)'], timeout=30)
        self.assertEqual(res.returncode, 124)
        self.assertEqual(self.read_state()['status'], 'timeout')
        self.assert_dry_run(res)

    @unittest.skipUnless(IS_LINUX, 'linux only')
    def test_process_tree_cleanup_on_timeout(self):
        # Child spawns a grandchild sleeping detached; guard must SIGTERM the
        # whole group on timeout, including the nohup'd grandchild.
        script = os.path.join(self.tmp, 'parent.sh')
        with open(script, 'w') as f:
            f.write('#!/bin/bash\nnohup sleep 300 &\necho $! > %s/gc.pid\nsleep 300\n'
                    % self.tmp)
        os.chmod(script, 0o755)
        res = self.run_guard(['--deadline-utc', deadline_in(3), '/bin/bash', script],
                             timeout=45)
        self.assertEqual(res.returncode, 124)
        time.sleep(1.5)
        with open(os.path.join(self.tmp, 'gc.pid')) as f:
            gc = int(f.read().strip())
        self.assertTrue(_pid_dead(gc))
        self.assert_dry_run(res)

    @unittest.skipUnless(IS_LINUX, 'linux only')
    def test_normal_leader_exit_still_cleans_child(self):
        # Leader exits 0 on its own but leaves a detached descendant behind;
        # guard must clean the whole process group even on success.
        script = os.path.join(self.tmp, 'leaver.sh')
        with open(script, 'w') as f:
            f.write('#!/bin/bash\nsetsid --fork sleep 300 &\necho $! > %s/gc.pid\nexit 0\n'
                    % self.tmp)
        os.chmod(script, 0o755)
        res = self.run_guard(['--deadline-utc', deadline_in(30), '/bin/bash', script],
                             timeout=45)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(self.read_state()['status'], 'success')
        time.sleep(1.5)
        with open(os.path.join(self.tmp, 'gc.pid')) as f:
            gc = int(f.read().strip())
        self.assertTrue(_pid_dead(gc))
        self.assert_dry_run(res)

    @unittest.skipUnless(IS_LINUX, 'linux only')
    def test_launch_failure(self):
        res = self.run_guard(
            ['--deadline-utc', deadline_in(60), '/nonexistent/binary/xyz'])
        self.assertEqual(res.returncode, 125)
        st = self.read_state()
        self.assertEqual(st['status'], 'wrapper_error')
        self.assert_dry_run(res)

    def test_invalid_deadlines(self):
        # Correct policy: preflight rejection exits 2, does NOT shut down,
        # and does NOT launch anything (dry-run marker printed consistently).
        now = datetime.datetime.now(datetime.timezone.utc)
        expired = (now - datetime.timedelta(seconds=1)).isoformat()
        naive = (datetime.datetime.utcnow() + datetime.timedelta(hours=1)).isoformat()
        too_far = deadline_in(13 * 3600)
        for bad in (expired, naive, too_far):
            res = self.run_guard(['--deadline-utc', bad,
                                  sys.executable, '-c', 'pass'])
            self.assertEqual(res.returncode, 2, msg=bad)
            st = self.read_state()
            self.assertEqual(st['status'], 'preflight_rejected', msg=bad)
            self.assertEqual(st['command'], [sys.executable, '-c', 'pass'])
            self.assertNotIn('pid', st)
            self.assert_dry_run(res)
        # Z suffix is valid UTC and must be accepted.
        res = self.run_guard(['--deadline-utc',
                              (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=60))
                              .replace(tzinfo=None)
                              .isoformat() + 'Z',
                              sys.executable, '-c', 'pass'])
        self.assertEqual(res.returncode, 0, msg=res.stdout + res.stderr)

    def test_missing_command_rejected(self):
        res = self.run_guard(['--deadline-utc', deadline_in(60)])
        self.assertEqual(res.returncode, 2)
        self.assertEqual(self.read_state()['status'], 'preflight_rejected')
        self.assert_dry_run(res)

    def test_state_written_when_log_parent_missing(self):
        # Preflight rejection must still write state atomically even though
        # the log's parent directory does not exist yet.
        bad = (datetime.datetime.now(datetime.timezone.utc)
               - datetime.timedelta(seconds=1)).isoformat()
        res = self.run_guard(['--deadline-utc', bad,
                              sys.executable, '-c', 'pass'])
        self.assertEqual(res.returncode, 2)
        self.assertTrue(os.path.isfile(self.log + '.status.json'))

    @unittest.skipUnless(IS_LINUX, 'linux only')
    def test_signal_cleanup(self):
        p = subprocess.Popen(
            [sys.executable, GUARD, '--log', self.log,
             '--deadline-utc', deadline_in(120),
             sys.executable, '-c', 'import time; time.sleep(90)'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(3)
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=45)
        self.assertEqual(p.returncode, 130)
        self.assertEqual(self.read_state()['status'], 'interrupted')
        self.assertIn('DRY_RUN_SHUTDOWN', out)

    @unittest.skipUnless(IS_LINUX, 'linux only')
    def test_no_network_no_creds(self):
        src = open(GUARD).read()
        for banned in ('requests', 'urllib', 'socket.socket', 'http', 'paramiko',
                       'boto3', 'AWS_', 'os.environ['):
            self.assertNotIn(banned, src)


if __name__ == '__main__':
    unittest.main()
