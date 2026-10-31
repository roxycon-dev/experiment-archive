"""Reviewed replacement for the GLM draft. Linux; dry-run unless explicitly armed."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--execute-shutdown', action='store_true')
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ['--'] else args.cmd
    state = {'started_unix': time.time(), 'deadline_unix': args.deadline,
             'command': cmd, 'status': 'starting'}
    proc = None
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    try:
        if not cmd or time.time() >= args.deadline:
            raise RuntimeError('Missing command or expired deadline')
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, 'ab', buffering=0) as output:
            proc = subprocess.Popen(cmd, stdout=output, stderr=output, start_new_session=True)
            while proc.poll() is None and time.time() < args.deadline and not interrupted:
                time.sleep(min(1, max(0, args.deadline - time.time())))
            if proc.poll() is None:
                state['status'] = 'interrupted' if interrupted else 'deadline'
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            else:
                state['status'] = 'success' if proc.returncode == 0 else 'failure'
            state['returncode'] = proc.returncode
    except BaseException as exc:
        state['status'] = 'wrapper_error'
        state['error'] = repr(exc)
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        state['ended_unix'] = time.time()
        try:
            Path(args.log + '.status.json').write_text(json.dumps(state, indent=2))
            print(json.dumps(state), flush=True)
        finally:
            if args.execute_shutdown:
                subprocess.run(['/bin/bash', '/usr/bin/shutdown'], timeout=30, check=True)
            else:
                print('DRY_RUN_SHUTDOWN', flush=True)


if __name__ == '__main__':
    main()
