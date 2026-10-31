"""Wait for this installation, then exec the controller; setup never shuts down."""
import json
import os
from pathlib import Path
import time
import controller as c

if __name__=='__main__':
    with (c.BASE/'bootstrap_started.json').open('x') as stream:
        json.dump({'pid':os.getpid(),'started_at':c.now(),'source_sha256':c.SOURCE_SHA,
                   'installation_pid':1138,'setup_wait_limit_seconds':1800},stream)
    try:
        limit=time.monotonic()+1800
        while not (c.BASE/'receipts/environment_ready').is_file():
            if time.monotonic()>limit:
                raise RuntimeError('Environment setup did not finish within the remaining 30-minute setup window')
            command=Path('/proc/1138/cmdline')
            if not command.exists() or str(c.BASE).encode() not in command.read_bytes():
                raise RuntimeError('Owned environment installer exited without readiness marker')
            time.sleep(5)
        if not (c.BASE/'receipts/assets_ready.json').is_file():
            raise RuntimeError('Accepted data and pretrained weight are not ready')
        os.execv(c.PYTHON,[c.PYTHON,str(c.BASE/'controller.py')])
    except BaseException as error:
        c.state.update(outcome='SETUP_FAILED',error_type=type(error).__name__,error=str(error))
        c.finish_and_shutdown()
