"""Upload only new inference preparation files, then stage asynchronously on the servers."""
from pathlib import Path
from datetime import datetime,timezone
import json
import shlex
from training_remote_control import connect,execute,ROOT,REMOTE

files=['stage_evaluation59.py','bootstrap_evaluation59.sh','cache_common_system59.py','evaluation59_common.py','launch_common_cache59.py','exact_weight_transport.py']
with connect() as client:
    with client.open_sftp() as s:
        for name in files:s.put(str(ROOT/'scripts'/name),REMOTE+'/scripts/'+name)
        s.put(str(ROOT/'plan/paired_evaluation_runtime.md'),REMOTE+'/plan/paired_evaluation_runtime.md')
    log=REMOTE+'/log_stage_evaluation59_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.txt'
    command='nohup /root/miniconda3/bin/python -u '+shlex.quote(REMOTE+'/scripts/stage_evaluation59.py')+' > '+shlex.quote(log)+' 2>&1 < /dev/null &'
    result=execute(client,command)
    assert result['exit_code']==0,result
    print(json.dumps({'stage':'PREPARATION_DISPATCHED','remote_log':log}))
