set -eu
cd /root/autodl-tmp/HRDecoder_Q_20261003
/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,time,os
from qlib.common import read_json,sha
root=Path.cwd()
for seed in (42,2024):
    state=read_json(root/'runs'/f'Q_DDR_seed{seed}'/'status.json')
    assert state['stage']=='COMPLETE' and state['completed']==16000
lock=read_json(root/'reports/final_selection.json')
assert lock['selection']['selected']['test_eligible']
assert not (root/'reports/final_evaluation_state.json').exists(), 'Existing evaluation state: do not relaunch blindly'
with (root/'reports/final_evaluation.log').open('ab') as log:
    p=subprocess.Popen(['/root/miniconda3/envs/hrdecoder_q_20261003/bin/python','-u',str(root/'reports/run_locked_test.py')],
        cwd=str(root),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
time.sleep(5)
state=read_json(root/'reports/final_evaluation_state.json')
print(json.dumps({'pid':p.pid,'state':state,'lock_sha256':sha(root/'reports/final_selection.json'),
                  'recent_log':(root/'reports/final_evaluation.log').read_text(errors='replace').splitlines()[-8:]}))
assert state['stage']=='RUNNING', 'Final evaluation paused; preserve evidence and repair root cause'
PY
