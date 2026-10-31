# -*- coding: utf-8 -*-
"""Generate auto_resume5.py from auto_resume.py (r1 -> r5 parameterization)."""
import io

src = io.open('auto_resume.py', encoding='utf-8').read()

REPL = [
    ("RUN = GLM + '/runs/r1'", "RUN = GLM + '/runs/r5'"),
    ("'/summary.json'", "'/round5_summary.json'"),
    ("summary.json exists", "round5_summary.json exists"),
    ("'run_protocol|run_guarded|ops.py watchdog'",
     "'run_round5|run_guarded|ops.py watchdog'"),
    ("if 'run_protocol' not in procs:", "if 'run_round5' not in procs:"),
    # relaunch with the round-5 script via run_guarded.sh SCRIPT env
    ("(PYTHON=%s nohup bash run_guarded.sh settings.json ",
     "(PYTHON=%s SCRIPT=run_round5.py nohup bash run_guarded.sh settings.json "),
    ("'RELAY: run_protocol NOT alive after relaunch'",
     "'RELAY: run_round5 NOT alive after relaunch'"),
    ("run_protocol NOT alive", "run_round5 NOT alive"),
]
for a, b in REPL:
    src = src.replace(a, b)

# ensure the upload list is round-5 (run_protocol.py already uploaded; keep)
src = src.replace(
    "        sftp.put(os.path.join(LOCAL_DIR, 'run_protocol.py'),\n"
    "                 GLM + '/run_protocol.py')\n",
    "        sftp.put(os.path.join(LOCAL_DIR, 'run_protocol.py'),\n"
    "                 GLM + '/run_protocol.py')\n"
    "        sftp.put(os.path.join(LOCAL_DIR, 'run_round5.py'),\n"
    "                 GLM + '/run_round5.py')\n")

io.open('auto_resume5.py', 'w', encoding='utf-8').write(src)

chk = io.open('auto_resume5.py', encoding='utf-8').read()
print('runs/r5 refs      :', chk.count('runs/r5'))
print('round5_summary    :', chk.count('round5_summary.json'))
print('SCRIPT=run_round5 :', chk.count('SCRIPT=run_round5.py'))
print('run_protocol refs :', chk.count('run_protocol'))
print('run_round5 refs   :', chk.count('run_round5'))
assert 'runs/r1' not in chk, 'r1 path remnant'
assert chk.count('summary.json') == chk.count('round5_summary.json'), 'mixed summary refs'
print('OK')
