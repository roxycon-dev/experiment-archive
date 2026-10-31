"""Commit an adaptive operations-only budget policy before M1's 2k result."""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirror_remote import ROOT, REMOTE, connect, remote_exec, write_json, file_sha


def main():
    retry = '--repair' in sys.argv
    policy = dict(version='adaptive-cost-gate-2000-v1', declared_at=time.time(),
        source_sha256='0dc17cf13e331d2d5a12a4b28fe67db75422547f0c078305b95e543a8d9f50d4',
        minimum_gap=.01, trend_epsilon=.0001, bootstrap_samples=2000, bootstrap_seed=7303,
        applies_to='Both complete 76-image M0/M1 nodes at 2000; unchanged training/data/optimizer',
        action='Jointly stop at 2000 only if both historical bests are at least 1pp below old deployment, neither rises from 1k, and both image-bootstrap 95% upper bounds vs old are below zero; otherwise retain original 4k/8k rule',
        authority='User explicitly permits adaptive changes and requires fastest useful experiment, stopping unpromising directions',
        interpretation='Budget decision, not a convergence or general architecture claim')
    path = ROOT/'reports/adaptive_budget_policy.json'
    c = connect()
    try:
        pre = json.loads(remote_exec(c, "/root/miniconda3/bin/python - <<'PY'\nfrom pathlib import Path\nimport json\nr=Path('"+REMOTE+"');h=json.loads((r/'work/M_seed42/M1/history.json').read_text());print(json.dumps(dict(nodes=[x['update'] for x in h],policy_exists=(r/'reports/adaptive_budget_policy.json').exists())))\nPY", 20))
        if retry:
            assert pre['policy_exists'] and path.exists()
            policy = json.loads(path.read_text(encoding='utf-8'))
            assert policy['declared_before_M1_2000_complete']
        else:
            assert 2000 not in pre['nodes'] and not pre['policy_exists'], pre
            policy['declared_before_M1_2000_complete'] = True
            write_json(path, policy)
        with c.open_sftp() as sftp:
            for local, rel in [(ROOT/'ops/adaptive_budget_guard.py','ops/adaptive_budget_guard.py'),
                               (path,'reports/adaptive_budget_policy.json')]:
                sftp.put(str(local), REMOTE+'/'+rel)
        code = """from pathlib import Path
import subprocess,os,json,hashlib
r=Path(REMOTE);policy=r/'reports/adaptive_budget_policy.json'
assert hashlib.sha256(policy.read_bytes()).hexdigest()==EXPECTED
with (r/'reports/adaptive_budget_guard.log').open('ab') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','ops/adaptive_budget_guard.py'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
out=dict(pid=p.pid,time=__import__('time').time(),policy_sha256=EXPECTED)
(r/'reports/adaptive_budget_guard_pid.json').write_text(json.dumps(out));print(json.dumps(out))
""".replace('REMOTE',repr(REMOTE)).replace('EXPECTED',repr(file_sha(path)))
        out = json.loads(remote_exec(c,"/root/miniconda3/bin/python - <<'PY'\n"+code+"\nPY",20))
        write_json(ROOT/'reports/adaptive_budget_guard_pid.json',out)
        print(json.dumps(out))
    finally:
        c.close()


if __name__=='__main__':main()
