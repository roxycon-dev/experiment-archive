"""Upload small registered source files and run CPU checks; do not start any GPU task."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'local_preparation'))
from remote import connect
REMOTE = '/root/autodl-tmp/HRDecoder_v3_20261008'


def main():
    proof = (ROOT / 'sacd_execution/diagnosis_20261009/verify_feature_isolation.py').read_text(encoding='utf-8')
    proof = proof.replace('CORE=HERE.parent', "CORE=HERE.parent/'sacd_execution'")
    proof = proof.replace('new=copy.deepcopy(old)', 'new=copy.deepcopy(old)\nfrom intervention import FeatureIsolatedPrivate\nnew.__class__=FeatureIsolatedPrivate')
    (HERE / 'verify_feature_isolation.py').write_text(proof, encoding='utf-8')
    (HERE / 'feature_isolation.patch').write_bytes((ROOT / 'sacd_execution/diagnosis_20261009/feature_isolation.patch').read_bytes())
    client = connect()
    try:
        with client.open_sftp() as ftp:
            for directory in ('sacd_micro_execution', 'sacd_isolated_execution'):
                remote_dir = REMOTE + '/' + directory
                try:
                    ftp.mkdir(remote_dir)
                except OSError:
                    pass
                local = ROOT / directory
                names = [p for p in local.iterdir() if p.suffix in ('.py', '.json', '.patch', '.md')
                         and p.name not in ('deploy.py', 'prepare_candidate.py')
                         and not p.name.endswith('PASS.json')]
                for path in names:
                    ftp.put(str(path), remote_dir + '/' + path.name)
        code = '''import json,os,py_compile,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
checks=[]
for directory,names in [('sacd_micro_execution',['verify_feature_isolation.py']),('sacd_isolated_execution',['verify_sacd.py','verify_checkpoint.py','verify_feature_isolation.py'])]:
 for source in (root/directory).glob('*.py'):py_compile.compile(str(source),doraise=True)
 for name in names:
  result=subprocess.run(['/root/miniconda3/bin/python',str(root/directory/name)],cwd=str(root/directory),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120)
  if result.returncode:raise RuntimeError(directory+'/'+name+' '+result.stderr[-4000:])
  checks.append({'directory':directory,'check':name,'status':'PASS'})
print(json.dumps({'status':'CPU_DEPLOYMENT_PASS','checks':checks,'GPU_used':False,'training_process_changed':False}))
'''
        _, out, err = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(code), timeout=180)
        output, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error)
        receipt = json.loads(output)
        receipt['source_files'] = {}
        with client.open_sftp() as ftp:
            for directory in ('sacd_micro_execution', 'sacd_isolated_execution'):
                for filename in ('FEATURE_ISOLATION_CPU_PASS.json', 'CPU_SACD_PASS.json', 'CHECKPOINT_RESUME_PASS.json'):
                    try:
                        raw = ftp.open(REMOTE + '/' + directory + '/' + filename, 'rb').read()
                    except OSError:
                        continue
                    target = ROOT / directory / filename
                    target.write_bytes(raw)
                receipt['source_files'][directory] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (ROOT / directory).glob('*.py') if p.name not in ('deploy.py', 'prepare_candidate.py')}
        (HERE / 'DEPLOYMENT_CPU_PASS.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt))
    finally:
        client.close()


if __name__ == '__main__':
    main()
