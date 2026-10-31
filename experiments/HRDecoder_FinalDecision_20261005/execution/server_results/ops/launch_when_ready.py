"""Overlap required GPU subgates with data upload; launch full driver once ready."""
import sys,time,json,shlex,os,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,utc_now
from ops.remote import connect,command
from ops.deploy import main as deploy

def main():
    lock=ROOT/'reports/preparation_driver.lock'
    with lock.open('x') as f:json.dump({'pid':os.getpid(),'time':utc_now()},f)
    rt=read_json(ROOT/'configs/runtime.local.json');remote=rt['server']['remote_root'];statefile=ROOT/'reports/preparation_driver.json'
    try:
        deploy(source_only=True)
        started=time.monotonic()
        while True:
            code="""
from pathlib import Path
import json
r=Path(%r)
def read(p):return json.loads(p.read_text()) if p.exists() else None
compile=read(r/'reports/kernel_precompile_pid.json');pid=compile['pid'];log=(r/'reports/kernel_precompile.log').read_text(errors='replace')
early=read(r/'reports/early_gpu_gate_pid.json');gate=read(r/'reports/early_gpu_gates.json')
print(json.dumps({'compiler_alive':Path('/proc/'+str(pid)).exists(),'compiler_pass':log.rstrip().endswith('PINNED_KERNEL_BUILD_PASS'),'compiler_tail':log[-2500:],
 'parent_ready':Path('/root/autodl-tmp/HRDecoder_Finetune_20261004/runs/F12_seed42/checkpoint_u12000.pth').exists(),
 'early':early,'early_alive':Path('/proc/'+str(early['pid'])).exists() if early else False,'gate':gate,
 'early_tail':(r/'reports/early_gpu_gates.log').read_text(errors='replace')[-2500:] if (r/'reports/early_gpu_gates.log').exists() else None}))
"""%remote
            with connect() as c:current=json.loads(command(c,shlex.quote(rt['server_python'])+" - <<'PY'\n"+code+"\nPY",30))
            write_json(ROOT/'reports/preparation_remote_gates.json',current)
            if not current['compiler_alive'] and not current['compiler_pass']:raise RuntimeError('Concrete kernel compilation failure: '+current['compiler_tail'])
            if current['compiler_pass'] and current['parent_ready'] and current['early'] is None:
                launch="""
from pathlib import Path
import subprocess,json
r=Path(%r);assert not (r/'reports/early_gpu_gate_pid.json').exists();assert not (r/'reports/driver_state.json').exists()
with (r/'reports/early_gpu_gates.log').open('ab') as log:
 p=subprocess.Popen([%r,'-u','-c','from tests.gpu_acceptance import early;early()'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
result={'pid':p.pid,'stage':'REQUIRED_NATIVE_KERNEL_RESUME_SUBGATES'};(r/'reports/early_gpu_gate_pid.json').write_text(json.dumps(result));print(json.dumps(result))
"""%(remote,rt['server_python'])
                with connect() as c:print(command(c,shlex.quote(rt['server_python'])+" - <<'PY'\n"+launch+"\nPY",30),flush=True)
            if current['early'] and not current['early_alive'] and (current['gate'] or {}).get('status')!='PASS':raise RuntimeError('Required early GPU gate failed: '+str(current['early_tail']))
            assets=ROOT/'reports/required_assets_receipt.json'
            if assets.exists() and (current['gate'] or {}).get('status')=='PASS':
                assert read_json(assets)['stage']=='ALL_REQUIRED_ASSETS_SHA_VERIFIED'
                deploy(launch=True);write_json(statefile,{'stage':'DRIVER_LAUNCHED','time':utc_now()});return
            err=ROOT/'reports/required_assets_native_R.err.log'
            if err.exists() and 'Traceback' in err.read_text(errors='replace'):raise RuntimeError('Asset upload failed; preserve partial prefixes and diagnose '+str(err))
            write_json(statefile,{'stage':'OVERLAPPED_INPUT_TRANSFER_AND_REQUIRED_GPU_GATES','pid':os.getpid(),'time':utc_now(),'early_gate':(current['gate'] or {}).get('status'),'compiler_pass':current['compiler_pass']})
            if time.monotonic()-started>6*3600:raise TimeoutError('Preparation exceeded6h; preserve logs and diagnose concrete transfer/gate')
            time.sleep(30)
    except BaseException as e:
        write_json(statefile,{'stage':'TECHNICAL_PREPARATION_PAUSE','reason':str(e),'automatic_training_restart':False,'time':utc_now()});traceback.print_exc();raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
