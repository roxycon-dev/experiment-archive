"""Dispatch the verified one-shot stop and retain an honest local acknowledgement."""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.remote import connect
from qlib.common import read_json,write_json,sha,utc_now

REMOTE='/root/autodl-tmp/HRDecoder_Q_20261003'
handoff=read_json(ROOT/'reports/temporary_stop_handoff.json')
assert handoff['local_results_sha_verified'] and not handoff['original_plan_complete']
receipt_path=ROOT/'reports/temporary_shutdown_receipt.json'
assert not receipt_path.exists(),'A stop request already exists; inspect it instead of dispatching again'
paths=['reports/perform_temporary_stop.py','reports/execution_scope_override.json',
       'reports/original_plan_completion.json','reports/temporary_stop_handoff.json',
       'reports/preparation_status.json','reports/external_readiness.json',
       'reports/idrid_polygon_conversion_receipt.json',
       'reports/Q_DDR_stage_report_receipt.json']
with connect() as client:
    with client.open_sftp() as sftp:
        for rel in paths:sftp.put(str(ROOT/rel),REMOTE+'/'+rel)
    manifest={rel:sha(ROOT/rel) for rel in paths}
    command="/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+f"expected={manifest!r}\n"+"""
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_Q_20261003')
for name,h in expected.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
request=r/'reports/temporary_shutdown_request.json'
assert not request.exists(),'Already requested; no duplicate shutdown'
handoff=json.loads((r/'reports/temporary_stop_handoff.json').read_text())
assert handoff['local_results_sha_verified'] and not handoff['original_plan_complete']
assert json.loads((r/'reports/final_evaluation_state.json').read_text())['stage']=='COMPLETE'
x={'stage':'SCHEDULED','requested_at':datetime.now(timezone.utc).isoformat(),
 'reason':'Latest user instruction: do not leave server idle overnight; official external data/access missing',
 'original_plan_complete':False,'platform_OFF_verified':False,'automatic_restart':False,
 'handoff_sha256':hashlib.sha256((r/'reports/temporary_stop_handoff.json').read_bytes()).hexdigest()}
request.write_text(json.dumps(x,indent=2)+'\n')
with (r/'reports/temporary_shutdown_worker.log').open('wb') as log:
 p=subprocess.Popen([sys.executable,str(r/'reports/perform_temporary_stop.py')],cwd=r,
     stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
x['worker_pid']=p.pid
request.write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x))
PY
"""
    _,out,err=client.exec_command(command,timeout=60)
    stdout,stderr=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(stderr)
    receipt=json.loads(stdout)
receipt.update(local_recorded_at=utc_now(),uploaded_policy_sha_verified=True,
    platform_OFF_verified=False,platform_status_channel='Unavailable; no account/API state query authorized or accessible')
write_json(receipt_path,receipt)
print({'stop_request':receipt['stage'],'worker_pid':receipt['worker_pid'],'platform_OFF_verified':False})

time.sleep(8)
try:
    with connect() as client:
        with client.open_sftp() as sftp:
            with sftp.open(REMOTE+'/reports/temporary_shutdown_request.json','r') as f:
                remote=json.loads(f.read().decode())
            with sftp.open(REMOTE+'/reports/temporary_shutdown_worker.log','r') as f:
                log=f.read()
        receipt.update(remote_request=remote,ssh_reachable_after_request=True,
            worker_log_bytes=len(log),worker_log_sha256=__import__('hashlib').sha256(log).hexdigest(),
            post_request_checked_at=utc_now())
        if remote['stage']=='SHUTDOWN_COMMAND_FAILED':raise RuntimeError('Official platform shutdown command failed')
except Exception as e:
    receipt.update(post_request_connection_error=type(e).__name__,post_request_checked_at=utc_now())
    # An unreachable SSH endpoint is evidence of disconnect, not platform OFF.
write_json(receipt_path,receipt)
status=read_json(ROOT/'reports/preparation_status.json')
status.update(stage='TEMPORARY_STOP_REQUESTED_OFF_NOT_YET_VERIFIED',updated_at=utc_now(),
    temporary_stop_requested=True,temporary_stop_receipt='reports/temporary_shutdown_receipt.json',
    platform_OFF_verified=False,original_plan_complete=False,suppress_repeated_ssh_poll_while_stopped=True)
write_json(ROOT/'reports/preparation_status.json',status)
print({'stop_request_dispatched':True,'remote_stage':receipt.get('remote_request',{}).get('stage'),
    'post_request_connection_error':receipt.get('post_request_connection_error'),
    'platform_OFF_verified':False,'original_plan_complete':False})
