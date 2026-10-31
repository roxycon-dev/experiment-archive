"""Upload closure evidence and dispatch one final stop; never equate SSH loss with OFF."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.remote import connect
from qlib.common import read_json,write_json,sha,utc_now
REMOTE='/root/autodl-tmp/HRDecoder_Q_20261003'
RECEIPT=ROOT/'reports/final_shutdown_receipt.json'

def main():
    assert not RECEIPT.exists(), 'An existing shutdown request requires review; no duplicate dispatch'
    delivery=read_json(ROOT/'reports/original_plan_delivery_manifest.json')
    assert delivery['stage']=='COMPLETE_SCIENTIFIC_DELIVERY_LOCAL_SHA_AND_CLEANUP'
    assert delivery['shutdown_authorized'] and not delivery['shutdown_dispatched']
    completion=read_json(ROOT/'reports/original_plan_completion.json')
    assert completion['scientific_scope_complete'] and completion['delivery_complete'] and completion['shutdown_allowed']
    paths=['reports/perform_final_shutdown.py','reports/original_plan_delivery_manifest.json',
           'reports/original_plan_completion.json','reports/preparation_status.json','reports/execution_scope_override.json',
           'reports/original_plan_report_receipt.json','reports/HRDecoder_原方案完整实验报告_20261004.md',
           'reports/original_plan_experiment_result.json','reports/paired_validation_all_nodes_24k.json',
           'reports/supervision_comparison_common24k.json','reports/final_test_uncertainty_and_error_buckets.json',
           'reports/HRDecoder_paired_validation_20261004.png','reports/HRDecoder_paired_validation_20261004.svg']
    with connect() as client:
        with client.open_sftp() as sftp:
            for rel in paths: sftp.put(str(ROOT/rel),REMOTE+'/'+rel)
        expected={rel:sha(ROOT/rel) for rel in paths}
        code='expected=%r\n'%expected+'''
from pathlib import Path
import hashlib,json,subprocess,sys
from datetime import datetime,timezone
r=Path('/root/autodl-tmp/HRDecoder_Q_20261003')
for relative,h in expected.items(): assert hashlib.sha256((r/relative).read_bytes()).hexdigest()==h
request=r/'reports/final_shutdown_request.json'
assert not request.exists(),'Already requested; do not duplicate final shutdown'
delivery=json.loads((r/'reports/original_plan_delivery_manifest.json').read_text())
assert delivery['shutdown_authorized'] and delivery['original_plan_scientific_scope_complete']
x=dict(stage='SCHEDULED_AFTER_COMPLETE_SCIENTIFIC_DELIVERY',requested_at=datetime.now(timezone.utc).isoformat(),
       delivery_manifest_sha256=expected['reports/original_plan_delivery_manifest.json'],
       reason='User-authorized final shutdown after all applicable original-plan experiments, evaluation, local SHA backup, report and agreed cleanup',
       platform_OFF_verified=False,automatic_retry=False,automatic_restart=False)
request.write_text(json.dumps(x,indent=2)+'\\n')
with (r/'reports/final_shutdown_worker.log').open('wb') as log:
 p=subprocess.Popen([sys.executable,str(r/'reports/perform_final_shutdown.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
x['worker_pid']=p.pid
request.write_text(json.dumps(x,indent=2)+'\\n')
print(json.dumps(x))
'''
        command="/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+code+'\nPY'
        _,out,err=client.exec_command(command,timeout=60)
        stdout=out.read().decode();stderr=err.read().decode()
        assert out.channel.recv_exit_status()==0,stderr[-2000:]
        receipt=json.loads(stdout)
    receipt.update(local_recorded_at=utc_now(),uploaded_delivery_SHA_verified=True,
        platform_status_channel='No accessible logged-in platform or authorized status API; browser inventory contains no platform page',platform_OFF_verified=False)
    write_json(RECEIPT,receipt)
    time.sleep(10)
    try:
        with connect() as client:
            with client.open_sftp() as sftp:
                with sftp.open(REMOTE+'/reports/final_shutdown_request.json','r') as f: receipt['remote_request']=json.loads(f.read().decode())
                with sftp.open(REMOTE+'/reports/final_shutdown_worker.log','r') as f: log=f.read()
            receipt.update(ssh_reachable_after_request=True,worker_log_bytes=len(log),worker_log_sha256=__import__('hashlib').sha256(log).hexdigest())
            if receipt['remote_request']['stage']=='SHUTDOWN_COMMAND_FAILED' or log:
                receipt['technical_error_requires_review']=True
    except Exception as error:
        receipt.update(post_request_connection_error=type(error).__name__,ssh_disconnect_is_not_OFF_proof=True)
    receipt.update(post_request_checked_at=utc_now())
    write_json(RECEIPT,receipt)
    print(json.dumps({'shutdown_dispatched':True,'remote_stage':receipt.get('remote_request',{}).get('stage'),
        'connection_result':receipt.get('post_request_connection_error','SSH reachable'),'platform_OFF_verified':False,
        'technical_error_requires_review':receipt.get('technical_error_requires_review',False)}))

if __name__=='__main__': main()
