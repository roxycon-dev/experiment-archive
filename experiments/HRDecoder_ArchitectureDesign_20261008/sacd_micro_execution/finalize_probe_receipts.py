"""Archive the completed paired diagnostic and observe only its registered process IDs."""
import json
from pathlib import Path
import shlex
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
client=connect()
try:
    with client.open_sftp() as ftp:
        names=['reports/sacd_micro/MICRO_GATE.json','reports/sacd_micro/GATE_RECEIPT_REPAIR.json',
               'reports/sacd_micro/CONTROLLER_ERROR_before_reference_repair.json',
               'reports/sacd_micro/STAGING_STATUS.json','sacd_execution/val_label_reference.json']
        for name in names:
            raw=ftp.open(REMOTE+'/'+name,'rb').read()
            path=ROOT/'cache_execution/server_receipts/sacd_micro'/name
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    code='''import json,time,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
out={'server_online':True,'checked_at_unix':time.time(),'shutdown_command_issued':False,
 'process_checks_only_own_registered_PIDs':True,'training_updates_added':0}
for name,pid in [('micro_controller',31202),('control',31205),('feature_isolated',31648)]:
 p=Path('/proc/%d'%pid)
 try:
  state=(p/'stat').read_text().split(') ',1)[1].split()[0]
  out[name]={'pid':pid,'alive':state!='Z','state':state,'cmdline':(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()}
 except OSError:out[name]={'pid':pid,'alive':False}
ready=json.loads((root/'reports/sacd/RAM_CACHE_READY.json').read_text())
out['own_RAM_owner']={'pid':ready['pid'],'bytes':ready['bytes'],'retained_pending_user_choice':True}
out['GPU_status']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv'],universal_newlines=True)
out['no_fresh_run_registered']=not (root/'reports/sacd_isolated/SCREEN_STARTED.json').exists()
print(json.dumps(out))
'''
    _,out,err=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
    raw,error=out.read().decode(),err.read().decode()
    if out.channel.recv_exit_status():raise RuntimeError(error)
    status=json.loads(raw)
    assert all(not status[name]['alive'] for name in ('micro_controller','control','feature_isolated'))
    assert status['no_fresh_run_registered']
    (HERE/'PROBE_TERMINAL_OBSERVATION.json').write_text(json.dumps(status,indent=2)+'\n')
    archive=ROOT/'cache_execution/server_receipts/sacd_micro/PROBE_TERMINAL_OBSERVATION.json'
    archive.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))
finally:client.close()
