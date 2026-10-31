"""Small paired-diagnostic receipts and local ledger; never controls processes."""
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import shlex
import stat
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
sys.path.insert(0,str(ROOT/'local_preparation'))
sys.path.insert(0,str(ROOT/'sacd_micro_execution'))
from remote import connect
from gate import verify


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--mark-reported',nargs='*',type=int,default=[])
    args=parser.parse_args()
    now=datetime.now(timezone(timedelta(hours=8)))
    ledger_path=HERE/'validation_reporting.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {'reported_probe_steps':[]}
    if args.mark_reported:
        series=json.loads((HERE/'paired_series.json').read_text(encoding='utf-8'))
        assert set(args.mark_reported)<=set(series['paired_steps'])
        ledger['reported_probe_steps']=sorted(set(ledger['reported_probe_steps'])|set(args.mark_reported))
        ledger['last_reported_at']=now.isoformat();write(ledger_path,ledger)
        series['unreported_pair_steps']=[n for n in series['paired_steps'] if n not in ledger['reported_probe_steps']]
        write(HERE/'paired_series.json',series)
        print(json.dumps({'status':'MARKED_LOCALLY','SSH_poll':False}));return
    last=ledger.get('last_routine_checked_at')
    if last and not args.force and now<datetime.fromisoformat(last)+timedelta(minutes=45)-timedelta(seconds=60):
        print(json.dumps({'status':'WAITING_FOR_MONITOR_WINDOW','SSH_poll':False}));return
    receipt_root=ROOT/'cache_execution/server_receipts/residual_normalization'
    client=connect();archived=[]
    try:
        with client.open_sftp() as ftp:
            def archive(directory):
                try:entries=ftp.listdir_attr(REMOTE+'/'+directory)
                except OSError:return
                for entry in entries:
                    rel=directory+'/'+entry.filename
                    if stat.S_ISDIR(entry.st_mode):
                        if directory=='training_runs/residual_normalization' and entry.filename in ('control','gt_stratified'):
                            archive(rel)
                        continue
                    if not entry.filename.endswith(('.json','.log')):continue
                    assert entry.st_size<15*1024**2
                    with ftp.open(REMOTE+'/'+rel,'rb') as file:raw=file.read()
                    if entry.filename.endswith('.json'):json.loads(raw)
                    target=receipt_root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
                    archived.append(rel)
            for directory in ('reports/residual_normalization','training_runs/residual_normalization','normalization_execution'):
                archive(directory)
        code="""import json,time,subprocess
from pathlib import Path
r=Path('/root/autodl-tmp/HRDecoder_v3_20261008');p=r/'reports/residual_normalization'
out={'server_online':True,'checked_at_unix':time.time(),'shutdown_command_issued':False,'external_diagnostics_controlled':False}
for n in ('RUNNING.json','OUTCOME.json','ERROR.json','DISPATCHED.json'):
 f=p/n
 if f.exists():out[n]=json.loads(f.read_text())
run=out.get('RUNNING.json',out.get('DISPATCHED.json',{}));out['own_process_checks']={}
for key in ('pid','controller_pid'):
 pid=run.get(key)
 if not pid:continue
 path=Path('/proc/%d/cmdline'%pid);cmd=path.read_bytes().replace(b'\\0',b' ').decode() if path.exists() else ''
 out['own_process_checks'][key]={'pid':pid,'alive':bool(cmd),'cmdline':cmd,'matches_task':bool(cmd) and (str(r/'normalization_execution')==str(Path('/proc/%d/cwd'%pid).resolve()))}
ready=json.loads((r/'reports/sacd/RAM_CACHE_READY.json').read_text());path=Path('/proc/%d/cmdline'%ready['pid'])
cmd=path.read_bytes() if path.exists() else b''
out['own_RAM']={'pid':ready['pid'],'bytes':ready['bytes'],'alive':bool(cmd),'matches_own_RAM':b'sacd_execution/ram_cache_owner.py' in cmd}
out['GPU_status']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv'],universal_newlines=True)
print(json.dumps(out))
"""
        _,output,error=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
        raw,err=output.read().decode(),error.read().decode();assert output.channel.recv_exit_status()==0,err
        runtime=json.loads(raw)
    finally:client.close()
    write(HERE/'latest_snapshot.json',{'checked_at':now.isoformat(),'runtime':runtime,'archived':archived})
    reference=json.loads((ROOT/'sacd_execution/val_label_reference.json').read_text())
    arms={}
    for arm in ('control','gt_stratified'):
        records=[]
        for path in sorted((receipt_root/'training_runs/residual_normalization'/arm).glob('val_probe_*.json')):
            val=verify(json.loads(path.read_text()),reference['ground_truth_positive_pixels'])
            verify(val['base_head_metrics'],reference['ground_truth_positive_pixels'])
            assert val['parent_completed']==8000 and val['arm']==arm and val['probe_updates'] in (0,250,500)
            records.append(val)
        arms[arm]=records
    paired_steps=sorted(set(v['probe_updates'] for v in arms['control'])&set(v['probe_updates'] for v in arms['gt_stratified']))
    comparisons=[]
    for step in paired_steps:
        control=next(v for v in arms['control'] if v['probe_updates']==step)
        variant=next(v for v in arms['gt_stratified'] if v['probe_updates']==step)
        comparisons.append({'probe_updates':step,'delta_mIoU_variant_minus_control_pp':variant['mIoU_percent']-control['mIoU_percent'],
           'delta_class_IoU_pp':[a-b for a,b in zip(variant['IoU_percent'],control['IoU_percent'])],
           'control_final_minus_base_pp':control['mIoU_percent']-control['base_head_metrics']['mIoU_percent'],
           'variant_final_minus_base_pp':variant['mIoU_percent']-variant['base_head_metrics']['mIoU_percent']})
    series={'kind':'paired warm8k+500 diagnostic; not fresh40k model','checked_at':now.isoformat(),'arms':arms,
            'paired_steps':paired_steps,'comparisons':comparisons,'counts_verified':True,
            'unreported_pair_steps':[n for n in paired_steps if n not in ledger['reported_probe_steps']],
            'automatic_extension':False,'automatic_fresh_training':False,'test_for_selection':False}
    write(HERE/'paired_series.json',series)
    ledger['last_checked_at']=now.isoformat()
    ledger['last_manual_checked_at' if args.force else 'last_routine_checked_at']=now.isoformat()
    write(ledger_path,ledger)
    state_path=ROOT/'iterative_execution/iteration_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    state.update(active_run='normalization',status=runtime.get('RUNNING.json',{}).get('status','DEPLOYED'),
                 active_execution_directory=str(HERE),active_monitor=str(HERE/'monitor_normalization.py'),
                 latest_runtime_observation=runtime,last_real_monitor_at=now.isoformat(),
                 current_task_training_active=runtime.get('own_process_checks',{}).get('pid',{}).get('alive',False))
    write(state_path,state)
    print(json.dumps({'status':'ARCHIVED','unreported_pair_steps':series['unreported_pair_steps'],'runtime':runtime}))


if __name__=='__main__':main()
