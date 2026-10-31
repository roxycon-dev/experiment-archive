"""45-minute receipts, independent count checks and local reporting ledger."""
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import shlex
import stat
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
sys.path.insert(0,str(ROOT/'local_preparation'))
sys.path.insert(0,str(ROOT/'sacd_micro_execution'))
from remote import connect
from gate import verify

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--force',action='store_true');p.add_argument('--local-only',action='store_true')
    p.add_argument('--mark-reported',type=int,nargs='*',default=[]);a=p.parse_args()
    now=datetime.now(timezone(timedelta(hours=8)))
    ledger_path=HERE/'validation_reporting.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {'reported_steps':[]}
    if a.mark_reported:
        series=json.loads((HERE/'validation_series.json').read_text(encoding='utf-8'))
        assert set(a.mark_reported)<={r['step'] for r in series['records']}
        ledger['reported_steps']=sorted(set(ledger['reported_steps'])|set(a.mark_reported))
        ledger['last_reported_at']=now.isoformat();write(ledger_path,ledger)
        series['unreported_steps']=[r['step'] for r in series['records'] if r['step'] not in ledger['reported_steps']]
        write(HERE/'validation_series.json',series)
        print(json.dumps({'status':'MARKED_LOCALLY','SSH_poll':False}));return
    last=ledger.get('last_routine_checked_at')
    if last and not a.force and not a.local_only:
        due=datetime.fromisoformat(last)+timedelta(minutes=45)
        if now<due-timedelta(seconds=60):
            print(json.dumps({'status':'WAITING_FOR_MONITOR_WINDOW','next_check_at':due.isoformat(),'SSH_poll':False}));return
    receipt_root=ROOT/'cache_execution/server_receipts/detail'
    if a.local_only:
        cached=json.loads((HERE/'latest_snapshot.json').read_text(encoding='utf-8'))
        now=datetime.fromisoformat(cached['checked_at'])
        runtime=cached['runtime'];downloaded=cached['downloaded']
    else:
        client=connect();downloaded=[]
        try:
            with client.open_sftp() as ftp:
                def archive(directory):
                    try:entries=ftp.listdir_attr(REMOTE+'/'+directory)
                    except OSError:return
                    for entry in entries:
                        rel=directory+'/'+entry.filename
                        if stat.S_ISDIR(entry.st_mode):continue
                        if not entry.filename.endswith(('.json','.log')):continue
                        assert entry.st_size<15*1024**2,'Unexpectedly large receipt'
                        with ftp.open(REMOTE+'/'+rel,'rb') as f:raw=f.read()
                        if entry.filename.endswith('.json'):json.loads(raw)
                        target=receipt_root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
                        downloaded.append(rel)
                for d in ('reports/detail','training_runs/detail','detail_execution'):archive(d)
            code='''import json,time,subprocess,os
    from pathlib import Path
    r=Path('/root/autodl-tmp/HRDecoder_v3_20261008');p=r/'reports/detail'
    out={'server_online':True,'checked_at_unix':time.time(),'shutdown_command_issued':False,'external_processes_controlled':False}
    for name in ('GPU_CAMPAIGN_RUNNING.json','GPU_CAMPAIGN_OUTCOME.json','CONTROLLER_ERROR.json','LAUNCH_ERROR.json','DEADLINE_GATE.json','DEADLINE_STOP.json','DEADLINE_EXTENSION_ACTIVATED.json'):
     f=p/name
     if f.exists():out[name]=json.loads(f.read_text())
    run=out.get('GPU_CAMPAIGN_RUNNING.json',{})
    out['own_process_checks']={}
    for k in ('pid','controller_pid'):
     pid=run.get(k)
     if pid:
      path=Path('/proc/%d/cmdline'%pid)
      cmd=path.read_bytes().replace(b'\\0',b' ').decode() if path.exists() else ''
      out['own_process_checks'][k]={'pid':pid,'alive':bool(cmd),'cmdline':cmd,'matches_current_task':('detail_execution' in cmd or (any(n in cmd for n in ('train.py','controller.py')) and str(r/'detail_execution')==str(Path('/proc/%d/cwd'%pid).resolve()))) if cmd else False}
    session=run.get('controller_pid');out['own_session_id']=session;out['own_session_live_processes']=[]
    if session:
     for path in Path('/proc').iterdir():
      if path.name.isdigit():
       try:
        pid=int(path.name)
        if os.getsid(pid)==session and (path/'stat').read_text().split(')')[1].strip().split()[0]!='Z':
         out['own_session_live_processes'].append({'pid':pid,'cmdline':(path/'cmdline').read_bytes().replace(b'\\0',b' ').decode()})
       except OSError:pass
    ready=json.loads((r/'reports/sacd/RAM_CACHE_READY.json').read_text())
    path=Path('/proc/%d/cmdline'%ready['pid']);cmd=path.read_bytes() if path.exists() else b''
    out['own_RAM']={'pid':ready['pid'],'bytes':ready['bytes'],'alive':bool(cmd),'matches_own_RAM':b'sacd_execution/ram_cache_owner.py' in cmd}
    out['GPU_status']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv'],universal_newlines=True)
    print(json.dumps(out))
    '''
            # The local-only branch added a Python indentation level around
            # this multiline literal. Remove that level from its payload.
            lines=code.splitlines()
            code=lines[0]+'\n'+'\n'.join(line[4:] if line.startswith('    ') else line for line in lines[1:])
            _,o,e=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(code))
            raw,error=o.read().decode(),e.read().decode();assert o.channel.recv_exit_status()==0,error
            runtime=json.loads(raw)
        finally:client.close()
    write(HERE/'latest_snapshot.json',{'checked_at':now.isoformat(),'runtime':runtime,'downloaded':downloaded})
    reference=json.loads((ROOT/'sacd_execution/val_label_reference.json').read_text())
    gt=reference['ground_truth_positive_pixels']
    baseline={r['step']:r['mIoU'] for r in json.loads((ROOT/'training_execution/historical_checkpoint_comparison.json').read_text(encoding='utf-8'))['baseline_history']}
    cet={r['step']:r['mIoU'] for r in json.loads((ROOT/'cet_execution/validation_series.json').read_text(encoding='utf-8'))['records']}
    sacd={r['step']:r['mIoU'] for r in json.loads((ROOT/'sacd_execution/validation_series.json').read_text(encoding='utf-8'))['records']}
    residual={r['step']:r['mIoU'] for r in json.loads((ROOT/'residual_execution/validation_series.json').read_text(encoding='utf-8'))['records']}
    adapter={r['step']:r['mIoU'] for r in json.loads((ROOT/'adapter_execution/validation_series.json').read_text(encoding='utf-8'))['records']}
    records=[]
    for path in sorted((receipt_root/'training_runs/detail').glob('val_*.json')):
        val=verify(json.loads(path.read_text()),gt);base=verify(val['base_head_metrics'],gt)
        step=val['completed'];previous=records[-1] if records else None
        classes=[{'name':name,'counts':val['counts'][i],'IoU':val['IoU_percent'][i],
                  'precision':val['precision_percent'][i],'recall':val['recall_percent'][i],
                  'precision_undefined':sum(val['counts'][i][:2])==0} for i,name in enumerate(reference['class_order'])]
        records.append({'step':step,'mIoU':val['mIoU_percent'],'classes':classes,'counts_verified':True,
          'base_head_mIoU':base['mIoU_percent'],'base_head_metrics':base,
          'delta_mIoU_pp':val['mIoU_percent']-previous['mIoU'] if previous else None,
          'delta_class_IoU_pp':[x['IoU']-y['IoU'] for x,y in zip(classes,previous['classes'])] if previous else None,
          'versus_original_HRDecoder_pp':val['mIoU_percent']-baseline[step] if step in baseline else None,
          'versus_CET_same_step_pp':val['mIoU_percent']-cet[step] if step in cet else None,
          'versus_original_SACD_same_step_pp':val['mIoU_percent']-sacd[step] if step in sacd else None,
          'versus_residual_same_step_pp':val['mIoU_percent']-residual[step] if step in residual else None,
          'versus_adapter_same_step_pp':val['mIoU_percent']-adapter[step] if step in adapter else None})
    series={'checked_at':now.isoformat(),'run':'detail','records':records,'counts_verified':True,
      'best_val':max(records,key=lambda r:r['mIoU']) if records else None,
      'unreported_steps':[r['step'] for r in records if r['step'] not in ledger['reported_steps']],
      'primary_endpoint':40000,'test_for_selection':False}
    write(HERE/'validation_series.json',series)
    lines=['# 半分辨率细节纠错候选验证指标','', '完整76图；所有IoU/P/R单位为%。40k固定端点为主结果，best-val为次。', '',
           '|step|mIoU|EX IoU/P/R|HE IoU/P/R|SE IoU/P/R|MA IoU/P/R|基础分支mIoU|','|---:|---:|---|---|---|---|---:|']
    for row in records:
        cells=['%.6f/%.6f/%.6f'%(x['IoU'],x['precision'],x['recall']) for x in row['classes']]
        lines.append('|%d|%.6f|%s|%.6f|'%(row['step'],row['mIoU'],'|'.join(cells),row['base_head_mIoU']))
    (HERE/'验证指标与趋势.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ledger['last_checked_at']=now.isoformat()
    ledger['last_manual_checked_at' if a.force else 'last_routine_checked_at']=now.isoformat()
    write(ledger_path,ledger)
    state_path=ROOT/'iterative_execution/iteration_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    state.update(active_run='detail',status=runtime.get('GPU_CAMPAIGN_RUNNING.json',{}).get('status','DEPLOYED'),
                 active_execution_directory=str(HERE),active_monitor=str(HERE/'monitor_detail.py'),
                 latest_runtime_observation=runtime,last_real_monitor_at=now.isoformat(),
                 next_scheduled_monitor=(now+timedelta(minutes=45)).isoformat(),
                 current_task_training_active=runtime.get('own_process_checks',{}).get('pid',{}).get('alive',False))
    if runtime.get('DEADLINE_EXTENSION_ACTIVATED.json'):
        state.update(conditional_extension_activated=True,deadline_rule_active='same screened fresh run until completion')
    write(state_path,state)
    print(json.dumps({'status':'ARCHIVED','unreported_steps':series['unreported_steps'],'runtime':runtime}))

if __name__=='__main__':main()
