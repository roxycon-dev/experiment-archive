"""Archive only CET text receipts and verify each available full-val point."""
from datetime import datetime, timezone, timedelta
import argparse
import json
import math
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'local_preparation'))
from remote import connect
REMOTE='/root/autodl-tmp/HRDecoder_v3_20261008'
LOCAL=ROOT/'cache_execution/server_receipts/cet'
CLASSES=['EX','HE','SE','MA']

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mark-reported',type=int,nargs='*',default=[])
    args=parser.parse_args()
    stamp=datetime.now(timezone(timedelta(hours=8)))
    client=connect()
    try:
        with client.open_sftp() as ftp:
            paths=['reports/cet/GPU_CAMPAIGN_RUNNING.json','reports/cet/GPU_CAMPAIGN_OUTCOME.json',
                   'reports/cet/CONTROLLER_ERROR.json','training_runs/cet/progress.json',
                   'training_runs/cet/performance.json','training_runs/cet/protocol.json',
                   'training_runs/cet/TRAINING_COMPLETE.json','training_runs/cet/GPU_SMOKE_PASS.json',
                   'training_runs/cet/validation_progress.json','cet_execution/CPU_CET_PASS.json',
                   'cet_execution/CHECKPOINT_RESUME_PASS.json']
            paths+=['training_runs/cet/val_%05d.json'%step for step in range(4000,40001,4000)]
            downloaded=[];missing=[]
            for rel in paths:
                try:
                    raw=ftp.open(REMOTE+'/'+rel,'rb').read()
                except OSError:missing.append(rel);continue
                target=LOCAL/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
                json.loads(raw);downloaded.append(rel)
            for rel in ['reports/cet/cet_training.log','reports/cet/controller.log','reports/cet/gpu_smoke.log']:
                try:raw=ftp.open(REMOTE+'/'+rel,'rb').read()
                except OSError:continue
                target=LOCAL/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            rel='reports/cet/gpu_telemetry.csv'
            with ftp.open(REMOTE+'/'+rel,'rb') as file:
                size=file.stat().st_size;file.seek(max(0,size-32768));raw=file.read()
            target=LOCAL/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        command='''import json,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
running=json.loads((root/'reports/cet/GPU_CAMPAIGN_RUNNING.json').read_text())
out={'checked_at_unix':__import__('time').time(),'server_online':True}
for name,key in [('training','pid'),('controller','controller_pid')]:
 try:
  p=Path('/proc/%d'%running[key]);cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  state=(p/'stat').read_text().split(') ',1)[1].split()[0]
  out[name+'_alive']=state!='Z';out[name+'_cmdline']=cmd
 except OSError:out[name+'_alive']=False
try:
 ready=json.loads((root/'reports/cet/RAM_CACHE_READY.json').read_text())
 out['RAM_owner_alive']=b'ram_cache_owner.py' in Path('/proc/%d/cmdline'%ready['pid']).read_bytes()
 out['RAM_bytes']=ready['bytes'];out['RAM_files']=ready['files']
except OSError:out['RAM_owner_alive']=False
out['memory_current_bytes']=int(Path('/sys/fs/cgroup/memory.current').read_text())
out['gpu_status']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv'],universal_newlines=True)
print(json.dumps(out))
'''
        _,stdout,stderr=client.exec_command('/root/miniconda3/bin/python -c '+shlex.quote(command))
        raw=stdout.read().decode();error=stderr.read().decode()
        if stdout.channel.recv_exit_status()!=0:raise RuntimeError(error)
        runtime=json.loads(raw)
    finally:client.close()
    ledger_path=HERE/'validation_reporting.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {'reported_steps':[],'check_interval_minutes':45,'validation_images':76,'class_order':CLASSES}
    historical=json.loads((ROOT/'training_execution/historical_checkpoint_comparison.json').read_text(encoding='utf-8'))
    baseline={x['step']:x['mIoU'] for x in historical.get('baseline_history',[])}
    v3=json.loads((ROOT/'training_execution/validation_series.json').read_text(encoding='utf-8'))
    old={x['step']:x['mIoU_percent'] for x in v3['records']}
    records=[]
    for path in sorted((LOCAL/'training_runs/cet').glob('val_*.json')):
        val=json.loads(path.read_text());step=int(path.stem.split('_')[-1]);classes=[]
        assert val['images']==76
        for i,name in enumerate(CLASSES):
            tp,fp,fn=val['counts'][i]
            computed={'IoU':100*tp/(tp+fp+fn) if tp+fp+fn else 0,
                'precision':100*tp/(tp+fp) if tp+fp else 0,'recall':100*tp/(tp+fn) if tp+fn else 0,
                'Dice_F1':200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
            for metric,key in [('IoU','IoU_percent'),('precision','precision_percent'),('recall','recall_percent')]:
                assert math.isclose(computed[metric],val[key][i],abs_tol=1e-8)
            classes.append({'name':name,'counts':[tp,fp,fn],**computed})
        assert math.isclose(sum(c['IoU'] for c in classes)/4,val['mIoU_percent'],abs_tol=1e-8)
        prev=records[-1] if records else None
        records.append({'step':step,'mIoU':val['mIoU_percent'],'classes':classes,'counts_verified':True,
            'delta_mIoU_pp':val['mIoU_percent']-prev['mIoU'] if prev else None,
            'delta_class_IoU_pp':[c['IoU']-p['IoU'] for c,p in zip(classes,prev['classes'])] if prev else None,
            'versus_original_HRDecoder_pp':val['mIoU_percent']-baseline[step] if step in baseline else None,
            'versus_v3_same_step_pp':val['mIoU_percent']-old[step] if step in old else None,
            'base_head_mIoU':val['base_head_metrics']['mIoU_percent']})
    assert set(args.mark_reported).issubset({r['step'] for r in records})
    if args.mark_reported:
        ledger['reported_steps']=sorted(set(ledger['reported_steps'])|set(args.mark_reported))
        ledger['last_reported_at']=stamp.isoformat()
    ledger.update(last_checked_at=stamp.isoformat(),available_steps=[r['step'] for r in records])
    write(ledger_path,ledger)
    best=max(records,key=lambda r:r['mIoU']) if records else None
    series={'records':records,'best_val':best,'unreported_steps':[r['step'] for r in records if r['step'] not in ledger['reported_steps']]}
    write(HERE/'validation_series.json',series)
    lines=['# CET 验证指标与趋势','',
        '全76张val；EX/HE/SE/MA；阈值严格>.5。原始counts与指标已逐项核对。40k固定端点为主结果，best-val为次。',
        '训练缓存仅用于训练期目标；验证未输入教师或GT，也未对T*构造oracle验证精度。','',
        '| 步骤 | mIoU% | ΔmIoU pp | EX IoU% | HE IoU% | SE IoU% | MA IoU% | 相对原HRDecoder pp | 相对v3 pp |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return '—' if x is None else f'{x:.3f}'
    for record in records:
        values=[record['mIoU'],record['delta_mIoU_pp']]+[c['IoU'] for c in record['classes']]+[record['versus_original_HRDecoder_pp'],record['versus_v3_same_step_pp']]
        lines.append('| '+str(record['step'])+' | '+' | '.join(fmt(x) for x in values)+' |')
    if not records:lines+=['','尚无已完成的4k验证点，不作精度或趋势结论。']
    if best:lines+=['',f"best-val：{best['step']}步，mIoU {best['mIoU']:.3f}%。"]
    for record in records:
        lines+=['',f"## {record['step']}步",'',
            '| 类别 | IoU% | precision% | recall% | Dice/F1% | ΔIoU pp | TP / FP / FN |',
            '|---|---:|---:|---:|---:|---:|---|']
        for i,c in enumerate(record['classes']):
            delta=record['delta_class_IoU_pp'][i] if record['delta_class_IoU_pp'] else None
            lines.append('| '+c['name']+' | '+' | '.join(fmt(c[k]) for k in ('IoU','precision','recall','Dice_F1'))+' | '+fmt(delta)+' | '+str(c['counts'])+' |')
        lines+=['',f"主判断头自身mIoU {record['base_head_mIoU']:.3f}%，用于观察纠错前后行为，不能作为机制因果对照。"]
    lines+=['','precision在TP+FP=0时记录0是脚本约定，其数学分母为0；请结合原始counts。Dice/F1由counts另算，不能当作训练损失。',
        '历史原HRDecoder、v3同节点仅是已核对数据口径的参照；容量和损失不同，无新配对对照不能归因。锁定融合val=45.321831%，不与test混比。',
        '首点仅建立基线；后续需同时观察误检与漏检，至少三个点才讨论增益趋缓，不据单点判定过拟合。']
    (HERE/'验证指标与趋势.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    snapshot={'checked_at':stamp.isoformat(),'runtime':runtime,'downloaded':downloaded,'missing':missing,
        'progress':json.loads((LOCAL/'training_runs/cet/progress.json').read_text()) if (LOCAL/'training_runs/cet/progress.json').exists() else None,
        'performance':json.loads((LOCAL/'training_runs/cet/performance.json').read_text()) if (LOCAL/'training_runs/cet/performance.json').exists() else None,
        'outcome':json.loads((LOCAL/'reports/cet/GPU_CAMPAIGN_OUTCOME.json').read_text()) if (LOCAL/'reports/cet/GPU_CAMPAIGN_OUTCOME.json').exists() else None,
        'new_validation_steps':series['unreported_steps'],'server_shutdown_authorized':False}
    write(LOCAL/'latest_snapshot.json',snapshot)
    write(LOCAL/'monitoring'/('snapshot_'+stamp.strftime('%Y%m%dT%H%M%S')+'.json'),snapshot)
    handover_path=ROOT/'cache_execution/handover_state.json'
    handover=json.loads(handover_path.read_text(encoding='utf-8'))
    if handover.get('stage')=='CET_RUNNING' and handover.get('active_run')=='cet':
        handover.update(current_runtime=runtime,latest_performance=snapshot['performance'],
            last_monitor_receipt=str(LOCAL/'latest_snapshot.json'),
            available_validation_steps=ledger['available_steps'],reported_validation_steps=ledger['reported_steps'],
            last_monitor_time_unix=runtime['checked_at_unix'],
            current_shutdown_observation={'shutdown_command_issued':False,'server_online_verified':True,'checked_at':stamp.isoformat()})
        if snapshot['performance']:
            handover['estimated_remaining_train_hours_excluding_validation']=snapshot['performance']['estimated_remaining_train_hours_excluding_validation']
            handover['last_measured_update']=snapshot['performance']['completed_update']
        if snapshot['progress']:
            handover['last_logged_loss_update']=snapshot['progress']['completed']
        if snapshot['outcome']:
            running=json.loads((LOCAL/'reports/cet/GPU_CAMPAIGN_RUNNING.json').read_text())
            if snapshot['outcome'].get('start_unix')==running.get('start_unix'):
                handover.update(current_outcome=snapshot['outcome'],formal_training_status=snapshot['outcome']['status'])
        write(handover_path,handover)
    print(json.dumps(snapshot,ensure_ascii=True))

if __name__=='__main__':main()
