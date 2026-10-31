"""Keep timestamped small monitoring evidence; normal progress requires no notice."""
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
receipts=ROOT/'cache_execution/server_receipts'
snapshot=json.loads((receipts/'GPU_CURRENT_SNAPSHOT.json').read_text(encoding='utf-8'))
observed=snapshot['receipts']
protocol=observed['training_runs/v3/protocol.json']['identity']['protocol']
assert (protocol['design_version'],protocol['canvas'],protocol['global_batch'],protocol['updates'])==(3,2048,2,40000)
performance=observed['training_runs/v3/performance.json']
saved=observed['training_runs/v3/progress.json']
assert performance['completed_update']>=saved['completed']
utc=datetime.fromtimestamp(snapshot['time_unix'],timezone.utc)
destination=receipts/'monitoring'/('heartbeat_'+utc.strftime('%Y%m%dT%H%M%SZ')+'.json')
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state_path=ROOT/'cache_execution/handover_state.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
state['estimated_remaining_train_hours_excluding_validation']=(40000-performance['completed_update'])*performance['mean_batch_seconds']/3600
state['last_monitor_receipt']=str(destination)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
local=utc.astimezone(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
with (HERE/'4090训练启动记录.md').open('a',encoding='utf-8') as stream:
    stream.write(f'\n监控 {local}：运行状态{state["formal_training_status"]}，training_PID_alive={snapshot["runtime"]["training_pid_alive"]}；最新测到 {performance["completed_update"]}/40,000，持久检查点 {saved["completed"]}；最近100步 {performance["mean_batch_seconds"]:.3f}秒/步。RAM_owner_alive={snapshot["runtime"]["RAM_owner_alive"]}、RAM文件数{snapshot["runtime"]["RAM_files"]}；shutdown_hold={snapshot["runtime"]["shutdown_hold"]}；完成标志存在={observed["training_runs/v3/TRAINING_COMPLETE.json"] is not None}，outcome存在={observed["reports/GPU_CAMPAIGN_OUTCOME.json"] is not None}。小型记录已归档。\n')
print(json.dumps({'archived_monitor_receipt':str(destination),'status':state['formal_training_status'],
    'completed_update':performance['completed_update'],'saved_update':saved['completed']},ensure_ascii=False))
