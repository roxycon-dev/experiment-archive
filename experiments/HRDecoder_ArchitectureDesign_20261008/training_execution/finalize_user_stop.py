"""Record the user-requested stop and the verified local text archive."""
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

summary = load(HERE / 'stopped_run_archive.json')
archive = Path(summary['archive'])
outcome = load(archive / 'reports/GPU_CAMPAIGN_STOPPED.json')
verification = summary['verification']
manifest = load(archive / 'download_manifest.json')
assert outcome['status'] == 'STOPPED_BY_USER'
assert not outcome['remaining_target_processes']
assert not verification['remaining_campaign_processes']
assert verification['server_online'] and not outcome['shutdown_command_issued']
assert not manifest['checkpoint_downloaded']
assert not any(Path(item['relative_path']).suffix.lower() in {'.pt', '.pth', '.ckpt'} for item in manifest['files'])
for item in manifest['files']:
    assert (archive / item['relative_path']).stat().st_size == item['bytes']
rows = []
for line in (archive / 'reports/v3_training.log').read_text(encoding='utf-8', errors='replace').splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if isinstance(row, dict) and isinstance(row.get('completed'), int) and isinstance(row.get('losses'), dict):
        rows.append(row)
assert rows
first_log, last_log = min(r['completed'] for r in rows), max(r['completed'] for r in rows)
validation = []
for path in sorted((archive / 'training_runs/v3').glob('val_*.json')):
    raw = load(path)
    for i, (tp, fp, fn) in enumerate(raw['counts']):
        for key, numerator, denominator in [('IoU_percent', tp, tp + fp + fn),
                                             ('precision_percent', tp, tp + fp),
                                             ('recall_percent', tp, tp + fn)]:
            value = 100 * numerator / denominator if denominator else 0
            assert math.isclose(value, raw[key][i], abs_tol=1e-8)
    assert math.isclose(sum(raw['IoU_percent']) / 4, raw['mIoU_percent'], abs_tol=1e-8)
    validation.append({'step': int(path.stem.split('_')[-1]), 'mIoU': raw['mIoU_percent']})
ledger = load(HERE / 'validation_reporting.json')
assert set(x['step'] for x in validation).issubset(ledger['reported_steps']), 'Unreported validation points require user report.'
best = max(validation, key=lambda x: x['mIoU'])
last = validation[-1]
stamp = datetime.fromisoformat(outcome['stopped_at_utc']).astimezone(timezone(timedelta(hours=8)))
checked = datetime.fromisoformat(verification['checked_at_utc']).astimezone(timezone(timedelta(hours=8)))
cgroup = verification['cgroup_memory']
memory_bytes = int(cgroup.get('/sys/fs/cgroup/memory.current', cgroup.get('/sys/fs/cgroup/memory/memory.usage_in_bytes', '0')))
report = HERE / '训练停止与日志归档记录.md'
contents = f'''# v3 用户停止与日志归档记录

用户于2026-10-09要求停止训练、日志传本地、不下载检查点，并保持服务器开机等待新框架。本指令取代此前40k训练及完成后关机安排。

状态：**STOPPED_BY_USER**。停止时间：北京时间 {stamp:%Y-%m-%d %H:%M:%S}。本轮未完成40k，也未生成40k完成标志。

## 进程与服务器

已先启用关机保护并终止自动关机控制器，再停止训练、12个DataLoader worker、GPU遥测和RAM缓存owner。再次独立SSH连接于 {checked:%Y-%m-%d %H:%M:%S} 核实：原训练相关活动进程为空，GPU计算进程为空。GPU状态原始回执：

```text
{verification['gpu_status_csv'].strip()}
```

96GiB共享memfd缓存已随owner及训练worker退出释放，cgroup内存采样 {memory_bytes / 1024**3:.2f} GiB（含文件页缓存与系统服务）。磁盘数据、教师缓存和已有检查点未删除；检查点未传回本地，未等待下一次保存，也未读取模型张量。服务器SSH可达，保持开机，没有执行关机命令。

诊断操作更正：初次停止时误将另一个AI的CPU诊断 probe_normalization.py（PID10320）一同终止。用户要求不停止诊断后，已于北京时间00:21:16以原命令和原工作目录恢复为PID10977，并将已有部分输出另存 normalization_probe_interrupted_20261009T001430.json。用户随后说明无需恢复、只做本任务；恢复动作此前已经完成，此后不再操作任何诊断，也不将它纳入训练停止范围。恢复脚本明确仅CPU FP32推理、无SGD、不写检查点，日志 normalization_probe_restored_20261009.log。诊断由另一个AI继续处理，本任务不监控、停止或重启它。

心跳 hrdecoder-v3 已设为 PAUSED，提示词同步改为停止且禁止自动恢复/关机。服务器 reports/shutdown_hold.enabled 保留。旧训练不会自动恢复；待用户提供新框架后再按新指令开展工作。

## 已归档结果

最后性能记录：{outcome['performance']['completed_update']}步；最后损失日志：{last_log}步，每50步记录一个训练批次，不据此推定最后实际完成更新的精确数目。完整服务器当前训练日志包含 {first_log}–{last_log} 步的记录；更早运行阶段日志按其原文件单独保留，不补造缺失记录。

最后验证：{last['step']}步，全76张val mIoU **{last['mIoU']:.6f}%**。本轮best-val：{best['step']}步 **{best['mIoU']:.6f}%**。20k的EX/SE输出抑制异常证据已归档，原因仍未确认。本轮为提前终止结果，不能当作40k端点结果。

归档目录：`{archive}`。

完整训练日志、GPU遥测、控制器/性能优化日志、进度和performance、4k/8k/12k/16k/20k全部原始验证counts、已有诊断结果和运行源码记录均已保存。本次共 {manifest['file_count']} 个文本文件，{manifest['total_bytes'] / 1024**2:.2f} MiB。逐文件核对本地字节数与停止后稳定的远端文件大小；没有重复全量数据或缓存哈希检验。下载清单为归档目录的 download_manifest.json，独立停止后核验为 post_stop_verification.json。

TRAINING_COMPLETE.json不存在。进度和performance原始文件保持原样；reports/GPU_CAMPAIGN_OUTCOME.json和GPU_CAMPAIGN_RUNNING.json已明确标记为STOPPED_BY_USER，自动关机标志已关闭。
'''
report.write_text(contents, encoding='utf-8')
state_path = ROOT / 'cache_execution/handover_state.json'
state = load(state_path)
state.update({
    'stage': 'V3_STOPPED_BY_USER_AWAITING_NEW_FRAMEWORK',
    'formal_training_status': 'STOPPED_BY_USER',
    'latest_user_instruction': 'Stop training; archive logs locally; do not download checkpoints; do not shut down; wait for new framework. Do not interfere with another AI diagnostic; no further restore operations.',
    'resume_first': 'User stopped v3. Do not resume, restart, or shut down automatically. Await explicit new framework/instructions.',
    'automatic_shutdown': 'DISABLED_BY_USER: controller terminated; shutdown_hold enabled; heartbeat PAUSED; server remains online.',
    'heartbeat_automation_status': 'PAUSED',
    'latest_performance': outcome['performance'],
    'estimated_remaining_train_hours_excluding_validation': None,
    'current_completion_receipt': None,
    'current_outcome': {'status': 'STOPPED_BY_USER', 'stopped_at_utc': outcome['stopped_at_utc'],
                        'protocol_completed': False, 'shutdown_command_issued': False,
                        'receipt': str(archive / 'reports/GPU_CAMPAIGN_STOPPED.json')},
    'server_controller': load(archive / 'reports/GPU_CAMPAIGN_RUNNING.json'),
    'current_runtime': {'training_pid_alive': False, 'controller_pid_alive': False, 'RAM_owner_alive': False,
                        'RAM_files': 0, 'RAM_bytes': 0, 'memory_current_bytes': memory_bytes,
                        'shutdown_hold': True, 'gpu_compute_processes_at_post_stop_check': [], 'server_online': True,
                        'external_CPU_diagnostic': {'restored_pid': 10977, 'restored_at_utc': '2026-10-08T16:21:16.899041+00:00',
                                                    'current_liveness_not_rechecked': True, 'do_not_control': True}},
    'current_shutdown_observation': {'shutdown_command_issued': False, 'server_online_verified': True,
                                     'checked_at_utc': verification['checked_at_utc']},
    'last_monitor_receipt': str(archive / 'post_stop_verification.json'),
    'stop_report': str(report), 'stopped_run_archive': str(archive),
    'last_logged_loss_update': last_log, 'last_measured_update': outcome['performance']['completed_update'],
    'available_validation_steps': [x['step'] for x in validation],
    'training_complete': False, 'checkpoint_downloaded': False,
    'waiting_for_new_framework': True,
})
state['full_RAM_cache'].update({'status': 'RELEASED_BY_USER_STOP', 'all_teachers_in_RAM': False,
                               'owner_alive': False, 'resident_bytes': 0,
                               'released_at_utc': outcome['stopped_at_utc']})
state['validation_anomaly'].update({'budget': 'User stopped before 40k; last measured update ' + str(outcome['performance']['completed_update']),
                                    'followup': 'No automatic diagnostics or further validation. Existing anomaly evidence retained; await new framework.'})
save(state_path, state)
ledger.update({'run_status': 'STOPPED_BY_USER', 'monitoring_status': 'PAUSED',
               'last_checked_at': checked.isoformat(), 'stopped_at': stamp.isoformat(),
               'fixed_40000_endpoint_completed': False})
save(HERE / 'validation_reporting.json', ledger)
startup_path = HERE / '4090训练启动记录.md'
old = startup_path.read_text(encoding='utf-8')
prefix = f'''# 4090 v3 正式训练与性能调优记录

## 当前状态：用户已停止，服务器保持开机

北京时间 {stamp:%Y-%m-%d %H:%M:%S} 按用户最新指令停止本轮，最后性能记录 {outcome['performance']['completed_update']}步、最后损失日志 {last_log}步，未完成40k。训练、worker、RAM owner和自动关机控制器均已退出，停止后核查GPU计算进程为空。心跳PAUSED，关机保护保留；没有执行关机。日志/指标完整归档，检查点未下载或删除。另一个AI的CPU诊断曾被误停，已恢复，此后不再干预；详情见收尾记录。

详见 [训练停止与日志归档记录]({report.as_posix()})。后续等待用户的新框架，不自动恢复v3。

---

## 历史运行记录（下文运行和关机计划均已被上方停止指令取代）

'''
startup_path.write_text(prefix + old.split('\n', 1)[1].lstrip(), encoding='utf-8')
trend = HERE / '验证指标与趋势.md'
trend.write_text(trend.read_text(encoding='utf-8') + f'\n\n## 用户停止（{stamp:%Y-%m-%d %H:%M:%S}）\n\n本轮已按用户指令停止，最后性能记录{outcome["performance"]["completed_update"]}步。全部五个验证点已汇报并保留原始counts；40k固定端点未完成。best-val仍为8k 37.988197%，末次20k为15.050912%。服务器保持开机，监控暂停，不再自动训练或关机。\n', encoding='utf-8')
print(json.dumps({'status': state['formal_training_status'], 'report': str(report),
                  'last_logged_update': last_log, 'validation_steps': state['available_validation_steps'],
                  'archive_files': manifest['file_count'], 'archive_bytes': manifest['total_bytes'],
                  'cgroup_memory_GiB': memory_bytes / 1024**3}, ensure_ascii=True))
