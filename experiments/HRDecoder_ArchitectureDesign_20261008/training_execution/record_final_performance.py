"""Write the measured final runtime and honest tuning limits to local handover."""
import csv
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DEST=ROOT/'cache_execution/server_receipts'
snapshot=json.loads((DEST/'GPU_CURRENT_SNAPSHOT.json').read_text(encoding='utf-8'))
receipts=snapshot['receipts']; runtime=snapshot['runtime']
perf=receipts['training_runs/v3/performance.json']
running=receipts['reports/GPU_CAMPAIGN_RUNNING.json']
layout=receipts['reports/GPU_LAYOUT_BENCHMARK.json']
cpu=receipts['reports/RAM_PIPELINE_BENCHMARK_PASS.json']
ram=receipts['reports/RAM_MIXED_READY_SUMMARY.json']
assert runtime['training_pid_alive'] and runtime['RAM_owner_alive']
assert runtime['RAM_files']==1362 and not runtime['shutdown_hold']
now=datetime.now(timezone(timedelta(hours=8)))
remaining=(40000-perf['completed_update'])*perf['mean_batch_seconds']/3600
rows=list(csv.reader((DEST/'reports/gpu_telemetry.csv').read_text().splitlines()))[-60:]
samples=[r for r in rows if len(r)==9]
util=statistics.mean(float(r[1]) for r in samples)
power=statistics.mean(float(r[5]) for r in samples)
report=f'''# 4090 v3 正式训练与性能调优记录

更新：{now.isoformat(timespec='seconds')}。服务器 connect.nmb1.seetacloud.com:11347；实验目录 `/root/autodl-tmp/HRDecoder_v3_20261008`。实际硬件 RTX4090 24GB 显存、120GiB 系统 RAM、16 CPU 配额。

## 当前运行

唯一 v3 主模型，2048、batch2、40,000 次 SGD 更新，原增强、损失、私有梯度路由、初始化和预算保持。正式进程 PID {running['pid']}、控制器 PID {running['controller_pid']}，显式 `--resume` 从模型、优化器和 RNG 检查点恢复。未运行额外训练对照，未再次扫描全量数据或缓存哈希。

最新测到 {perf['completed_update']}/40,000，持久 last.pt 保存于 3,000 步，文件 {runtime['last_checkpoint_bytes']:,} bytes。每1k保存模型/optimizer/RNG，每4k对全部76张 val 验证，40k 固定端点为主结果。当前还未到首个4k验证点，尚无最终精度或训练完成结论。

## 全部教师缓存驻留 RAM

454张 × A/U/S = 1,362 份全部驻留共享 memfd，占 {ram['bytes']/1024**3:.3f}GiB。A/U 保持解压后的 NPY，S 中较大图解压，其余保存原无损 NPZ；共 {ram['format_counts']['npy']} 个 NPY、{ram['format_counts']['npz']} 个 NPZ。保留原 FP32 值，不量化，不改坐标或目标。磁盘原缓存继续保留。

先前105GiB全解压方案触发内存保护，11张图回退磁盘；已替换为96GiB混合格式方案以保持全量 RAM 驻留并留出运行空间。当前总 cgroup 内存 {runtime['memory_current_bytes']/1024**3:.2f}/120GiB，RAM owner PID {ram['pid']} 存活，1,362 份映射全部存在。内存占用是有效缓存与训练工作集。

## 实测性能与取舍

新缓存下同一曝光序列 CPU 读取：8 worker 平均 {cpu['CPU_benchmarks'][0]['mean_CPU_batch_seconds']:.3f} 秒/批；12 worker {cpu['CPU_benchmarks'][1]['mean_CPU_batch_seconds']:.3f} 秒/批，因此采用12 worker、各 OpenCV/PyTorch 1线程、prefetch1、pinned memory、异步 CUDA 拷贝流和事件等待。实际2048增强后的 image/label/valid/A/U/S 抽查逐位一致，CPU→GPU传输的所有这些张量也已逐位核对。

同一实际2048/batch2的 GPU 前向与路由反向按 NCHW→channels_last→NCHW 顺序比较，所有测试优化器 step 禁用、未写正式检查点。NCHW 中位数分别 {layout['measurements'][0]['median_seconds']:.3f}/{layout['measurements'][2]['median_seconds']:.3f} 秒；channels_last {layout['measurements'][1]['median_seconds']:.3f} 秒，慢约 {-layout['speedup_percent']:.2f}%。其梯度相对 L2 差异 {layout['global_gradient_relative_L2']:.4f} 超过预设0.05容限，损失最大绝对差 {layout['max_loss_absolute_difference']:.6f}。未采用此候选，正式模型保持原 NCHW 布局。

正式训练最近 {perf['window_batches']} 步稳态窗口平均 {perf['mean_batch_seconds']:.3f} 秒/步，计算 {perf['mean_step_compute_seconds']:.3f} 秒、数据等待 {perf['mean_data_wait_seconds']*1000:.3f} 毫秒。读取等待基本消除，目前主要耗时为模型计算。这是已比较配置的实测结论，不宣称穷尽所有实现后的全局最快。CUDA峰值 allocated {perf['peak_allocated_bytes']/1024**3:.2f}GiB、reserved {perf['peak_reserved_bytes']/1024**3:.2f}GiB；最近 {len(samples)} 个两秒采样 GPU 利用率均值 {util:.1f}%、功耗均值 {power:.1f}W。功耗和显存填满并不作为优化指标。

目前剩余纯训练外推 {remaining:.2f} 小时，另需验证与保存，暂按 **剩余11–12小时**、北京时间 **10月9日约6–7点**安排。后续根据实际4k验证耗时修正，不是完成时间保证。

## 检查点与关机

首次4-worker启动的无检查点前缀最后观测120步，已丢弃并以原 seed42重新开始。第一次 RAM 调优从已保存1,000步恢复，最后观测1,280步，未保存后缀重跑。第二次切换精确等待3,000步保存，最后测到也为3,000步，本次未丢失已完成更新。整个主结果仍为累计40k更新。

shutdown_hold 已移除，控制器完成后会核对40k完成标志，实际加载持久 last.pt，核对completed=40000、身份、optimizer/RNG及最终val_40000结果，写 outcome、sync 后执行 `/bin/bash /usr/bin/shutdown`。致命错误先保存证据并留10分钟修复窗口。心跳 hrdecoder-v3 保持 ACTIVE，正常进度不重复通知。

**当前状态 RUNNING；尚未关机。** 只有实际执行关机并观察 SSH 断开后才记录已验证的关机连接状态。
'''
(HERE/'4090训练启动记录.md').write_text(report,encoding='utf-8')
(HERE/'RAM与吞吐调优记录.md').write_text(report,encoding='utf-8')
state_path=ROOT/'cache_execution/handover_state.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
state.update(performance_tuning_in_progress=False,stage='V3_FULL_RAM_TUNING_COMPLETE_RUNNING',
    latest_performance=perf,latest_performance_report=str(HERE/'RAM与吞吐调优记录.md'),
    estimated_remaining_train_hours_excluding_validation=remaining,
    historical_RAM_tuning='First resume1000, last observed1280; second switch saved and observed3000. Current mixed RAM cache1362/1362 files.',
    current_shutdown_observation=None)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'RUNNING','completed':perf['completed_update'],
    'mean_batch_seconds':perf['mean_batch_seconds'],'data_wait_ms':perf['mean_data_wait_seconds']*1000,
    'memory_GiB':runtime['memory_current_bytes']/1024**3,'remaining_pure_train_hours':remaining,
    'GPU_utilization_mean':util,'GPU_power_mean':power,'report':str(HERE/'RAM与吞吐调优记录.md')},ensure_ascii=False))
