"""Archive small training receipts, logs and telemetry; never scan bulk cache tensors."""
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import statistics
import sys
import time
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"local_preparation"))
from remote import connect, HOST, PORT
ROOT = "/root/autodl-tmp/HRDecoder_v3_20261008"
DEST = HERE.parent/"cache_execution/server_receipts"
client = connect()
with client.open_sftp() as ftp:
    names = ["reports/GPU_CAMPAIGN_RUNNING.json", "reports/GPU_CAMPAIGN_OUTCOME.json",
        "reports/TRAINING_SHUTDOWN_COMMAND_ISSUED.json", "reports/STARTUP_PIPELINE_REPLACEMENT.json",
        "reports/STARTUP_WORKERS4_PERFORMANCE.json", "reports/v3_training.log", "reports/gpu_telemetry.csv",
        "training_runs/v3/GPU_SMOKE_PASS.json", "training_runs/v3/performance.json",
        "training_runs/v3/protocol.json", "training_runs/v3/progress.json", "training_runs/v3/TRAINING_COMPLETE.json"]
    names += ["training_runs/v3/val_%05d.json"%step for step in range(4000,40001,4000)]
    for name in names:
        destination = DEST/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        try: ftp.get(ROOT+"/"+name, str(destination))
        except FileNotFoundError: pass
client.close()
performance = json.loads((DEST/"training_runs/v3/performance.json").read_text())
smoke = json.loads((DEST/"training_runs/v3/GPU_SMOKE_PASS.json").read_text())
running = json.loads((DEST/"reports/GPU_CAMPAIGN_RUNNING.json").read_text())
csv_text = (DEST/"reports/gpu_telemetry.csv").read_text()
rows = list(csv.reader(io.StringIO(csv_text)))[-60:]
util = [float(row[1]) for row in rows if len(row)==9]
power = [float(row[5]) for row in rows if len(row)==9]
snapshot = {"time_unix": time.time(), "performance": performance,
    "telemetry_window_samples": len(util), "telemetry_interval_seconds": 2,
    "mean_GPU_utilization_percent": statistics.mean(util),
    "median_GPU_utilization_percent": statistics.median(util),
    "mean_power_W": statistics.mean(power), "data_hash_rescan": False,
    "heartbeat_automation_id": "hrdecoder-v3", "server_shutdown_after_training_enabled": True}
(DEST/"GPU_LAUNCH_SNAPSHOT.json").write_text(json.dumps(snapshot,indent=2)+"\n")
state_path = HERE.parent/"cache_execution/handover_state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
state.update({"stage": "GPU_SMOKE_PASS_FORMAL_V3_RUNNING", "server": {
    "host": HOST, "port": PORT, "remote_root": ROOT, "password_stored": False},
    "latest_user_instruction": "Start on new 4090 endpoint; no repeated full hashes. Report ETA, shut down after completion.",
    "GPU_smoke": smoke, "formal_training_started": True, "formal_training_status": "RUNNING",
    "server_controller": running, "latest_performance": performance,
    "source_update_after_CPU_preparation": "CUDA BF16 bilinear interpolation uses FP32 and casts back; actual 2048 batch2 GPU smoke passed.",
    "startup_pipeline_replacement": "Initial 4-worker pre-checkpoint run discarded after last observed 120 batches; final retained trajectory restarted seed42 with 12 workers, OpenCV1 thread.",
    "automatic_shutdown": "Server controller verifies 40k completion and saved last.pt, writes outcome, syncs and calls native shutdown. Fatal exit saves logs and allows 10-minute repair grace before shutdown.",
    "heartbeat_automation_id": "hrdecoder-v3", "local_report": str(HERE/"4090训练启动记录.md"),
    "resume_first": "Inspect running PID/progress before any new launch; use explicit --resume with last.pt only after a failure. Do not duplicate active training.",
    "launch_mode": "v3 only; no paid control runs"})
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
report = f'''# 4090 正式训练启动记录

记录时间：{now}。实际硬件 RTX4090 24GB 显存、120GiB 系统内存、16 CPU。服务器 {HOST}:{PORT}，实验目录 `{ROOT}`。按用户要求复用既有缓存与完整验收，不再次扫描 78.45GiB 缓存/数据哈希。

## 已启动的主实验

- 唯一 v3 主运行，2048、batch2、40,000 SGD 更新，锁定 seed42/ImageNet 初始化及原预算。未启动付费对照。
- 新服务器旧 torch2.0.1 的 BF16 双线性插值缺少 CUDA 实现；仅将这些插值以 FP32 计算再转回 BF16，保持梯度。卷积/注意力仍使用 BF16，损失和概率差分仍用 FP32。CPU 路径没有改变。修改后实际 2048/batch2 GPU 前向、私有梯度反向和 SGD probe 已通过；probe 权重丢弃。
- GPU smoke 峰值 allocated {smoke['peak_allocated_bytes']/1024**3:.2f}GiB、reserved {smoke['peak_reserved_bytes']/1024**3:.2f}GiB，骨干调用一次。当前正式训练峰值 allocated {performance['peak_allocated_bytes']/1024**3:.2f}GiB、reserved {performance['peak_reserved_bytes']/1024**3:.2f}GiB。
- 首次读取配置的启动测量发现数据等待明显；首个检查点前中止并舍弃该轨迹，最后记录 120 个 batch，不将其权重、BN 或动量带入主运行。最终主运行从同一 seed42 初始化重新开始。其第一步损失与此前逐项一致。
- 读取使用 12 worker、每 worker OpenCV/PyTorch 1 CPU 线程、prefetch1、persistent workers、pinned memory、nonblocking CUDA 拷贝。固定曝光 seed 和顺序保留，未增加 batch 或改变增强/训练预算。

## 吞吐与时间估算

已测 {performance['measured_batches']} 步，最近窗口 {performance['window_batches']} 步，排除初始 warmup。平均 GPU step（含原有限值检查和损失同步）{performance['mean_step_compute_seconds']:.3f} 秒，CPU 数据等待 {performance['mean_data_wait_seconds']:.3f} 秒，每 batch 合计 {performance['mean_batch_seconds']:.3f} 秒。纯 40k 训练外推 {performance['estimated_40000_train_hours_excluding_validation']:.2f} 小时，包含验证/保存先按 **13–15 小时**安排，约北京时间 10月9日清晨 6–8 点。该数值是早期吞吐外推，不是保证；后续根据实际进度更新。

最近 {len(util)} 个两秒 GPU 采样，利用率均值 {statistics.mean(util):.1f}%、中位数 {statistics.median(util):.1f}%，平均功耗 {statistics.mean(power):.1f}W。用训练吞吐判断优化成效；功耗不要求达到450W，不用无用张量或额外运算把曲线填满。

## 保存、监控与关机

正式进程 PID {running['pid']}，脱离 SSH 的服务器控制器 PID {running['controller_pid']}。日志 `reports/v3_training.log`，GPU 两秒采样 `reports/gpu_telemetry.csv`，即时性能 `training_runs/v3/performance.json`。每1k保存 last.pt（模型/optimizer/RNG），每4k全量76张val。主结果40k固定端点，best-val为次，不用test选点。

控制器在训练进程正常结束、40k标志和last.pt存在后写 `GPU_CAMPAIGN_OUTCOME.json`，执行sync及AutoDL原生关机。若致命错误，先保存日志并留10分钟修复窗口，之后停止空闲GPU。当前聊天已启用30分钟后续检查（心跳id `hrdecoder-v3`），正常进度不重复通知。服务器端关机不依赖本地电脑保持在线。

**当前为 RUNNING，尚无本轮验证精度或训练完成结果。** 关机观察仅在实际执行之后记录，不能把自动关机配置当作已经关机。
'''
(HERE/"4090训练启动记录.md").write_text(report,encoding="utf-8")
print(json.dumps(snapshot))
