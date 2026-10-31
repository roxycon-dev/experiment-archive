"""Finalize archived CPU evidence and execute the user's shutdown instruction."""
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta, timezone
import paramiko

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE.parent/"cache_execution/server_receipts"
sys.path.insert(0, str(HERE.parent/"local_preparation"))
from remote import connect

def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))

ready = read(RECEIPTS/"CACHE_READY.json")
gpu = read(RECEIPTS/"GPU_CACHE_READY.json")
assert ready["status"] == "PASS" and ready["images_per_member"] == 454
assert ready["all_file_and_tensor_SHA_verified"]
assert ready["member_identity_sha256"] == gpu["member_identity_sha256"]
assert ready["compressed_bytes"] == gpu["compressed_bytes"]
assert all(row["covariance_rank_relative_tolerance_1e_minus6"] == 2 for row in ready["class_statistics"])
preparation = read(RECEIPTS/"reports/TRAINER_PREPARATION_PASS.json")
manifest = HERE/"execution_manifest.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == preparation["source_manifest_SHA"]
for name, digest in read(manifest).items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest, name

# The running archiver may predate the additional route report; always fetch it.
client = connect()
with client.open_sftp() as ftp:
    for name in ("SERVER_INITIALIZATION_PASS.json", "SERVER_OPTIMIZER_RESUME_PASS.json", "SERVER_REAL_HRNET_ROUTE_PASS.json"):
        destination = RECEIPTS/"reports"/name
        ftp.get("/root/autodl-tmp/HRDecoder_v3_20261008/reports/"+name, str(destination))
        assert read(destination)["status"] == "PASS"

resource = read(RECEIPTS/"reports/low_memory_audit_resources.json")
now = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
rows = "\n".join(f"| {row['class']} | {row['du_std']:.6f} | {row['ds_std']:.6f} | {100*row['opposite_sign_fraction']:.2f}% | {row['covariance_rank_relative_tolerance_1e_minus6']} | {row['images_with_gt_positive']}/454 |" for row in ready["class_statistics"])
report = f"""# 无卡准备与全量验收完成

记录时间：{now}（Asia/Shanghai）。本轮未开始正式学生训练。

三路 train 缓存 A/U/S 各 454 张，共 1,362 份，原图坐标、EX/HE/SE/MA 四通道、FP32 无损压缩。实际缓存 {ready['compressed_bytes']/1024**3:.2f} GiB，解压张量负载 {ready['uncompressed_bytes']/1024**3:.2f} GiB。完整数据留在服务器 `/root/autodl-tmp/HRDecoder_v3_20261008/cache/train`，本地只有清单/证据和一张用于真实管线检查的缓存样本。

## 缓存全量复核

454 张的全部文件 SHA 和全部 FP32 张量 SHA 已再次读盘核对，形状、范围、有限值、数据身份与有符号坐标逆变换通过。S20 EMA 历史验证哨兵逐位相同，最大差值 0。全量复核耗时 {ready['seconds']/60:.1f} 分钟；采用单 worker、分块 ZIP 读取，峰值 RSS {resource['peak_RSS_bytes']/1024**2:.1f} MiB，适配无卡实例的 0.5 CPU/2 GiB 内存。

| 类别 | dU 标准差 | dS 标准差 | 两差分反向比例 | 协方差秩 | 有 GT 前景的图像 |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

秩使用相对阈值 1e-6。该结果证明训练分歧目标存在且没有退化为同一个标量方向，不证明隐表示独立、知识有益、泛化或 mIoU 改善。教师预测仍然是 in-sample，OOF=false。

## 训练器准备与审查

- 本地真实 HRNet 的 28 项 CPU 检查通过，真实三路缓存的 2048 画布几何与差分构造通过。
- 服务器 torch2.0.1+cu117 中，真实 ImageNet 初始化、真实 HRNet 私有梯度路由及 optimizer/动量/RNG 断点恢复检查通过；路由相对参考的各组最大差值均为 0。
- 每个分割骨干张量都与锁定的 ImageNet W48 文件一致；没有将 DDR、ConvNeXt 或 DINO 教师任务权重加载到学生。
- 基座 65,572,420 参数，新增 6,419,648，完整学生 71,992,068。服务器 seed42 初始化状态单独锁定，本地旧版本的状态不冒充它，也未声称比对到历史 B0 的初始快照。
- 原 LR/HR 解码顺序、四个 HR feature crops、BN 行为、0.1 HR Dice、40k 全局 batch、batch2 和原 SGD/poly/warmup 保留。统一修正后的有效 mask 用于全部监督；同条件 B0/B1、同容量 KD 和普通辅助回传模式已经准备，尚未开展付费对照训练。
- 当前只支持单卡。DDP 被显式拒绝；40k 固定端点为主、4k 间隔 val best 为次，不用 test 选点。详情见 `review_notes.md`。

## 切回 4090 后

服务器启动脚本：

```bash
/bin/bash /root/autodl-tmp/HRDecoder_v3_20261008/training_execution/launch_gpu.sh
```

脚本先运行 batch2/2048 的真实 CUDA BF16 前向、反向和一次独立 SGD probe，记录显存、损失、骨干调用次数；该更新丢弃。通过后新进程重新构造 ImageNet seed42 学生，立即开始正式 40k。本次尚未验证 2048 CUDA 显存/速度和收敛，不能把 CPU PASS 当作该 GPU 检查结果。

缓存、代码或协议身份改变时会拒绝沿用旧 smoke。已有正式 last.pt 必须显式走 `resume_gpu.sh`，不会悄悄覆盖或从头重跑。

已在证据归档后执行 AutoDL 原生关机命令，并观察 SSH 断开；平台实例状态以控制台为准。切回 4090 后即可继续执行上述检查与正式训练。

## 诊断样本

以下只用于核对原图坐标与有符号概率差分；选取 train 样本，不是精度成绩。无该类 GT 时显示图像中央并标记 no GT。

![Train cache native ROI diagnostic](D:/dataset/experiment/HRDecoder_ArchitectureDesign_20261008/training_execution/train_cache_diagnostics.png)
"""
(HERE/"无卡准备与全量验收记录.md").write_text(report, encoding="utf-8")

issued = {"command": "/bin/bash /usr/bin/shutdown", "authorized_by_prior_user_instruction": True, "time_unix": time.time()}
(RECEIPTS/"no_gpu_shutdown_command_issued.json").write_text(json.dumps(issued, indent=2)+"\n")
client.exec_command("sync\n/bin/bash /usr/bin/shutdown", timeout=20)
client.close()
observation = None
for attempt in range(6):
    time.sleep(5)
    try:
        check = connect(); check.close()
    except (OSError, paramiko.SSHException) as error:
        observation = {"ssh_unreachable_after_authorized_shutdown": True,
            "time_unix": time.time(), "error": str(error)}
        break
assert observation is not None, "Shutdown sent but SSH still reachable; inspect before claiming power-off"
(RECEIPTS/"no_gpu_shutdown_observation.json").write_text(json.dumps(observation, indent=2)+"\n")
state_path = HERE.parent/"cache_execution/handover_state.json"
state = read(state_path)
state.update({"stage": "FULL_CACHE_AUDIT_AND_CPU_PREPARATION_PASS_SERVER_SHUTDOWN",
    "latest_user_instruction": "Server reopened in no-GPU mode. All CPU preparation completed. Prior instruction: shut down when finished.",
    "full_cpu_cache_audit": "PASS: 454/454 all file and tensor SHA; signed statistics and GT coverage",
    "cpu_preparation_after_gpu": "PASS: real HRNet integration, masks, routing, optimizer resume, pinned initialization, source audit",
    "GPU_smoke": "PENDING_4090_RESTART", "formal_training_started": False,
    "active_local_monitor_session": None, "server_controller": "No training or audit process remains; shutdown was executed",
    "shutdown": observation, "local_report": str(HERE/"无卡准备与全量验收记录.md"),
    "resume_first": "Reconnect and verify 4090; run training_execution/launch_gpu.sh. User already authorized beginning training after preparation.",
    "launch_mode": "v3 only; do not launch paid control runs automatically"})
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"status": "CPU_PREPARATION_COMPLETE_AND_SERVER_SSH_UNREACHABLE",
    "report": str(HERE/"无卡准备与全量验收记录.md"), "formal_training_updates": 0}, ensure_ascii=False), flush=True)
