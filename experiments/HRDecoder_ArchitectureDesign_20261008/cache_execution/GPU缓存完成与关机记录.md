# GPU 缓存完成与关机记录

关机观察时间：2026-10-08T14:11:52+08:00

本轮只完成教师软概率缓存，尚未启动学生训练。A/U/S 各 454 张训练图，共 1,362 份缓存；四通道依次为 EX、HE、SE、MA，保留原图坐标与 FP32，采用 NPZ 无损压缩。

| 教师 | 图像数量 | 缓存体积 |
| --- | ---: | ---: |
| A | 454 | 29.84 GiB |
| U | 454 | 30.58 GiB |
| S | 454 | 18.03 GiB |
| 合计 | 1,362 | 78.45 GiB |

不压缩时为 104.90 GiB。关机前数据盘剩余 29.45 GiB，110 GiB 数据盘足够保存本轮缓存、模型与执行材料。

服务器缓存目录：`/root/autodl-tmp/HRDecoder_v3_20261008/cache/train/{A,U,S}`。完整概率文件保存在服务器数据盘；本地保存清单、哈希与验收记录，没有下载整套 78 GiB 缓存。

## 已通过的检查

- 输入清单核对：757 张图像与 757 张标签内容 SHA、尺寸，以及 438 个历史源码文件。
- 每份缓存在写入时验证 FP32、形状、有限值与 [0,1] 范围；压缩后重新读取，概率位模式 SHA 与压缩前一致，文件 SHA 已记录。
- 三路索引状态 COMPLETE，均为 454/454；所有 ID、尺寸、标签/图像身份及文件大小核对通过；归档到本地的三个索引 SHA 再次匹配服务器清单。
- S20 EMA 验证哨兵图与历史缓存逐位相等：最大差值 0，阈值分歧像素 0。

## 关机与剩余工作

已按用户授权执行 AutoDL 原生 `/bin/bash /usr/bin/shutdown`，随后 SSH 连接不可达。该观察确认命令执行后连接断开；平台实例/计费状态以 AutoDL 控制台为准。

全量 CPU 复核在完成 210/454 张时停止，避免继续占用有卡模式。完整导出后的再次全量读盘 SHA 复核、差分能量/协方差秩统计仍待无卡模式完成，当前不宣称全量 CPU 验收通过。

无卡模式启动后先运行：

```bash
cd /root/autodl-tmp/HRDecoder_v3_20261008
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 /root/miniconda3/envs/hrdecoder_q_20261003/bin/python -u cache_audit.py
```

该命令不需要 GPU，成功时生成 `CACHE_READY.json` 与 `reports/train_cache_audit.json`。随后继续完整 HRNet 集成、训练几何处理、AMP/DDP 路由与训练器审核；当前只做过本地小尺寸模块检查，不能将其当作完整训练器验收。

## 本地记录

- `server_receipts/GPU_CACHE_READY.json`：完整导出、体积与身份信息。
- `server_receipts/cache/train/{A,U,S}/index.json`：逐文件与逐张量 SHA。
- `server_receipts/reports/cpu_audit_deferred.json`：无卡恢复命令。
- `server_receipts/shutdown_command_issued.json` 和 `shutdown_observation.json`：授权关机与连接观察。
- `handover_state.json`：后续接续状态。SSH 凭据未写入报告。
