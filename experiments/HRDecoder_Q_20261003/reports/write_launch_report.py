import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from qlib.common import read_json, utc_now, write_json

gpu = read_json(root/'server_results/reports/gpu_preflight.json')
learn = read_json(root/'server_results/reports/learnability.json')
source = read_json(root/'reports/source_identity.json')
p = gpu['performance']
status = {'stage': 'FORMAL_Q_DDR_SEED42_TRAINING', 'formal_training_started': True,
          'endpoint': 16000, 'seed': 42, 'arm': 'Q_DDR', 'gpu_acceptance': 'PASS',
          'Q_MIX': 'NOT_READY_MISSING_OFFICIAL_ORIGINAL_IMAGES', 'monitor_minutes': 45,
          'shutdown_authorized': True, 'budget_authorization': 'unlimited_explicit_user_authorization_20261003',
          'source_sha256': source['sha256'], 'updated_at': utc_now()}
write_json(root/'reports/preparation_status.json', status)
rows = [r for r in gpu['cost_rows'] if r['formal_runs'] in (1, 2)]
costs = '\n'.join(f"|{r['endpoint']}|{r['formal_runs']}|{r['gpu_hours_including_buffer']:.2f}|" for r in rows)
report = f'''# 准备验收与开训记录

必要验收全部通过，Q_DDR seed42 已正式启动，首个发现运行固定至 16000 次有效更新。服务器实际训练日志已观察到连续更新；本文件不把准备试跑或 loss 下降当作正式验证收益。

## 已冻结身份

- 官方 DINOv3 ViT-B/16 LVD 权重 SHA256：`73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`；官方许可邮件已取得，严格加载 missing/unexpected 均为 0。
- 官方代码 commit：`6876159a11b4df116f30f667f8c9888617df0751`；本轮验收源文件 SHA256：`{source['sha256']}`。
- DDR train454/val76/test227 身份固定；当前没有读取 test 标签像素。P0 的 76 张逐图及汇总计数精确复现，val 43.4877032210%。
- 实际资源：micro2×accum4，BF16，FP32 损失与概率；已观察官方 Flash SDPA 算子。
- 原生 1024 窗口、中心768、halo128；全部几何、忽略、损失、冻结/梯度/参数更新、严格恢复检查通过。

## 必要 GPU 实测

|项目|结果|
|---|---|
|连续真实末四层反传|100 次有效更新|
|冻结更新平均耗时|{p['frozen_seconds_per_update_mean']:.3f} 秒|
|解冻更新平均耗时|{p['unfrozen_seconds_per_update_mean']:.3f} 秒|
|解冻耗时 P90|{p['unfrozen_seconds_p90']:.3f} 秒|
|平均数据等待|{p['data_wait_seconds_mean']:.3f} 秒|
|峰值 allocated/reserved|{p['peak_allocated_bytes']/1024**3:.2f}/{p['peak_reserved_bytes']/1024**3:.2f} GiB|
|完整 val76 耗时|{p['full_76_val_seconds']:.1f} 秒|
|固定样本初/末10次平均 loss|{learn['initial_loss']:.4f}→{learn['final_loss']:.4f}|
|最大类别空间概率标准差|{learn['max_class_spatial_std']:.4f}|

固定样本仅用于接线验收，模型、优化器和随机种子已重置，正式运行未继承任何试跑更新。

## 时长估计与投入规则

下表含正式训练、规定 val 次数、30分钟备份估计及20%缓冲，尚不含晋级后最终 test。实际价格未知，不能填写人民币实付；用户已明确费用没有上限。

|共同终点|正式运行数|预计 GPU 小时，含缓冲|
|---|---|---|
{costs}

初始只执行 Q_DDR seed42。8k 仅在三项严重失败条件同时成立时停止；16k 单模相对12k增加≥0.3点才延至24k。同一公式两个相邻晚期节点均≥P0+0.5点才复核 seed2024。两种子共同节点一次锁定选择、通过测试闸门后才 test；没有持续信号就终止，不追加模型或搜索。

Q_MIX 缺官方配套原图，仍为 NOT_READY。本轮不用衍生方图替代。

## 存储、巡检与关机

已按授权删除本轮无用 A0/H1/U1 历史非入选缓存约15.9 GiB，保留模型、数据、日志及 P0。删除清单已回收并核验 SHA；没有声称这些原始缓存已备份。

45分钟巡检已启用。每套 val76 FP32 四通道概率未压缩约5.98 GiB；首臂12k/16k可容纳，延长或第二种子前必须安排已核验备份或其他实验存储，避免再次满盘。

结束时先本机回收必要成果并核验 SHA，再关闭本轮服务器并核验平台 OFF。当前训练正在运行，尚未关机。后续步骤见《后续执行账本》；验收原始 JSON 在 `server_results/reports`。
'''
(root/'reports/准备验收与开训记录.md').write_text(report, encoding='utf-8')
print('Launch report and local stage saved')
