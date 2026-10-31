"""Persist the verified CET launch separately from the stopped v3 trajectory."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
LOCAL=ROOT/'cache_execution/server_receipts/cet'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
snapshot=load(LOCAL/'latest_snapshot.json')
running=load(LOCAL/'reports/cet/GPU_CAMPAIGN_RUNNING.json')
smoke=load(LOCAL/'training_runs/cet/GPU_SMOKE_PASS.json')
protocol=load(LOCAL/'training_runs/cet/protocol.json')
assert snapshot['runtime']['training_alive'] and snapshot['runtime']['controller_alive']
assert smoke['status']=='PASS' and smoke['identity_SHA']==protocol['identity_SHA']
progress,perf=snapshot['progress'],snapshot['performance']
assert progress and progress['completed']>0 and perf['completed_update']>0
start=datetime.fromtimestamp(running['start_unix'],timezone(timedelta(hours=8)))
report=HERE/'CET服务器训练启动记录.md'
text=f'''# HRDecoder-CET 正式训练启动记录

更新时间：{snapshot['checked_at']}。状态 **RUNNING**，本轮尚未完成40k；验证精度尚未知。

## 训练与服务器

方案：research_v5/00_HRDecoder_CET_架构与训练方案.md 与01_CET_架构规格.json。CET程序位于 cet_execution，运行数据位于 training_runs/cet，日志及控制器回执位于 reports/cet。服务器 connect.nmb1.seetacloud.com:11347，RTX4090实际24GB显存、120GiB系统RAM。SSH仅使用原专用key，未存储或输出密码。

正式训练起于北京时间 {start:%Y-%m-%d %H:%M:%S}，PID {running['pid']}，控制器PID {running['controller_pid']}。与v3检查点及诊断文件分开；没有改写旧任务。仅CET主模型，2048、batch2、40,000次SGD更新。ImageNet HRNet-W48、base种子42、新增种子20261008，未使用v3的8k/20k任务参数。原曝光、增强、SGD/lr/λK保持方案定义。每1k原子保存自身last.pt，每4k完整76张val，固定40k为主结果，best-val为次，不用test选择。

## 实现核对

原HRNet/HRDecoder单主干、原HR裁块路径；L/G定义与v3相同。每类keep/on/off，概率前向P=πkeep·p0+πon，p0条件输入detach，但概率合成主路径不断梯度。头W=0、bias按每个PixelShuffle子像素重复log(.98/.01/.01)。训练使用FP32 log-softmax与logaddexp，不对概率clamp后求logit。推理分别resize主头和12通道动作logits再sigmoid/softmax与合成，没有教师、GT、在线H4或TTA输入。

GT采用阳性对才启用平方Dice＋.25×分组正/负BCE；.75最终＋.25主头＋.1HRcrop。KD按GT一致区域筛选，以全部有效像素为分母；动作目标由C+/C−/C0构造，Cunknown及invalid剔除，区域固定1/3、缺组为0。动作.25λK的梯度只写入L/G/头，Ltask更新全部参数，每次更新一次optimizer.step。

新增参数实测 **6,416,576**，全模型 **71,988,996**，与方案一致。新增参数零末层导致首步塔的任务梯度为0是预期行为，第二次更新塔已出现非零梯度。新增MAC为方案解析值，未作为实测吞吐或精度证据。

本地Torch2.6及服务器固定Torch2.0.1+cu117均通过动作初始化/子像素顺序、条件detach、端点无0/0、Cunknown排除、固定三组归一化、all-invalid反传零、极端动作恢复梯度、GT类恢复、KL筛选、辅助梯度边界、原尺寸推理及学习率计划检查。向量化GT/KL的数值和梯度已与v4修复参考逐项一致。复用原初始化审计和已有数据/缓存身份，不重复全量数据或缓存SHA扫描；新增代码/方案小文件的身份已固定。

## 真实GPU验证与速度

真实2048/batch2、BF16网络/FP32概率损失，完整前向、路由反向与两次SGD（含动量）通过。两次冒烟更新均丢弃，正式进程重新加载ImageNet起点。每次HRNet只调用一次，全部梯度有限，第二次L/G/头均非零。冒烟峰值allocated {smoke['peak_allocated_bytes']/1024**3:.2f} GiB，reserved {smoke['peak_reserved_bytes']/1024**3:.2f} GiB；第二次更新 {smoke['step_seconds'][1]:.3f}秒。此验证不证明40k全程稳定或最终精度。

1,362份教师缓存全部放入96GiB memfd，FP32值与磁盘原缓存不变；8线程一次载入约103秒，无全量hash复查。12 worker，每个OpenCV/PyTorch单线程，pin_memory、prefetch1、一个未来GPU批次、CUDA事件等待；保持NCHW，不为显存或功耗改变batch和预算。首批CPU→GPU的image/label/valid/A/U/S逐位一致。

最新进度 {progress['completed']}/40,000（性能记录 {perf['completed_update']}）；最近{perf['window_batches']}步约{perf['mean_batch_seconds']:.3f}秒/步，数据等待{perf['mean_data_wait_seconds']*1000:.3f}毫秒。首次稳定窗口的纯训练外推{perf['estimated_remaining_train_hours_excluding_validation']:.2f}小时，但1500后动作路由反向增加，另含10次完整验证和保存；暂按 **总计12–14小时，今天约14–16点**，4k后再用实测验证时间修正。

当前GPU原始采样：

```text
{snapshot['runtime']['gpu_status'].strip()}
```

## 监控与退出

每45分钟检查，只在新验证点、完成、失败、明显停滞或ETA明显变化时通知。逐点汇报mIoU及EX/HE/SE/MA IoU、precision、recall、较前点百分点变化和best-val，并与原HRDecoder及v3同节点参照；首点只建立基线。reported_steps另存cet_execution/validation_reporting.json，不继承v3的已汇报步骤。

遵循用户最新“不关机”和“不干预另一位AI诊断”要求：CET控制器没有关机命令，完成/失败仅记录状态并释放自己拥有的RAM缓存；服务器保持开机。v3的shutdown_hold保留、旧控制器已退出。不得终止、恢复或修改外部诊断。监控接管CET后不得运行旧archive_current_training.py或恢复v3。

原始小型记录归档在 cache_execution/server_receipts/cet；主日志为reports/cet/cet_training.log，启动协议为training_runs/cet/protocol.json，GPU冒烟回执为training_runs/cet/GPU_SMOKE_PASS.json。协议身份 `{protocol['identity_SHA']}`。
'''
report.write_text(text,encoding='utf-8')
handover_path=ROOT/'cache_execution/handover_state.json'
state=load(handover_path)
old_stop=HERE/'previous_v3_handover.json'
if not old_stop.exists():save(old_stop,state)
state.update(stage='CET_RUNNING',formal_training_status='RUNNING',active_run='cet',
    latest_user_instruction='Run research_v5 HRDecoder-CET on the same server; main model only, 2048 batch2 40k. Keep server on; do not interfere with other AI diagnostics.',
    server_controller=running,current_outcome=None,current_completion_receipt=None,
    automatic_shutdown='DISABLED: keep server online on completion or failure; CET controller has no shutdown command.',
    resume_first='Inspect CET only. Never resume v3. If authorized recovery is needed, only explicit CET own last.pt with matching identity.',
    local_report=str(report),active_execution_directory=str(HERE),active_remote_execution_directory='/root/autodl-tmp/HRDecoder_v3_20261008/cet_execution',
    active_remote_run='/root/autodl-tmp/HRDecoder_v3_20261008/training_runs/cet',
    active_remote_reports='/root/autodl-tmp/HRDecoder_v3_20261008/reports/cet',
    previous_v3_handover=str(old_stop),previous_v3_stop_report=state.get('stop_report'),
    latest_performance=perf,current_runtime=snapshot['runtime'],last_monitor_receipt=str(LOCAL/'latest_snapshot.json'),
    estimated_remaining_train_hours_excluding_validation=perf['estimated_remaining_train_hours_excluding_validation'],
    validation_reporting_ledger=str(HERE/'validation_reporting.json'),validation_report=str(HERE/'验证指标与趋势.md'),
    reported_validation_steps=[],available_validation_steps=[],validation_anomaly=None,
    checkpoint_downloaded=False,waiting_for_new_framework=False,training_complete=False,
    full_RAM_cache={'status':'READY','pid':13598,'files':1362,'bytes':snapshot['runtime']['RAM_bytes'],
                    'all_teachers_in_RAM':True,'storage':'shared memfd FP32 NPY/NPZ','hash_rescan':False},
    heartbeat_automation_status='ACTIVE',heartbeat_check_interval_minutes=45,
    current_shutdown_observation={'shutdown_command_issued':False,'server_online_verified':True,'checked_at':snapshot['checked_at']})
save(handover_path,state)
(HERE/'验证指标与趋势.md').write_text('# CET 验证指标与趋势\n\n正式训练已启动，尚无完成的4k验证点，当前不作精度或趋势结论。每个点按76张val原始TP/FP/FN核对并汇报；40k固定端点为主结果。\n',encoding='utf-8')
save(HERE/'launch_receipt.json',{'status':'RUNNING','checked_at':snapshot['checked_at'],'server':state['server'],
    'running':running,'protocol_identity':protocol['identity_SHA'],'local_report':str(report),
    'do_not_control_external_diagnostics':True,'shutdown_authorized':False})
print(json.dumps({'status':'RUNNING','training_pid':running['pid'],'controller_pid':running['controller_pid'],
                  'completed':progress['completed'],'report':str(report)},ensure_ascii=True))
