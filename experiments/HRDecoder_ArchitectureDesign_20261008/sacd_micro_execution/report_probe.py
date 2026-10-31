"""Verified per-point metrics and a cautious terminal diagnostic analysis."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
from gate import verify

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
ARCHIVE=ROOT/'cache_execution/server_receipts/sacd_micro'
gate=json.loads((ARCHIVE/'reports/sacd_micro/MICRO_GATE.json').read_text())
reference=json.loads((ROOT/'sacd_execution/val_label_reference.json').read_text())
terminal=json.loads((HERE/'PROBE_TERMINAL_OBSERVATION.json').read_text())
assert gate['status']=='REJECTED' and gate['initial_count_equality_required_not_relaxed']
series=json.loads((HERE/'screening_series.json').read_text(encoding='utf-8'))
records=series['warm_val']
assert len(records)==6
for row in records:
    verify(row,reference['ground_truth_positive_pixels'])
    arm=row['arm']
    protocol=json.loads((ARCHIVE/'training_runs/sacd_micro'/arm/'protocol.json').read_text())
    complete=json.loads((ARCHIVE/'training_runs/sacd_micro'/arm/'PROBE_COMPLETE.json').read_text())
    smoke=json.loads((ARCHIVE/'training_runs/sacd_micro'/arm/'GPU_SMOKE_PASS.json').read_text())
    assert row['identity_SHA']==complete['identity_SHA']==smoke['identity_SHA']==protocol['identity_SHA']
    assert hashlib.sha256(json.dumps(protocol['identity'],sort_keys=True).encode()).hexdigest()==protocol['identity_SHA']
    assert smoke['status']=='PASS' and complete['status']=='COMPLETE'
    assert complete['probe_updates']==500 and smoke['formal_updates']==0 and smoke['discarded_optimizer_steps']==3
    for filename,digest in protocol['identity']['code'].items():
        assert hashlib.sha256((HERE/filename).read_bytes()).hexdigest()==digest
series.update(warm_gate=gate,warm_gate_unreported=True,terminal_observation=terminal)
(HERE/'screening_series.json').write_text(json.dumps(series,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
by_arm={arm:sorted([r for r in records if r['arm']==arm],key=lambda r:r['probe_updates']) for arm in ('control','feature_isolated')}
lines=['# 配对短测指标与分析','',
 '同一20k父模型、SGD momentum和RNG，两个诊断臂各额外500更新；各点均完整76图val。不是fresh初始化，不是40k结果，未读取test。',
 '', '表中四类列均按IoU / precision / recall排列，单位为%。原始TP/FP/FN与独立标签GT阳性总数逐点核对。', '',
 '| 组别 | 额外步 | mIoU | EX IoU/P/R | HE IoU/P/R | SE IoU/P/R | MA IoU/P/R |',
 '|---|---:|---:|---|---|---|---|']
for arm in ('control','feature_isolated'):
    for row in by_arm[arm]:
        values=['%.3f/%.3f/%.3f'%(row['IoU_percent'][i],row['precision_percent'][i],row['recall_percent'][i]) for i in range(4)]
        lines.append('| %s | %d | %.3f | %s |'%(arm,row['probe_updates'],row['mIoU_percent'],' | '.join(values)))
lines+=['', '| 额外步 | 修改−原结构 mIoU百分点 | EX | HE | SE | MA |', '|---|---:|---:|---:|---:|---:|']
for original,modified in zip(by_arm['control'],by_arm['feature_isolated']):
    changes=[b-a for a,b in zip(original['IoU_percent'],modified['IoU_percent'])]
    lines.append('| %d | %+.6f | %s |'%(original['probe_updates'],modified['mIoU_percent']-original['mIoU_percent'],' | '.join('%+.6f'%x for x in changes)))
lines += ['', '各臂相邻点变化（mIoU及EX/HE/SE/MA IoU，均为百分点）：','']
for arm,rows in by_arm.items():
    for before,after in zip(rows,rows[1:]):
        changes=[after['mIoU_percent']-before['mIoU_percent']]+[b-a for a,b in zip(before['IoU_percent'],after['IoU_percent'])]
        lines.append('%s %d→%d：%s。'%(arm,before['probe_updates'],after['probe_updates'],' / '.join('%+.6f'%x for x in changes)))
lines += ['', '## 预登记门槛判定', '', '结果：REJECTED，未启动fresh8k或新的40k。指标门槛不调整。','']
for name,passed in gate['decisions'].items():lines.append('- %s：%s。'%(name,'PASS' if passed else 'REJECTED'))
lines += ['',
 '500步修改组高2.494001个百分点，主要来自EX IoU高10.125778个百分点：EX FP421640→199624，约减少52.66%；EX recall63.447764%→60.752468%，降低2.695295个百分点。该点有减少EX误检的信号。',
 '',
 'HE FP421837→317254，减少24.792278%；precision63.045217%→67.483673%，但recall59.676895%→54.598963%，减少5.077932个百分点，超过预登记上限2个百分点；HE IoU仍低0.984669个百分点。不能只称误检减少而隐去漏检增加。',
 '',
 '250步修改组mIoU落后1.317960个百分点（HE IoU低4.350010个百分点），500步才反超。仅两次训练后的验证，证据不支持持续稳定优势；不据此判定架构无希望，也不能据500步单点保证完整40k达标。',
 '',
 '原结构自身250→500的EX IoU38.492593%→25.897931%，FP175123→421640；其基础头同时下降，显示共享主干/概率校准也有短期波动。修改组500步基础头mIoU36.153516%，原结构33.363257%，说明差异涉及基础路径。尚不能把影响全部归因于间接feature梯度。',
 '',
 '两组初始mIoU有0.005072个百分点差别：control37.189870%、modified37.184798%，原父模型37.184798%。原始counts并不完全相同。CPU同权重数学前向相同已验证，但真实BF16/CUDA在不同训练计算图smoke后使用cuDNN benchmark，数值/内核选择是可能解释，未做复现确定；不能伪称完全配对初始预测，也不把这点当作已证实的代码错误。严格初始一致性门槛保持REJECTED。',
 '',
 '两组真实2048/batch2非零辅助GPU smoke、全部梯度有限、CPU梯度路径与采样种子复用均PASS。两组正式500步和checkpoint回执完整，代码与身份一致；缺失的小型val计数参考文件已补传，仅重做文本gate，不重复优化器更新。失败loader尝试与收尾错误均保留。',
 '',
 '机制范围：feature detach只移除间接L/G→HRNet梯度，不消除final对z0的直接耦合，不改变0.5处学生action目标突变，也不恢复原版纯BinaryDice。暖切换保留旧20k动量和已有权重，因此不能直接外推fresh初始化结果；本测试不构成机制因果证明。',
 '',
 '服务器SSH已核实在线，两组训练及控制器均已结束，GPU0%/1MiB。原SACD仍暂停在20k持久父模型，未恢复v3、未关机、未触碰另一AI诊断。自己的RAM缓存保留等待用户对同候选fresh8k探索是否继续的选择。',
 '',
 '目标未达：没有新的完整40k，也没有独立test结果。继续探索必须明确承认当前warm筛查未过；不能事后改阈值或覆写REJECTED。']
(HERE/'配对短测指标与分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
state_path=ROOT/'iterative_execution/iteration_state.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
state.update(status='WARM_SCREEN_REJECTED_AWAITING_USER_CHOICE',active_run='sacd_micro',
             latest_warm_gate=gate,screening_terminal_observation=terminal,
             another_full_run_authorized=False,conditional_extension_activated=False,
             fresh8k_started=False,current_task_training_active=False,goal_achieved=False,
             reason_waiting='Registered warm screening failed. Same-candidate fresh8k exploration question pending; no dependent paid work until reply.',
             diagnostic_report=str(HERE/'配对短测指标与分析.md'))
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
hand_path=ROOT/'cache_execution/handover_state.json'
hand=json.loads(hand_path.read_text(encoding='utf-8'))
hand.update(stage='WARM_SCREEN_REJECTED_AWAITING_USER_CHOICE',formal_training_status='PAUSED_AT20K',
            current_task_training_active=False,server_shutdown_authorized=False,
            warm_screen_rejected=True,fresh8k_started=False,latest_probe_terminal_observation=terminal)
hand_path.write_text(json.dumps(hand,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'METRICS_AND_DIAGNOSIS_SAVED','counts_verified_points':len(records),
                 'gate':'REJECTED','goal_achieved':False,'report':str(HERE/'配对短测指标与分析.md')}))
