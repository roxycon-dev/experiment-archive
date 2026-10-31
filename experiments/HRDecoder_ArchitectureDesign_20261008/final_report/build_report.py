"""Rebuild the report from small local receipts; never contact the server."""
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECEIPTS = ROOT / 'cache_execution/server_receipts'
VAL_TARGET = 45.32183122738841
TEST_TARGET = 45.860383832668
NAMES = ['EX', 'HE', 'SE', 'MA']
RUNS = {
    'v3': ('training_execution', '机制重构v3，用户主动停止'),
    'cet': ('cet_execution', 'CET训练目标；完成固定40k'),
    'sacd': ('sacd_execution', '动作纠错；20k后转入配对诊断'),
    'residual': ('residual_execution', 'detach连续FP32残差，移除动作KL'),
    'expert': ('expert_execution', '独立私有分割与可学习logit融合'),
    'adapter': ('adapter_execution', '轻量直接多尺度纠错，1/4网格'),
    'detail': ('detail_execution', '1/2细节纠错网格，后发生类别崩塌'),
    'foundation': ('foundation_execution', '基础阳性梯度保护与基础教师蒸馏'),
    'aligned': ('aligned_execution', '私有BCE/KL按全GT正负组平衡'),
    'semantic': ('semantic_execution', '尺度保留语义拼接、raw-logit旁路、分支分别裁剪'),
    'iou': ('iou_execution', '同Semantic网络，私有Dice改为精确binary Lovasz hinge'),
}


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def metrics(classes, gt):
    result = []
    for i, cls in enumerate(classes):
        tp, fp, fn = cls['counts']
        assert all(type(v) is int and v >= 0 for v in (tp, fp, fn))
        assert tp + fn == gt[i]
        values = [100 * tp / d if d else 0.0 for d in (tp + fp + fn, tp + fp, tp + fn)]
        for key, alternative, value in zip(('IoU', 'precision', 'recall'),
                                          ('IoU_percent', 'precision_percent', 'recall_percent'), values):
            assert math.isclose(cls.get(key, cls.get(alternative)), value, abs_tol=1e-8)
        result.append(dict(name=NAMES[i], counts=[tp, fp, fn], IoU=values[0],
                           precision=values[1], recall=values[2], precision_undefined=tp + fp == 0))
    return result


def raw_classes(v):
    return [dict(name=n, counts=v['counts'][i], IoU=v['IoU_percent'][i],
                 precision=v['precision_percent'][i], recall=v['recall_percent'][i])
            for i, n in enumerate(NAMES)]


def fmt(value):
    return '—' if value is None else f'{value:.6f}'


def link(path):
    return '[' + path.name + '](<' + path.as_posix() + '>)'


def main():
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    reference = read(ROOT / 'sacd_execution/val_label_reference.json')
    gt = reference['ground_truth_positive_pixels']
    baseline = read(ROOT / 'training_execution/historical_checkpoint_comparison.json')['baseline_history']
    base_at = {x['step']: x['mIoU'] for x in baseline}
    all_runs = {}
    for name, (directory, description) in RUNS.items():
        source = ROOT / directory / 'validation_series.json'
        series = read(source, {'records': []})
        rows = []
        for row in series['records']:
            classes = metrics(row['classes'], gt)
            mean = sum(x['IoU'] for x in classes) / 4
            assert math.isclose(mean, row.get('mIoU', row.get('mIoU_percent')), abs_tol=1e-8)
            base = row.get('base_head_mIoU')
            if row.get('base_head_metrics'):
                bc = metrics(raw_classes(row['base_head_metrics']), gt)
                assert math.isclose(sum(x['IoU'] for x in bc) / 4, base, abs_tol=1e-8)
            rows.append(dict(step=row['step'], mIoU=mean, classes=classes, base_head_mIoU=base,
                             versus_baseline_pp=mean - base_at[row['step']] if row['step'] in base_at else None))
        all_runs[name] = dict(description=description, source=str(source), records=rows)
    current = all_runs['iou']['records']
    outcome = read(RECEIPTS / 'iou/reports/iou/GPU_CAMPAIGN_OUTCOME.json')
    complete = read(RECEIPTS / 'iou/training_runs/iou/TRAINING_COMPLETE.json')
    final = read(RECEIPTS / 'iou/training_runs/iou/val_40000.json')
    test = read(RECEIPTS / 'iou/training_runs/iou/test_40000.json')
    audit = read(ROOT / 'iou_execution/FINAL_OUTCOME_AUDIT.json')
    shutdown = read(HERE / 'shutdown_observation.json')
    closeout = read(HERE / 'closeout_receipt.json')
    closeout_audit = read(HERE / 'EXPERIMENT_CLOSEOUT_AUDIT.json')
    terminal = bool(outcome or closeout)
    goal = bool(audit and audit.get('status') == 'GOAL_VERIFIED' and audit.get('goal_achieved'))
    status = '双目标经实测和审计达到' if goal else '本轮已结束，双目标未达到' if terminal else '训练中；最终结果待填，不能宣称达标'
    result = dict(generated_at=now, report_final=terminal, status=status, goal_achieved=goal,
                  baseline=baseline, runs=all_runs, current_outcome=outcome,
                  current_completion=complete, current_test=test, scientific_audit=audit, closeout=closeout,
                  experiment_closeout_audit=closeout_audit,
                  shutdown_observation=shutdown)
    write(HERE / 'report_data.json', result)
    with (HERE / 'validation_metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['run', 'step', 'mIoU_percent', 'base_mIoU_percent', 'class', 'IoU_percent',
                         'precision_percent', 'recall_percent', 'TP', 'FP', 'FN', 'precision_undefined'])
        for run, data in all_runs.items():
            for row in data['records']:
                for cls in row['classes']:
                    writer.writerow([run, row['step'], row['mIoU'], row['base_head_mIoU'], cls['name'],
                                     cls['IoU'], cls['precision'], cls['recall'], *cls['counts'], cls['precision_undefined']])
    log = RECEIPTS / 'iou/reports/iou/training.log'
    loss_rows = {}
    if log.exists():
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                row = json.loads(line)
                if 'completed' in row and 'losses' in row:
                    loss_rows[row['completed']] = row
            except ValueError:
                pass
    loss_keys = sorted({k for row in loss_rows.values() for k in row['losses']})
    with (HERE / 'iou_training_loss.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'lr', 'seconds', *loss_keys])
        for step, row in sorted(loss_rows.items()):
            writer.writerow([step, row.get('lr'), row.get('seconds'), *[row['losses'].get(k) for k in loss_keys]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot([x['step'] for x in baseline], [x['mIoU'] for x in baseline], 'k--o', label='Original HRDecoder')
    for run, data in all_runs.items():
        if data['records']:
            ax.plot([x['step'] for x in data['records']], [x['mIoU'] for x in data['records']],
                    marker='o', label=run, linewidth=2.5 if run == 'iou' else 1.2)
    ax.axhline(VAL_TARGET, color='red', linestyle=':', label='Fusion val target')
    ax.set(xlabel='Optimizer updates', ylabel='Full 76-image val mIoU (%)', title='Observed validation checkpoints (no interpolation)')
    ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8); fig.tight_layout()
    fig.savefig(HERE / 'validation_comparison.png', dpi=180); plt.close(fig)
    if loss_rows:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for key in ('task_loss', 'gt_final', 'gt_base', 'kl_final'):
            points = [(s, r['losses'][key]) for s, r in sorted(loss_rows.items()) if key in r['losses']]
            if points: ax.plot(*zip(*points), label=key, alpha=.8)
        ax.set(xlabel='Optimizer updates', ylabel='Logged loss', title='IoUCorrection training losses (raw logged points)')
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(HERE / 'iou_training_loss.png', dpi=180); plt.close(fig)
    lines = ['# HRDecoder完整实验报告', '', f'生成时间：{now}。状态：**{status}**。', '',
             '本报告仅整理已归档的小型原始回执与日志。当前轮未结束时属于待补全报告，终态重新生成。', '',
             '## 1. 目标、协议与结果', '',
             f'目标为单模型固定40k：全76图val mIoU严格超过{VAL_TARGET:.12f}%，通过后锁定同一40k模型，单次custom227图test严格超过{TEST_TARGET:.12f}%。', '',
             '固定2048×2048、batch2、40,000优化更新；fresh ImageNet W48、seed42、私有分支seed20261008。完整val每4k一次；Expert另有2k筛查探针。固定40k为主结果，best-val仅作次要结果。', '',
             '|项目|实测/状态|', '|---|---|',
             f'|本轮终态|{outcome.get("status") if outcome else closeout.get("terminal_status") if closeout else "待产生"}|',
             f'|固定40k完成回执|{complete.get("completed") if complete else "未取得"}|',
             f'|固定40k val mIoU|{fmt(final["mIoU_percent"] if final else None)}|',
             f'|锁定40k test mIoU|{fmt(test["mIoU_percent"] if test else None)}|',
             f'|本次目标结论|{closeout_audit.get("status") if closeout_audit else audit.get("status") if audit else "待终态审计"}|',
             f'|停止及收尾合规核验|{closeout_audit.get("experiment_closeout_status") if closeout_audit else "待核验"}|',
             f'|关机观察|{shutdown.get("status") if shutdown else "待训练终态归档后执行"}|', '',
             '最新用户指定IoUCorrection为最后一版：沿用已登记门槛，筛查不通过就停止、不再新建架构；通过才同一fresh继续40k。终态先保存本地报告所需日志/指标，再关机。没有固定40k和锁定test结果就不能声称完成双目标。', '',
             '## 2. 数据、验证口径与证据边界', '',
             '学生使用已审计的自定义454/76/227划分，学生各split的图像SID/内容身份互斥。历史教师与官方val/test存在交叠，因此不把这项自定义test宣称为官方教师数据独立测试，也不宣称该数据从未在历史研究中使用。', '',
             '四类顺序EX/HE/SE/MA，逐类累计TP/FP/FN，IoU=TP/(TP+FP+FN)，precision=TP/(TP+FP)，recall=TP/(TP+FN)，mIoU为四类IoU等权均值，预测阈值严格>0.5。所有本报告学生验证点从counts重新计算，并与独立76图GT阳性像素计数核对。无预测阳性时precision数学未定义，脚本记0并标注。', '',
             '训练教师与GT构造的T*只进入训练损失；验证/test不输入GT、教师，不用TTA或在线H4。test不参与架构、阈值或权重选择；40k val未过目标则按协议不测test。原HRDecoder历史mIoU来源是已核对同76图标签的原日志，历史值按原日志精度保留，不能与test混比。', '',
             '## 3. 各轮架构与筛查结论', '', '|轮次|关键改动/安排|最后完整val|best-val|实际40k|', '|---|---|---|---|---|']
    for run, data in all_runs.items():
        rows = data['records']; best = max(rows, key=lambda x: x['mIoU']) if rows else None
        last = rows[-1] if rows else None; endpoint = next((x for x in rows if x['step'] == 40000), None)
        lines.append(f'|{run}|{data["description"]}|{str(last["step"])+" / "+fmt(last["mIoU"]) if last else "暂无"}|{fmt(best["mIoU"])+"@"+str(best["step"]) if best else "—"}|{fmt(endpoint["mIoU"]) if endpoint else "未完成40k"}|')
    lines += ['', '这些不同轮次的早期结果不是同预算最终配对对照。短筛查只能拒绝已登记风险条件，不能确保40k或test必达目标。未出现的同节点不插值为实测。', '',
              '筛查回执索引及逐项判定：', '']
    for run, (directory, _) in RUNS.items():
        candidates = list((RECEIPTS / run / f'training_runs/{run}').glob('SCREEN_*.json'))
        for p in sorted(candidates):
            d = read(p)
            if isinstance(d, dict):
                decisions = d.get('decisions', {})
                failed = [k for k, value in decisions.items() if not value]
                lines.append(f'- {run}：{d.get("status", p.stem)}；未过条件：{", ".join(failed) or "回执未列失败条件"}。来源：{link(p)}。')
    if outcome and outcome.get('fresh_screen_rejection'):
        gate = outcome['fresh_screen_rejection']
        lines += ['', '本次最终轮明确在4k止步：', '',
                  '|门槛|要求|实测|结论|', '|---|---|---|---|',
                  f'|全76图mIoU|≥30.475812%|{gate["mIoU_4k"]:.6f}%|未过|',
                  f'|final−base|≥1pp|{gate["private_gain_pp"]:.6f}pp|未过|',
                  '|各类IoU相对base|不得下降超过2pp|EX −2.431492、HE −7.832486、SE −9.754134、MA −4.978461pp|未过|',
                  '|四类非零|全部非零|全部非零|通过|',
                  '|HE/MA FN/FP权衡|原登记组合条件|FN增加时FP显著减少|通过|', '',
                  '这属于门槛拒绝后的正常退出（exit0），没有恢复或新建fresh，没有40k完成标志，也未访问test。4k权重保存记录为saved_update=4000，权重未下载；没有40k持久检查点加载回执。', '']
    lines += ['', '残差8k从4k上升7.736538pp，只因私有收益+0.278182pp低于其原登记0.5pp门槛被拒绝，不能描述为持续退化。Adapter只差绝对floor约0.031606pp，旧门槛未放宽。Foundation私有收益为负且MA相对base跌超2pp；Aligned私有收益为负且SE跌超2pp；Semantic只有私有收益≥1pp未过，HE/MA仍有收益，不能称全类别退化。', '',
              '## 4. 全部完整val节点与原始计数', '',
              '以下P/R为百分比；计数顺序TP/FP/FN，Δ为相邻已测节点mIoU百分点变化。每轮基线、原HRDecoder同一步差值均按实测节点计算。', '']
    for run, data in all_runs.items():
        lines += [f'### {run}', '', f'账本：{link(Path(data["source"]))}。', '',
                  '|step|mIoU|ΔmIoU pp|vs原HRDecoder pp|base mIoU|final−base pp|', '|---:|---:|---:|---:|---:|---:|']
        previous = None
        for row in data['records']:
            gain = row['mIoU'] - row['base_head_mIoU'] if row['base_head_mIoU'] is not None else None
            lines.append(f'|{row["step"]}|{fmt(row["mIoU"])}|{fmt(row["mIoU"]-previous["mIoU"] if previous else None)}|{fmt(row["versus_baseline_pp"])}|{fmt(row["base_head_mIoU"])}|{fmt(gain)}|')
            previous = row
        lines += ['', '|step|类别|IoU|precision|recall|TP/FP/FN|', '|---:|---|---:|---:|---:|---|']
        for row in data['records']:
            for cls in row['classes']:
                pr = fmt(cls['precision']) + ('（分母0，约定值）' if cls['precision_undefined'] else '')
                lines.append(f'|{row["step"]}|{cls["name"]}|{fmt(cls["IoU"])}|{pr}|{fmt(cls["recall"])}|{"/".join(map(str,cls["counts"]))}|')
        lines.append('')
    lines += ['### 暖启动配对诊断（独立列示，不计为fresh训练节点）', '']
    warm = read(ROOT / 'sacd_micro_execution/screening_series.json', {})
    normalization = read(ROOT / 'normalization_execution/paired_series.json', {})
    v10 = read(ROOT / 'research_v10/validated_probe_series.json', {})
    diagnostic_sets = [
        ('SACD父20k：control / feature_isolated', warm.get('warm_val', [])),
        ('Residual父8k：归一化配对诊断', [row for rows in normalization.get('arms', {}).values() for row in rows]),
        ('research_v10配对一致性修复（未完成）', [v for row in v10.get('records', []) for v in row.get('arms', {}).values()]),
    ]
    diagnostics = []
    for name, vals in diagnostic_sets:
        lines += [f'#### {name}', '', '|arm|诊断更新|mIoU|final−base pp|EX IoU/P/R|HE IoU/P/R|SE IoU/P/R|MA IoU/P/R|',
                  '|---|---:|---:|---:|---|---|---|---|']
        for val in sorted(vals, key=lambda x: (x.get('probe_updates', 0), x['arm'])):
            cls = metrics(raw_classes(val), gt)
            mean = sum(c['IoU'] for c in cls) / 4
            assert math.isclose(mean, val['mIoU_percent'], abs_tol=1e-8)
            base = val.get('base_head_metrics')
            if base:
                bc = metrics(raw_classes(base), gt)
                assert math.isclose(sum(x['IoU'] for x in bc)/4, base['mIoU_percent'], abs_tol=1e-8)
            gain = mean-base['mIoU_percent'] if base else None
            cells = ['/'.join(fmt(c[k]) for k in ('IoU','precision','recall')) for c in cls]
            lines.append(f'|{val["arm"]}|{val.get("probe_updates",0)}|{fmt(mean)}|{fmt(gain)}|'+ '|'.join(cells)+'|')
            diagnostics.append(dict(diagnostic=name, parent_step=val.get('parent_completed'), arm=val['arm'],
                                    probe_updates=val.get('probe_updates',0), mIoU=mean, classes=cls))
        lines.append('')
    lines += ['原归一化配对诊断实际具有0/250/500点：实验组相对control分别+0.002436/+3.381834/+1.632108pp；500点实验组final−base仍为−0.112553pp。0点counts不完全相同，不能作严格单变量确定性因果证明，更不能自动激活fresh40k延期。之后research_v10修复0点counts完全一致，但只有128配对更新、缺失250/500，不能把前一诊断完成情况套用到修复试验。', '']
    result['warm_diagnostics'] = diagnostics
    write(HERE / 'report_data.json', result)
    with (HERE / 'warm_diagnostic_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(['diagnostic','parent_step','arm','probe_updates','mIoU_percent','class','IoU','precision','recall','TP','FP','FN'])
        for row in diagnostics:
            for c in row['classes']:writer.writerow([row['diagnostic'],row['parent_step'],row['arm'],row['probe_updates'],row['mIoU'],c['name'],c['IoU'],c['precision'],c['recall'],*c['counts']])
    lines += ['## 5. 趋势、退化诊断与改动依据', '',
              'CET固定40k为38.313759%，相对原HRDecoder40k 38.835%低0.521241pp；CET best为40.487643%@32k，不能用best替代固定40k。SACD16k到20k回升5.712220pp至37.184798%，原HRDecoder16k到20k也曾下降5.825pp，因此一般单点回落不证明永久退化。', '',
              'Detail从12k 36.735849%跌至16k 13.074770%（−23.661079pp），20k仍15.203819%；EX/SE出现类别崩塌。这与普通节点波动不同。用户指令后停止，最后日志21850、持久21000与best12000保留。后续加入结果前登记的严重崩塌保护，保存检查点后立即停止，避免继续等待另一4k。', '',
              '已确认的实现问题：全局梯度裁剪会把已detach的base/private在优化尺度上重新耦合。Semantic起改为分别L2clip5，CPU实际SGD配对验证私有梯度放大不改变基础参数更新。此实现事实不等于已测出其对泛化的独立贡献。', '',
              '恢复困难证据：旧Detail的训练阳性logit饱和，Dice阳性梯度接近消失；BN四组交叉与当前批统计未救回EX/SE。它支持阳性监督/恢复能力不足的解释，但最初触发原因未唯一锁定。Foundation加入基础GT平衡BCE和GT一致教师监督；Aligned把私有BCE/KL也改为按全GT正负计数固定0.5/0.5归一化，禁止trusted子集重归一化。', '',
              '语义/空间无SGD探针显示旧残差能同时增加TP与FP，零语义是分布外输入，不能据此判定语义无效。Semantic采用多尺度分别投影后拼接及base原始logit直接旁路，和分支裁剪同时改变，因此不能宣称单因素因果。', '',
              'Semantic4k的EX/SE新增FP多于TP收益；8个训练batch、0次SGD的已完成logit梯度探针显示SE基础FP处旧私有损失梯度绝对总量约0.021504，Lovasz约1.310746（约60.95倍），且保留FN方向。不同损失尺度、logit空间并非参数空间，此证据只支持下一损失假设，不能证明唯一因果或确保泛化提高。', '',
              'IoUCorrection只更换私有Dice为per-image/class binary Lovasz hinge，保留Semantic网络及其余训练项。严格positive-margin排序包含全部validGT阳性总数，非top-k；CPU与dense原式value/gradient、空类/ignore/排列/饱和以及优化隔离共19项通过；真实2048/batch2非零辅助12步GPU smoke通过。方法依据：[作者官方实现](https://github.com/bermanmaxim/LovaszSoftmax)。', '',
              '未完成或不足的诊断不能算成功证据：重构诊断仅实际10个训练batch、无SGD，负余弦均值−0.619592，没有454批完成回执；research_v10配对诊断只有0点完整val和128配对更新，250/500回执缺失、进程消失原因未知。暖feature-isolated虽500点有+2.494001pp配对收益，但初始counts不一致、250点负收益、HE recall损失约5.077932pp，按预登记门槛拒绝，不能宣称通过fresh筛查。', '',
              '## 6. 性能、日志与训练曲线', '',
              '缓存约96GiB由本任务RAM owner复用；供数等待的启动观测约0.000047秒，GPU启动核验100%/322.77W。这是观测窗口，不能宣称全程无抖动或功耗恒定。固定batch2不因显存占用改变实验协议；显存占满本身不是速度目标。', '',
              'IoU非零辅助GPU smoke均值1.160543秒/步；1500步前lambda_K=0，早期速度不能独立外推完整训练。正式1500后稳态及后续窗口才用于修正ETA。日志只记录部分步，损失CSV/图保留实际记录，不捏造逐步loss。Lovasz与Dice损失尺度不同，跨架构loss值不能直接比较精度。', '',
              f'![完整val曲线](<{(HERE/"validation_comparison.png").as_posix()}>)', '',
              f'损失CSV：{link(HERE/"iou_training_loss.csv")}；完整验证CSV：{link(HERE/"validation_metrics.csv")}。', '']
    analysis = ROOT / 'iou_execution/趋势分析记录.md'
    if analysis.exists():
        text = analysis.read_text(encoding='utf-8').splitlines()
        lines += ['### 最后一版实测结论', ''] + text[1:] + ['']
    if outcome:
        duration = outcome['time_unix']-outcome['start_unix']
        perf = read(RECEIPTS/'iou/training_runs/iou/performance.json')
        lines += [f'最后一轮从启动到控制器终态共{duration/3600:.6f}小时（约{int(duration//3600)}小时{int(duration%3600//60)}分{int(duration%60)}秒）。最终100批窗口平均{perf["mean_batch_seconds"]:.6f}秒/步，供数等待{perf["mean_data_wait_seconds"]:.8f}秒。完整76图验证耗时38.505830秒。日志CSV实际含{len(loss_rows)}个记录点，不补造未记录步。', '']
    if loss_rows: lines += [f'![IoU训练损失](<{(HERE/"iou_training_loss.png").as_posix()}>)', '']
    lines += ['## 7. 收尾、局限与可复核资料', '',
              '当前候选未过筛查时即结束；通过筛查仍需固定40k与独立锁定test双验证，微型测试不能保证达标。各候选均主要为单seed验证，没有独立多seed置信区间；连续使用同一val筛架构会带来选择偏倚，不能把一次无对照运行当成机制因果证明。', '',
              '已登记的筛查门槛是节约计算的规则，不是保证最终优于融合的统计定理。部分候选仅因很小的私有收益/floor差被拒绝，不能等价推出它们继续训练必无希望。用户最后要求遵守原门槛，当前轮未过则终止，不再新建候选。', '',
              'checkpoint保留在服务器持久数据盘，不下载；本地仅归档日志、协议、counts、CPU/GPU验证、控制器终态、checkpoint加载验证回执、test或跳过原因。整机关闭会结束该服务器上的活动；不单独操作另一AI诊断进程。SSH断开只能记录连接观察，不直接等价于云平台账单停止，除非有平台状态证据。', '',
              f'最新授权：{link(ROOT/"iou_execution/LAST_RUN_POLICY.json")}。',
              f'原始回执目录：[{RECEIPTS.name}](<{RECEIPTS.as_posix()}>)；报告数据：{link(HERE/"report_data.json")}。', '',
              '后续建议仅供本地研究：针对类别崩塌预登记更早的训练梯度/阳性召回保护；用严格同父状态的参数空间梯度及小规模配对更新验证损失假设；在未来获授权的数据/计算窗口内采用多个seed和一次锁定独立test确认效果。本次不再启动这些实验。', '']
    if closeout_audit:
        lines += ['### 本次研究的反思与审计解释', '',
                  '本次没有找到达到双目标的架构。较大的train-only logit梯度、正确的detach路由以及CPU/GPU数值门槛只说明实现/训练可执行，不能替代参数空间优化效果与完整val。最后一版虽然减少误检，却以大幅降低召回为代价，未验证原损失替换假设。', '',
                  '早期筛查规则的最终预测能力没有被独立验证；按规则拒绝候选，不代表证明其继续40k必定失败。Residual和Adapter就是应保留这一区别的实例。筛查应预登记、结果前固定，但也应在未来研究中独立评价它与最终性能的关联，而不是把它视为保胜规则。', '',
                  '严重崩塌的计算保护引入偏晚：Detail在16k完整val出现约23.66pp下降后仍继续至用户要求停止，最后日志为21850。这浪费了恢复前景不足阶段的计算。后续才增加保存即停的严重崩塌保护。供数/配对一致性故障与缺失诊断回执也说明CPU验证不能替代真实数据完整交接验证；失败尝试的计算不能归零或从报告中删去。', '',
                  '原FINAL_OUTCOME_AUDIT是完整40k成功链审计，因本轮实测筛查失败、终态为SCREEN_REJECTED而返回CONTRADICTED；它拒绝把本次写成成功。这个状态不表示本报告原始counts错误。另行收尾审计已核对source身份、原门槛、结果前封存、完整76counts、正常拒绝、进程结束及关机连接观察，结论为GOAL_NOT_ACHIEVED且收尾合规。两份原始回执均保留。', '',
                  f'合规收尾与目标结论：{link(HERE/"EXPERIMENT_CLOSEOUT_AUDIT.json")}；原成功链审计：{link(ROOT/"iou_execution/FINAL_OUTCOME_AUDIT.json")}。', '',
                  '没有再次调整结构、loss或阈值，没有重开训练，也没有为了获得正面结论访问test。iou4k及主hrdecoder-4监控均已删除，服务器不再用于本任务。', '']
    (HERE / 'HRDecoder完整实验报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(status='FINAL_REPORT_BUILT' if terminal else 'PENDING_REPORT_BUILT',
                          report=str(HERE/'HRDecoder完整实验报告.md'), records=sum(len(x['records']) for x in all_runs.values()),
                          counts_independently_verified=True, goal_achieved=goal), ensure_ascii=False))


if __name__ == '__main__':
    main()
