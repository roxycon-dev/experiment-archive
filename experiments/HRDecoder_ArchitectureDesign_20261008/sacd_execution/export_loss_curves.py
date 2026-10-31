"""Export logged SACD batches locally; no SSH, GPU use, or training changes."""
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'cache_execution/server_receipts/sacd/reports/sacd/sacd_training.log'
KEYS = ('gt_final', 'gt_base', 'gt_hr_crop', 'kl_final', 'kl_base',
        'action_kl', 'lambda_k', 'lambda_action', 'task_loss', 'weighted_action_loss')
samples = {}
for line in SOURCE.read_text(encoding='utf-8', errors='replace').splitlines():
    try:
        row = json.loads(line)
    except (ValueError, TypeError):
        continue
    if not isinstance(row, dict) or not isinstance(row.get('completed'), int):
        continue
    values = row.get('losses', {})
    if not all(k in values for k in KEYS):
        continue
    assert all(math.isfinite(values[k]) and values[k] >= 0 for k in KEYS)
    expected_task = (.75*values['gt_final'] + .25*values['gt_base'] + .1*values['gt_hr_crop']
                     + values['lambda_k']*(values['kl_final'] + .25*values['kl_base']))
    assert math.isclose(values['task_loss'], expected_task, abs_tol=1e-6)
    assert math.isclose(values['lambda_action'], .25*values['lambda_k'], abs_tol=1e-8)
    assert math.isclose(values['weighted_action_loss'], values['lambda_action']*values['action_kl'],
                        abs_tol=1e-6)
    if row['completed'] in samples:
        assert samples[row['completed']] == row, 'Conflicting duplicate update.'
    samples[row['completed']] = row
rows = [samples[s] for s in sorted(samples)]
assert len(rows) >= 20, 'At least 20 recorded batches are required.'
steps = np.array([r['completed'] for r in rows])
first, last = int(steps[0]), int(steps[-1])
assert steps.tolist() == [1] + list(range(50, last+1, 50)), 'Unexpected missing log samples.'
window = 20
raw = {k: np.array([r['losses'][k] for r in rows]) for k in KEYS}
smooth = {k: np.convolve(v, np.ones(window)/window, mode='valid') for k, v in raw.items()}
stamp = datetime.now(timezone(timedelta(hours=8)))
destination = HERE / 'loss_exports' / f'sacd_losses_u{first:05d}_u{last:05d}'
destination.mkdir(parents=True, exist_ok=True)
font = Path(r'C:\Windows\Fonts\msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path',
    'text.color': '#1E293B', 'axes.labelcolor': '#334155', 'axes.edgecolor': '#CBD5E1',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(2, 3, figsize=(16.2, 10.2))
fig.subplots_adjust(left=.065, right=.97, top=.80, bottom=.16, wspace=.27, hspace=.43)
fig.suptitle('SACD · 训练日志损失曲线', x=.065, y=.962, ha='left', fontsize=23, fontweight='bold')
fig.text(.065, .916, f'归档实测：{first:,}–{last:,}步 · {len(rows)}个批次采样 · 2048 / batch 2',
         fontsize=12, color='#64748B')
fig.text(.065, .875, '细线：原始批次值；粗线：20个日志点尾随均值（约1k步）。每项分别展示。',
         fontsize=11, color='#64748B')
panels = [
    ('主任务损失', [('task_loss', '主任务', '#2563EB')]),
    ('完整图 GT 损失', [('gt_final', '最终输出', '#2563EB'), ('gt_base', '基础输出', '#D97706')]),
    ('高分辨率裁剪 GT 损失', [('gt_hr_crop', 'HR 裁剪', '#0D9488')]),
    ('蒸馏 KL（未乘权重）', [('kl_final', '最终输出', '#7C3AED'), ('kl_base', '基础输出', '#DB2777')]),
    ('动作 KL（未乘权重）', [('action_kl', '动作 KL', '#D97706')]),
    ('动作损失（已乘权重，仅私有模块）', [('weighted_action_loss', '加权动作', '#0D9488')]),
]
for ax, (title, series) in zip(axes.flat, panels):
    for key, label, color in series:
        ax.plot(steps/1000, raw[key], color=color, alpha=.22, lw=.7)
        ax.plot(steps[window-1:]/1000, smooth[key], color=color, lw=2.1, label=label)
    for boundary in (1500, 4000):
        if boundary <= last:
            ax.axvline(boundary/1000, color='#94A3B8', alpha=.4, lw=.8, ls='--')
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('已完成 SGD 更新（千步）')
    ax.set_ylabel('日志损失值')
    ax.set_xlim(first/1000, last/1000)
    ax.set_ylim(bottom=0)
    ax.grid(True, color='#E2E8F0', lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
fig.text(.065, .093, '虚线：1.5k后蒸馏/动作权重开始爬升，4k后达到平台；加权动作损失早期为0来自方案设置。',
         fontsize=10, color='#64748B')
fig.text(.065, .053, '训练损失不能代替验证精度。动作辅助梯度仅回传私有模块，未构造统一总损失。',
         fontsize=10, color='#64748B')
png, svg = destination/'损失曲线_高清.png', destination/'损失曲线_矢量.svg'
fig.savefig(png, dpi=210)
fig.savefig(svg)
plt.close(fig)
data = {'source_local_log': str(SOURCE), 'export_time': stamp.isoformat(timespec='seconds'),
    'first_update': first, 'last_update': last, 'sample_count': len(rows),
    'recorded_steps': steps.tolist(), 'rolling_window_samples': window,
    'rolling_mean_updates': steps[window-1:].tolist(),
    'rolling_means': {k: v.tolist() for k, v in smooth.items()},
    'raw_training_records': rows, 'loss_formula_verified': True,
    'rolling_mean_rule': 'Trailing arithmetic mean of 20 logged batches, not every SGD batch.',
    'validation_metrics_included': False}
(destination/'损失原始数据与滚动均值.json').write_text(
    json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
shutil.copyfile(SOURCE, destination/'原始训练日志.log')
(destination/'导出说明.md').write_text(
    f'# SACD 日志损失曲线\n\n导出实测{first}–{last}步，共{len(rows)}个记录批次。'
    '第1步及其后每50步采样；没有补绘未记录的批次。粗线为20个日志点的尾随均值。\n\n'
    '主任务 = 0.75×最终GT + 0.25×基础GT + 0.1×HR裁剪GT + '
    'λK×(最终KL + 0.25×基础KL)。加权动作 = 0.25×λK×动作KL。'
    '逐点核对日志中的公式一致性。主任务与仅作用于私有模块的动作辅助分别展示。\n\n'
    '所有坐标均为线性轴，保留早期零值。训练损失不能用于代替完整验证集指标。'
    '本导出只读取本地归档日志，不连接服务器或改变训练。\n', encoding='utf-8')
print(json.dumps({'PNG': str(png), 'SVG': str(svg), 'samples': len(rows),
                  'last_update': last, 'loss_formula_verified': True}, ensure_ascii=False))
