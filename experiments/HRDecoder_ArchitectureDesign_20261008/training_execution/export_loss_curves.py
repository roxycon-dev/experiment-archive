"""Export four logged losses, raw samples and a documented trailing mean."""
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
SOURCE = HERE / 'loss_exports/v3_training_export.log'
KEYS = ('dice', 'hr_dice', 'kl', 'signed_aux')
samples = {}
for line in SOURCE.read_text(encoding='utf-8', errors='replace').splitlines():
    try:
        row = json.loads(line)
    except (ValueError, TypeError):
        continue
    if not isinstance(row, dict) or not isinstance(row.get('completed'), int):
        continue
    if not isinstance(row.get('losses'), dict) or not all(k in row['losses'] for k in KEYS):
        continue
    assert all(math.isfinite(row['losses'][k]) and row['losses'][k] >= 0 for k in KEYS)
    samples[row['completed']] = row
rows = [samples[s] for s in sorted(samples)]
assert len(rows) >= 20, 'Need at least 20 recorded samples.'
first, last = rows[0]['completed'], rows[-1]['completed']
window = 20
steps = np.array([r['completed'] for r in rows])
intervals = sorted(set(int(v) for v in np.diff(steps)))
assert intervals == [50], ('Unexpected sampling intervals', intervals)
stamp = datetime.fromtimestamp(SOURCE.stat().st_mtime, timezone(timedelta(hours=8)))
destination = HERE / 'loss_exports' / f'v3_losses_u{first:05d}_u{last:05d}'
destination.mkdir(parents=True, exist_ok=True)

font = Path(r'C:\Windows\Fonts\msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({
    'axes.unicode_minus': False, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#CBD5E1', 'text.color': '#1E293B',
    'axes.labelcolor': '#334155', 'xtick.color': '#475569', 'ytick.color': '#475569',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'svg.fonttype': 'path',
})

fig, axes = plt.subplots(2, 2, figsize=(14, 9.3))
fig.subplots_adjust(left=0.075, right=0.965, top=0.79, bottom=0.15, hspace=0.42, wspace=0.23)
fig.suptitle('HRDecoder v3 · 日志损失曲线', x=0.075, y=0.965, ha='left',
             fontsize=22, fontweight='bold')
fig.text(0.075, 0.92, f'实测记录：{first:,}–{last:,}步 · {len(rows)}个训练批次采样 · 2048 / batch 2',
         fontsize=12, color='#64748B')
fig.text(0.075, 0.878, f'日志读取时间：{stamp:%Y-%m-%d %H:%M:%S}（北京时间）',
         fontsize=11, color='#64748B')
specs = [
    ('dice', 'GT Dice 损失', '#2563EB', False),
    ('hr_dice', 'HR 辅助 Dice 损失', '#0D9488', False),
    ('kl', 'KL 蒸馏损失', '#7C3AED', True),
    ('signed_aux', '有符号辅助损失', '#D97706', True),
]
smoothed = {}
omitted_zero_samples = {}
for ax, (key, title, color, use_log) in zip(axes.flat, specs):
    values = np.array([r['losses'][key] for r in rows])
    mean = np.convolve(values, np.ones(window) / window, mode='valid')
    smoothed[key] = mean.tolist()
    omitted_zero_samples[key] = int(np.sum(values == 0)) if use_log else 0
    raw_plot = np.where(values > 0, values, np.nan) if use_log else values
    mean_plot = np.where(mean > 0, mean, np.nan) if use_log else mean
    ax.plot(steps / 1000, raw_plot, color=color, alpha=0.25, lw=0.75, label='原始日志值')
    ax.plot(steps[window - 1:] / 1000, mean_plot, color=color, lw=2.3, label='20个日志点尾随均值')
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', pad=12)
    ax.set_xlabel('已完成 SGD 更新（千步）')
    ax.set_ylabel('损失值（对数轴）' if use_log else '损失值')
    ax.set_xlim(first / 1000, last / 1000)
    if use_log:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, max(1.05, float(values.max()) * 1.05))
    ax.grid(True, color='#E2E8F0', lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False, fontsize=9)

fig.text(0.075, 0.074, '每50步记录一个训练批次；粗线为20个记录点的尾随均值（约1k步），不是全部SGD批次的均值。',
         fontsize=10, color='#64748B')
fig.text(0.075, 0.037, '训练损失不能代替验证精度；四项损失分别展示，未构造总损失。缺失的早期日志不补绘。',
         fontsize=10, color='#64748B')
png = destination / '损失曲线_高清.png'
svg = destination / '损失曲线_矢量.svg'
fig.savefig(png, dpi=240)
fig.savefig(svg)
plt.close(fig)

data = {
    'source_remote_log': '/root/autodl-tmp/HRDecoder_v3_20261008/reports/v3_training.log',
    'log_read_time_UTC_plus_8': stamp.isoformat(timespec='seconds'),
    'first_update': first, 'last_update': last, 'sample_count': len(rows),
    'logged_interval_updates': 50, 'rolling_window_samples': window,
    'rolling_mean_rule': 'Trailing arithmetic mean of 20 logged batches; no fabricated values.',
    'raw_training_records': rows, 'rolling_mean_updates': steps[window - 1:].tolist(),
    'rolling_means': smoothed, 'zero_samples_hidden_on_log_axes': omitted_zero_samples,
    'validation_metrics_included': False,
}
data_path = destination / '损失原始数据与滚动均值.json'
data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
shutil.copyfile(SOURCE, destination / '原始训练日志.log')
(destination / '导出说明.md').write_text(
    '# 损失曲线导出\n\n'
    f'日志读取于北京时间{stamp:%Y-%m-%d %H:%M:%S}，导出实测{first}至{last}步，共{len(rows)}个采样点。\n\n'
    '包含日志中的dice、hr_dice、kl、signed_aux四项损失。PNG为3360×2232像素；SVG为矢量格式。'
    '原始训练日志和JSON数据一并保存，JSON保留lr、lambda_k、lambda_d等日志原有字段。\n\n'
    '每50步记录一个训练批次。原始曲线没有平滑，粗线是20个记录点的尾随算术均值，约覆盖1k步，'
    '不能称为对全部SGD批次的平均。日志首次记录之前不补数据。KL和辅助损失用对数轴，零值如有则不绘制但保留在JSON中。\n\n'
    '这些是训练损失，不是验证指标；没有擅自相加为总损失，也没有修改服务器训练设置。\n', encoding='utf-8')
print(json.dumps({'PNG': str(png), 'SVG': str(svg), 'data': str(data_path),
                  'first_update': first, 'last_update': last, 'samples': len(rows),
                  'zero_samples_hidden_on_log_axes': omitted_zero_samples}, ensure_ascii=False))
