"""Plot measured validation and sampled training losses from local receipts only."""
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RECEIPTS = ROOT / 'cache_execution/server_receipts'
OUT = HERE / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
font = Path(r'C:\Windows\Fonts\msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({
    'axes.unicode_minus': False, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#BAC4D0', 'axes.labelcolor': '#334155',
    'text.color': '#1E293B', 'xtick.color': '#475569', 'ytick.color': '#475569',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    'svg.fonttype': 'path',
})

series = json.loads((HERE / 'validation_series.json').read_text(encoding='utf-8'))
records = series['records']
assert records and all(r['counts_verified'] for r in records)
for record in records:
    for c in record['classes']:
        tp, fp, fn = c['counts']
        assert abs(c['IoU_percent'] - 100 * tp / (tp + fp + fn)) < 1e-8
    assert abs(record['mIoU_percent'] - sum(c['IoU_percent'] for c in record['classes']) / 4) < 1e-8

snapshot = json.loads((RECEIPTS / 'GPU_CURRENT_SNAPSHOT.json').read_text(encoding='utf-8'))
stamp = datetime.fromtimestamp(snapshot['time_unix'], timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M')
loss_by_step = {}
log = RECEIPTS / 'reports/v3_training.log'
for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
    try:
        row = json.loads(line)
    except (ValueError, TypeError):
        continue
    if isinstance(row, dict) and isinstance(row.get('losses'), dict) and isinstance(row.get('completed'), int):
        assert all(math.isfinite(row['losses'][key]) for key in ('dice', 'hr_dice', 'kl', 'signed_aux'))
        loss_by_step[row['completed']] = row
run_start = snapshot['receipts']['reports/GPU_CAMPAIGN_RUNNING.json']['start_unix']
previous_data_path = OUT / 'curve_data.json'
reused_previous_samples = False
if previous_data_path.exists():
    previous_data = json.loads(previous_data_path.read_text(encoding='utf-8'))
    previous_rows = {r['completed']: r for r in previous_data.get('training_samples', [])}
    overlap = set(previous_rows) & set(loss_by_step)
    same_run = previous_data.get('training_run_start_unix') in (None, run_start)
    if same_run and overlap and all(previous_rows[s] == loss_by_step[s] for s in overlap):
        # A downloaded tail may omit earlier records. Keep only verified matching history.
        loss_by_step = {**previous_rows, **loss_by_step}
        reused_previous_samples = True
loss_rows = [loss_by_step[s] for s in sorted(loss_by_step)]
assert loss_rows

def style(ax, title, ylabel):
    ax.set_title(title, loc='left', fontsize=13, fontweight='bold', pad=14)
    ax.set_xlabel('已完成 SGD 更新（千步）', labelpad=9)
    ax.set_ylabel(ylabel, labelpad=9)
    ax.grid(True, color='#E2E8F0', linewidth=0.7)
    ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=150)
    fig.savefig(OUT / (name + '.svg'))
    plt.close(fig)

x = np.array([r['step'] / 1000 for r in records])
y = np.array([r['mIoU_percent'] for r in records])
fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
fig.subplots_adjust(left=0.065, right=0.975, bottom=0.22, top=0.76, wspace=0.24)
fig.suptitle(f'HRDecoder v3 · 验证曲线（截至 {records[-1]["step"] // 1000}k）',
             x=0.065, y=0.965, ha='left', fontsize=19, fontweight='bold')
fig.text(0.065, 0.89, f'全76张验证图 · 2048 / batch 2 · 最新归档：{stamp}（北京时间）', fontsize=11, color='#64748B')
ax = axes[0]
style(ax, '验证 mIoU 与历史同期', 'mIoU（%）')
ax.plot(x, y, '-o', color='#2563EB', lw=2.5, ms=7, label='本轮 v3')
historical = [r for r in records if r.get('historical_baseline_mIoU_percent') is not None]
hx = [r['step'] / 1000 for r in historical]
hy = [r['historical_baseline_mIoU_percent'] for r in historical]
ax.plot(hx, hy, '--s', color='#64748B', lw=1.8, ms=5, label='历史原 HRDecoder')
teacher_val = 45.32183122738841
ax.axhline(teacher_val, color='#D97706', linestyle=':', lw=2, label='锁定多模型 val：45.32%')
for xx, yy in zip(x, y):
    baseline_at_x = next((r['historical_baseline_mIoU_percent'] for r in historical if r['step'] / 1000 == xx), None)
    offset = 10 if baseline_at_x is None or yy >= baseline_at_x else -18
    ax.annotate(f'{yy:.2f}', (xx, yy), xytext=(0, offset), textcoords='offset points',
                ha='center', color='#1D4ED8', fontsize=11, fontweight='bold')
for xx, yy in zip(hx, hy):
    student_at_x = next(r['mIoU_percent'] for r in records if r['step'] / 1000 == xx)
    offset = -18 if yy <= student_at_x else 10
    ax.annotate(f'{yy:.2f}', (xx, yy), xytext=(0, offset), textcoords='offset points',
                ha='center', color='#64748B', fontsize=10)
ax.set_ylim(0 if min(y) < 20 else 15, 57)
ax.set_xlim(min(x) - 0.6, max(x) + 0.6)
ax.set_xticks(x)
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', framealpha=1, fontsize=9)
ax = axes[1]
style(ax, '四类验证 IoU', 'IoU（%）')
colors = ['#D97706', '#DC2626', '#0D9488', '#7C3AED']
end_labels = []
for index, (name, color) in enumerate(zip(('EX', 'HE', 'SE', 'MA'), colors)):
    values = [r['classes'][index]['IoU_percent'] for r in records]
    ax.plot(x, values, '-o', color=color, lw=2, ms=6, label=name)
    end_labels.append((values[-1], color))
last_label_y = -float('inf')
for value, color in sorted(end_labels):
    label_y = max(value, last_label_y + 2.0)
    ax.annotate(f'{value:.2f}', (x[-1], value), xytext=(x[-1] + 0.3, label_y),
                va='center', color=color, fontsize=10,
                arrowprops={'arrowstyle': '-', 'color': color, 'lw': 0.6})
    last_label_y = label_y
all_class_ious = [c['IoU_percent'] for r in records for c in r['classes']]
class_ymin = -2 if min(all_class_ious) == 0 else max(0, min(10, min(all_class_ious) - 5))
ax.set_ylim(class_ymin, max(55, last_label_y + 4, max(all_class_ious) + 5))
ax.set_xlim(min(x) - 0.6, max(x) + 1.3)
ax.set_xticks(x)
ax.legend(loc='upper left', frameon=False, ncol=4, fontsize=10)
fig.text(0.065, 0.085, '实心点为已测验证结果，连线仅辅助阅读；没有补绘未来节点。历史成绩作为参照，未重跑配对对照。',
         fontsize=10, color='#64748B')
fig.text(0.065, 0.04, f'当前 best-val：{series["best_val_step"] // 1000}k / {series["best_val_mIoU"]:.2f}%；40k 固定端点尚未完成。',
         fontsize=10, color='#64748B')
save(fig, 'validation_curves')

lx = np.array([r['completed'] / 1000 for r in loss_rows])
window = 20
assert len(loss_rows) >= window
def losses(ax, specs, log_scale=False):
    for key, label, color in specs:
        values = np.array([r['losses'][key] for r in loss_rows])
        if log_scale:
            assert np.all(values > 0), 'Log axis requires strictly positive recorded losses.'
        ax.plot(lx, values, color=color, lw=0.7, alpha=0.19)
        mean = np.convolve(values, np.ones(window) / window, mode='valid')
        ax.plot(lx[window - 1:], mean, color=color, lw=2.2, label=label)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlim(lx[0], lx[-1])
    ax.legend(loc='upper right', frameon=False, fontsize=10)

fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))
fig.subplots_adjust(left=0.065, right=0.975, bottom=0.22, top=0.76, wspace=0.24)
fig.suptitle('HRDecoder v3 · 训练损失曲线', x=0.065, y=0.965, ha='left', fontsize=19, fontweight='bold')
fig.text(0.065, 0.89, f'当前可用日志片段：{loss_rows[0]["completed"]:,}–{loss_rows[-1]["completed"]:,}步 · 最新归档：{stamp}（北京时间）',
         fontsize=11, color='#64748B')
style(axes[0], 'Dice 损失（训练批次）', '损失值')
losses(axes[0], [('dice', 'GT Dice', '#2563EB'), ('hr_dice', 'HR 辅助 Dice', '#0D9488')])
axes[0].set_ylim(0, 1.05)
style(axes[1], '蒸馏与辅助损失（对数轴）', '损失值（对数）')
losses(axes[1], [('kl', 'KL', '#7C3AED'), ('signed_aux', '有符号辅助损失', '#D97706')], log_scale=True)
fig.text(0.065, 0.085, '每50步记录一个训练批次；细线为原始值，粗线为20个记录点的滚动均值（约1k步）。缺失日志不补绘。',
         fontsize=10, color='#64748B')
fig.text(0.065, 0.04, '训练损失不是验证精度；不同损失的数值大小不能直接比较，也未合成为一个总损失。',
         fontsize=10, color='#64748B')
save(fig, 'training_loss_curves')

data = {'snapshot_time_UTC_plus_8': stamp, 'validation_images': 76, 'validation': records,
        'teacher_val_mIoU_percent': teacher_val, 'training_log_source': str(log),
        'training_samples': loss_rows, 'rolling_window_samples': window,
        'training_run_start_unix': run_start, 'reused_verified_previous_samples': reused_previous_samples,
        'training_sample_interval_updates': 50, 'future_values_generated': False}
(OUT / 'curve_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT / '曲线说明.md').write_text(
    '# HRDecoder v3 曲线\n\n'
    f'数据截至北京时间{stamp}的本地归档。验证节点为' + '、'.join(str(r['step']) for r in records) + '步。\n\n'
    '验证图分别显示实测mIoU、历史同节点参照、锁定多模型验证目标和四类IoU。未使用测试成绩，未外推未来节点。\n\n'
    f'训练损失仅覆盖现有日志{loss_rows[0]["completed"]}至{loss_rows[-1]["completed"]}步，共{len(loss_rows)}个日志点。'
    '粗线是20个采样记录的尾随滚动均值，并非对全部SGD批次平均。训练损失不能替代验证指标。\n\n'
    'PNG供聊天预览，SVG供矢量查看，curve_data.json保留绘图数据。运行plot_training_curves.py可根据最新本地归档更新。\n',
    encoding='utf-8')
print(json.dumps({'output': str(OUT), 'validation_steps': [r['step'] for r in records],
                  'training_samples': len(loss_rows), 'training_first_update': loss_rows[0]['completed'],
                  'training_last_update': loss_rows[-1]['completed']}, ensure_ascii=False))
