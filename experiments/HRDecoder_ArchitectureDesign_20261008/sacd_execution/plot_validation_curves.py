"""Plot only verified, measured validation points from local receipts."""
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'plots'
OUT.mkdir(parents=True,exist_ok=True)
series=json.loads((HERE/'validation_series.json').read_text(encoding='utf-8'))
records=series['records']
assert records and all(r['counts_verified'] for r in records)
for record in records:
    assert [c['name'] for c in record['classes']]==['EX','HE','SE','MA']
    for c in record['classes']:
        tp,fp,fn=c['counts']
        for name,denom in [('IoU',tp+fp+fn),('precision',tp+fp),('recall',tp+fn)]:
            assert math.isclose(c[name],100*tp/denom if denom else 0,abs_tol=1e-8)
    assert math.isclose(record['mIoU'],sum(c['IoU'] for c in record['classes'])/4,abs_tol=1e-8)
history=json.loads((ROOT/'training_execution/historical_checkpoint_comparison.json').read_text(encoding='utf-8'))
cet=json.loads((ROOT/'cet_execution/validation_series.json').read_text(encoding='utf-8'))
v3=json.loads((ROOT/'training_execution/validation_series.json').read_text(encoding='utf-8'))
snapshot=json.loads((ROOT/'cache_execution/server_receipts/sacd/latest_snapshot.json').read_text())
latest=records[-1]
stamp=snapshot['checked_at']
font=Path(r'C:\Windows\Fonts\msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,
    'axes.spines.right':False,'svg.fonttype':'path','text.color':'#1E293B',
    'axes.labelcolor':'#334155','figure.facecolor':'white','savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(14.5,6.5))
fig.subplots_adjust(left=.065,right=.97,top=.76,bottom=.23,wspace=.23)
fig.suptitle(f'SACD · 验证曲线（已归档至{latest["step"]//1000}k）',
             x=.065,y=.957,ha='left',fontsize=22,fontweight='bold')
fig.text(.065,.882,'完整76图val · 2048 / batch 2 · 固定40k端点为主结果',fontsize=12,color='#64748B')
fig.text(.065,.831,f'最近归档：{stamp[:19].replace("T"," ")}（北京时间）',fontsize=10,color='#64748B')
ax=axes[0]
comparisons=[('历史原HRDecoder',history['baseline_history'],'mIoU','#94A3B8','s'),
             ('历史CET（已完成40k）',cet['records'],'mIoU','#0D9488','^'),
             ('历史v3（20k后停止）',v3['records'],'mIoU_percent','#A78BFA','x')]
for label,rows,key,color,marker in comparisons:
    ax.plot([r['step']/1000 for r in rows],[r[key] for r in rows],
            color=color,marker=marker,lw=1.3,ms=4,label=label)
ax.axhline(45.32183122738841,color='#D97706',lw=1.8,ls='--',label='锁定融合val目标45.322%')
ax.plot([r['step']/1000 for r in records],[r['mIoU'] for r in records],
        color='#2563EB',lw=2.8,marker='o',ms=8,label='本轮SACD（实测）',zorder=5)
ax.annotate(f'{latest["mIoU"]:.3f}%',(latest['step']/1000,latest['mIoU']),
            xytext=(10,-26),textcoords='offset points',color='#2563EB',fontweight='bold')
ax.set_title('验证mIoU与历史同节点参照',loc='left',fontsize=13,fontweight='bold',pad=14)
ax.set_ylabel('mIoU（%）')
ax.set_ylim(0,53)
ax.set_xlim(2,41)
ax.set_xticks(range(4,41,4))
ax.legend(loc='lower right',frameon=False,fontsize=9)
ax=axes[1]
colors=['#D97706','#DC2626','#0D9488','#7C3AED']
end_labels=[]
for index,(name,color) in enumerate(zip(('EX','HE','SE','MA'),colors)):
    values=[r['classes'][index]['IoU'] for r in records]
    ax.plot([r['step']/1000 for r in records],values,color=color,lw=2,marker='o',ms=6,label=name)
    end_labels.append((values[-1],color))
last_label_y=-float('inf')
for value,color in sorted(end_labels):
    label_y=max(value,last_label_y+3)
    ax.annotate(f'{value:.2f}',(latest['step']/1000,value),
                xytext=(latest['step']/1000+.45,label_y),color=color,va='center',fontsize=10,
                arrowprops={'arrowstyle':'-','color':color,'lw':.6})
    last_label_y=label_y
ax.set_title('本轮四类IoU（仅已测节点）',loc='left',fontsize=13,fontweight='bold',pad=14)
ax.set_ylabel('IoU（%）')
ax.set_ylim(0,max(60,max(c['IoU'] for r in records for c in r['classes'])+8))
xmin=max(0,records[0]['step']/1000-1)
xmax=latest['step']/1000+2
ax.set_xlim(xmin,xmax)
ax.set_xticks([r['step']/1000 for r in records])
ax.legend(loc='upper left',frameon=False,ncol=4,fontsize=10)
for ax in axes:
    ax.set_xlabel('已完成SGD更新（千步）')
    ax.grid(True,color='#E2E8F0',lw=.7)
    ax.set_axisbelow(True)
best=max(records,key=lambda r:r['mIoU'])
fig.text(.065,.13,'点为已归档实测验证结果，连线仅辅助阅读；本轮未归档节点不补绘，也未预测最终成绩。',fontsize=10,color='#64748B')
fig.text(.065,.086,f'本轮best-val：{best["step"]//1000}k / {best["mIoU"]:.3f}%。历史实验仅作参照，未重跑配对对照。',fontsize=10,color='#64748B')
fig.text(.065,.042,'这是val曲线；独立test目标45.860384%不画在此图，不能混比。',fontsize=10,color='#64748B')
png,svg=OUT/'验证曲线_最新.png',OUT/'验证曲线_最新.svg'
fig.savefig(png,dpi=200);fig.savefig(svg);plt.close(fig)
data={'export_time':datetime.now(timezone(timedelta(hours=8))).isoformat(),
      'last_remote_archive_time':stamp,'validation_images':76,'SACD':records,
      'historical_original':history['baseline_history'],'historical_CET':cet['records'],
      'historical_v3':v3['records'],'fusion_val_target':45.32183122738841,
      'future_points_predicted':False,'test_metrics_included':False}
(OUT/'验证曲线原始数据.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'PNG':str(png),'SVG':str(svg),'last_validation_update':latest['step'],
                  'measured_points':len(records)}))
