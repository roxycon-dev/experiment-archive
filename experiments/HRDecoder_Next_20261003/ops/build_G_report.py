"""Create a compact scientific G report from already saved counts and histograms."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyze_cache import image_bootstrap
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
g=ROOT/'server_results/work/G';out=ROOT/'reports';out.mkdir(exist_ok=True)
matrix=read(g/'matrix.json');base=read(g/'current_deployment/metrics.json');best=read(g/'highres_fixed25/metrics.json')
calbase=read(ROOT/'server_results/reports/baseline_calibration_OOF.json')
calbest=read(ROOT/'server_results/reports/G_best_calibration_OOF.json')
classes=['EX','HE','SE','MA'];lock=read(ROOT/'server_results/work/geometry_lock.json')
bc={x['sid']:x['counts'] for x in base['per_image']};nc={x['sid']:x['counts'] for x in best['per_image']}
ids=sorted(bc);ci=image_bootstrap([nc[x] for x in ids],[bc[x] for x in ids])
fig,axes=plt.subplots(4,2,figsize=(11,13),layout='constrained')
curves=[('Current complete',base),('G highres fixed25',best),('U1 native whole',matrix['U1_native_whole'])]
for i,c in enumerate(classes):
 for label,report in curves:
  curve=report['curves'][c]
  axes[i,0].plot(curve['recall'],curve['precision'],label=label,lw=1.6)
  axes[i,1].plot(curve['threshold_lower_bound'],curve['iou'],label=label,lw=1.6)
 for j in [0,1]:
  axes[i,j].set_title(c);axes[i,j].grid(alpha=.2);axes[i,j].set_ylim(0,1)
 axes[i,0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1))
 axes[i,1].set(xlabel='Threshold (512-bin diagnostic)',ylabel='IoU',xlim=(.1,.9))
 axes[i,1].axvline(.5,color='black',lw=.8,linestyle='--')
axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
fig.suptitle('Complete validation: PR and threshold diagnostics\nHistogram curves are not deployed scores',fontsize=14)
fig.savefig(out/'G_PR_IoU_diagnostics.png',dpi=160);fig.savefig(out/'G_PR_IoU_diagnostics.pdf');plt.close(fig)
lines=['# G阶段结论：已完成，M进行中','',
 '完整76张val，原始坐标，独立四类，严格概率>0.5，逐类全图累计TP/FP/FN后平均IoU。没有把诊断曲线最佳阈值当部署成绩。',
 '', '|固定权重与几何|单模val %|冠军75%＋专家25% val %|','|---|---:|---:|']
for k,v in matrix.items():
 if 'fixed_25' in v:lines.append(f'|{k}|{100*v["mIoU_lesion4"]:.4f}|{100*v["fixed_25"]["mIoU_lesion4"]:.4f}|')
lines += ['',f'锁定几何后，高分辨率U0@1000的固定25%完整管线为 **{100*best["mIoU_lesion4"]:.4f}%**；旧完整部署为{100*base["mIoU_lesion4"]:.4f}%，增加{100*(best["mIoU_lesion4"]-base["mIoU_lesion4"]):.4f}个百分点。这是val发现，尚不是新test成绩。',
 f'按图配对bootstrap的增量95%区间为[{100*ci["percentile95"][0]:.3f}, {100*ci["percentile95"][1]:.3f}]个百分点；不替代训练种子复核。',
 '',f'U1原方形滑窗单模为{100*matrix["U1_original_square2048_slide"]["mIoU_lesion4"]:.4f}%，原生whole为{100*matrix["U1_native_whole"]["mIoU_lesion4"]:.4f}%。因此旧协议下较弱的成绩确实混入几何影响，但其固定融合仍没有超过最佳G候选。',
 f'H1同一2048方形输入，普通forward为{100*matrix["H1_forward"]["mIoU_lesion4"]:.4f}%，eval_forward为{100*matrix["H1_eval_forward"]["mIoU_lesion4"]:.4f}%，差{100*(matrix["H1_forward"]["mIoU_lesion4"]-matrix["H1_eval_forward"]["mIoU_lesion4"]):.4f}个百分点。此结论限于现有checkpoint与这两条推理路径。',
 '', '缓存补查H1普通前向：冠军75%＋H1 25%为43.8383%；旧完整部署50%＋H1 50%为42.8698%。前者仍低于43.8679%的最佳G，不追加比例扫描或旧H1训练。',
 '', '|类别|旧完整val IoU %|最佳G IoU %|TP变化|FP变化|FN变化|','|---|---:|---:|---:|---:|---:|']
for i,c in enumerate(classes):
 delta=np.asarray(best['counts'][i])-np.asarray(base['counts'][i])
 lines.append(f'|{c}|{100*base["per_class"][c]["iou"]:.4f}|{100*best["per_class"][c]["iou"]:.4f}|{delta[0]:+d}|{delta[1]:+d}|{delta[2]:+d}|')
lines += ['',f'三折val内拟合、折外评价：旧完整从{100*calbase["fixed_05"]["mIoU_lesion4"]:.4f}%变为{100*calbase["calibration_OOF"]["mIoU_lesion4"]:.4f}%；最佳G从{100*calbest["fixed_05"]["mIoU_lesion4"]:.4f}%变为{100*calbest["calibration_OOF"]["mIoU_lesion4"]:.4f}%。增加幅度约0.1—0.24点，不能证明剩余目标主要由校准解决。保持固定0.5候选，进入表示融合主实验。',
 '', '[PR/IoU诊断图](G_PR_IoU_diagnostics.png)；[24张实际误差选样的图库](../server_results/reports/G_error_gallery/index.html)；[逐图视觉记录](G_manual_review.csv)。图中叠色显示候选全部TP/FP/FN，人工分类描述视觉环境，不能把所有红色都称为新增误报，也没有血管/视盘像素GT来量化器官级错误。',
 '', '视觉检查发现大片漏检、整颗小区域漏检和边界误差并存；补回的部分像素仍伴随误报。连通域分桶完整数值保存在每个metrics.json，不能只用漂亮成功图解释总体增益。',
 '', f'M共同起点：{lock["start_asset"]}，原生1024窗口，halo128，唯一中心768回填；D使用实际增强后的全图真实前向。M0/M1第0步单模均40.7696%、固定半融合均41.6723%，像素输出一致。训练按同一采样清单、0/1k/2k/4k/8k匹配预算执行；第0步落后旧完整部署，后续以完整候选是否追平并超过旧部署决定价值。',
 '', 'PyTorch 2.0兼容修复只把前景统计的二维any改为先展平空间轴再单维any。修复前正式更新数为0，已完成第0步预测与初始化张量保留；训练模型、损失、几何和优化器没有改变。见pytorch_statistics_revision.json。',
 '', 'E：审计中缺官方IDRiD原图/像素GT或MAPLES配对原图，条件门槛不满足，不付费等待、不用54张加工polygon代替官方监督。',
 '', '最终仍需完成M选型、冻结管线后的一次227图test、结果报告和本机SHA备份。目标47.2115%保持，历史test属于探索性延续。全部完成并核查无人运行GPU后才触发已授权关机。']
(out/'G_阶段结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'G_fixed_gain_uncertainty.json').write_text(json.dumps(ci,indent=2),encoding='utf-8')
print('G report and diagnostic plots saved; no new model inference.')
