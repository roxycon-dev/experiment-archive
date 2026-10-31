"""Summarize completed saved artifacts only; never authorizes shutdown or reads test images."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyze_cache import image_bootstrap
from nextlib.decision import best_candidate
LOCAL=ROOT/'server_results';REPORTS=ROOT/'reports'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def fmt(x):return f'{100*x:.4f}%'
def main():
 completion=read(LOCAL/'reports/campaign_completion.json')
 assert completion['status']=='EXPERIMENTS_COMPLETE_BACKUP_REVIEW_PENDING'
 selection=read(LOCAL/'reports/val_final_selection.json');winner=selection['winner']
 lock=read(LOCAL/'work/geometry_lock.json');base=read(LOCAL/'work/G/current_deployment/metrics.json')
 histories={}
 for p in sorted((LOCAL/'work').glob('M_seed*/M*/history.json')):
  histories[(int(p.parent.parent.name.replace('M_seed','')),p.parent.name)]=read(p)
 lines=['# HRDecoder严格方案最终报告（汇总草稿，等待主代理审查）','',
  f'验证集选择的最高完整管线：{winner}。选择依据为完整76张val，固定0.5阈值；没有通过test调整比例或节点。',
  f'当前旧完整部署val {fmt(lock["baseline_val_miou"])}、历史test44.7809%；主目标47.2115%保持。']
 test=read(LOCAL/'work/final_test.json') if (LOCAL/'work/final_test.json').exists() else None
 if test:
  old=test['old_complete_pipeline']['mIoU_lesion4'];new=test['selected_complete_pipeline']['mIoU_lesion4']
  lines += ['',f'冻结后的227图最终test：旧完整管线{fmt(old)}；选中完整管线 **{fmt(new)}**，变化{100*(new-old):+.4f}个百分点。目标'+('达到。' if test['target_met'] else f'未达到，距离目标还差{100*(.472115-new):.4f}点。'),
   '此test在过往轮次已经查看，属于探索性延续。新管线按预先冻结的val选择评价，不在看到test后重新搜索。',
   '', '|类别|旧完整test IoU|选中test IoU|TP变化|FP变化|FN变化|','|---|---:|---:|---:|---:|---:|']
  for c in ['EX','HE','SE','MA']:
   a=test['old_complete_pipeline']['per_class'][c];b=test['selected_complete_pipeline']['per_class'][c]
   lines.append(f'|{c}|{fmt(a["iou"])}|{fmt(b["iou"])}|{b["tp"]-a["tp"]:+d}|{b["fp"]-a["fp"]:+d}|{b["fn"]-a["fn"]:+d}|')
  lines += ['',f'最终test的按图配对bootstrap增量95%区间：{test["paired_image_uncertainty"]["percentile95"]}。该区间使用IoU的小数单位，不替代训练种子差异。']
 else:lines += ['', '本轮没有晋级的新val管线，因此未重复最终test；原完整部署无条件保留。']
 lines += ['', '## G：已有权重与几何', '',
  'G矩阵、H1路径对照、高分辨率起点的一次评价、折外校准及24张代表误差检查均完成；详见[G阶段结论](G_阶段结论.md)。',
  '最佳G固定25%管线val43.8679%；H1普通forward41.7007%、eval_forward35.5916%，路径差6.1091点。其固定25%融合43.8383%，低于最佳G，没有继续扫比例或重开旧训练。',
  '最佳G的四类IoU增益主要来自降低FP，四类TP均减少，不能称为全面召回改善。校准OOF仅约0.1—0.24点，未替代主结构训练。',
  '', '## M：匹配结构与全部完整节点', '',
  'D@2000按原冠军加载器逐键还原，取已训练的64通道pre-classifier，权重/BN均冻结。共同局部起点U0高分辨率@1000，原生1024窗口、halo128、中心768唯一回填。M0有全局语义adapter，M1额外有真实RGB stride1/2跳连；零初始化末投影保持两臂0步完全一致。',
  '两臂重放同一图像/窗口/翻转旋转清单，micro4×2、FP32 BCE+Dice、AdamW、分组LR和早期层冻结保持。D缓存来自实际增强全图的真实FP32前向，未旋转旧特征替代。',
  '', '|seed|臂|更新数|单模val|固定半融合val|','|---:|---|---:|---:|---:|']
 for (seed,arm),history in histories.items():
  for row in history:lines.append(f'|{seed}|{arm}|{row["update"]}|{fmt(row["solo"]["mIoU_lesion4"])}|{fmt(row["fixed_half"]["mIoU_lesion4"])}|')
 paired=[]
 for seed in sorted(set(k[0] for k in histories)):
  h0={x['update']:x for x in histories[(seed,'M0')]};h1={x['update']:x for x in histories[(seed,'M1')]}
  assert set(h0)==set(h1),'Both completed arms must expose the same full-val nodes'
  common=sorted(set(h0)&set(h1));assert common
  for node in common:
   a,b=h0[node],h1[node]
   for mode in ['solo','fixed_half']:
    ac={x['sid']:x['counts'] for x in a[mode]['per_image']};bc={x['sid']:x['counts'] for x in b[mode]['per_image']}
    assert set(ac)==set(bc)
    ids=sorted(ac);ci=image_bootstrap([bc[x] for x in ids],[ac[x] for x in ids])
    paired.append(dict(seed=seed,update=node,output=mode,
       delta=b[mode]['mIoU_lesion4']-a[mode]['mIoU_lesion4'],paired_image_uncertainty=ci))
  state=read(LOCAL/f'work/M_seed{seed}/paired_state.json')
  assert state['node']==max(common),'Archive is not yet synchronized to the paired terminal node'
  decision=read(LOCAL/f'work/M_seed{seed}/selection.json')
  lines += ['',f'seed{seed}终点：{state["node"]}步，止损理由：{state["reason"]}。最佳完整候选为{decision["winner"]}，第二种子门槛{decision["second_seed_gate"]}。']
  if 'adaptive_stop' in decision:
   lines += ['本轮按用户授权自适应改为完整匹配2000步共同成本止损；两臂各自历史最佳落后旧部署至少1点、最近区间回落，且2000步相对旧部署的按图95%区间上界均低于0。门槛在M1完整2000步结果前声明。原定4k/8k没有继续；这属于发现预算决策，不宣称结构已经收敛或长训绝不改善。',
    '详细配对区间、面积桶漏检、12张人工误差检查及少量未入选的M0额外更新，见[M阶段结论](M_阶段结论.md)。']
  fig,ax=plt.subplots(figsize=(8,4.8),layout='constrained')
  for arm in ['M0','M1']:
   hist=histories[(seed,arm)]
   for mode,style in [('solo','--'),('fixed_half','-')]:
    ax.plot([x['update'] for x in hist],[100*x[mode]['mIoU_lesion4'] for x in hist],style,marker='o',label=arm+' '+mode)
  ax.axhline(100*lock['baseline_val_miou'],color='black',ls=':',label='Old complete')
  ax.axhline(43.86790869927479,color='gray',ls='-.',label='Best G fixed25')
  ax.set(xlabel='Successful optimizer updates',ylabel='Complete validation mIoU (%)',title=f'Matched structure experiment, seed {seed}')
  ax.grid(alpha=.2);ax.legend(fontsize=9);fig.savefig(REPORTS/f'M_curve_seed{seed}.png',dpi=160);plt.close(fig)
  lines += ['',f'![完整验证曲线](M_curve_seed{seed}.png)']
 if winner['kind']=='M':
  hist=histories[(winner['seed'],winner['arm'])];row=next(x for x in hist if x['update']==winner['update']);chosen=row[winner['output']]
 else:
  cache='highres_fixed25' if winner['asset']=='U0_highres' else winner['asset']+'_'+winner['canvas']+'_'+winner['inference']+'_fixed25'
  chosen=read(LOCAL/f'work/G/{cache}/metrics.json')
 ac={x['sid']:x['counts'] for x in base['per_image']};bc={x['sid']:x['counts'] for x in chosen['per_image']}
 ids=sorted(ac);assert set(ac)==set(bc)
 valci=image_bootstrap([bc[x] for x in ids],[ac[x] for x in ids])
 summary=dict(val_final_selection=selection,test=test,paired_structure_results=paired,
     chosen_val_uncertainty=valci,review_status='DRAFT',shutdown_status='not asserted by this report builder')
 (REPORTS/'final_analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 lines += ['', '完整val入围增量的按图95%区间：'+str(valci['percentile95'])+'（IoU小数单位）。逐节点M1−M0的配对区间见final_analysis_summary.json；有第二种子时分别报告，避免把图像重采样当成种子复核。',
  '', '## 条件分支、工程修复和保留资产', '',
  'E官方资产门槛不满足：没有齐备的官方原图/像素GT或MAPLES配对原图，没有使用加工polygon冒充官方监督，没有等待GPU空转。容量一致的低通机制对照未占用本轮发现预算。',
  'PyTorch2.0仅前景统计any兼容修复发生在正式更新数0时；完成的0步预测和所有初始模型张量保留。训练图、损失、优化器与采样没有改变，身份迁移及统计等价已审计。工程异常不作为算法负结果。',
  '真实服务器验收：D逐键/特征等价、M0/M1零步最大误差0、冻结BN、两次更新后真实细节支路接通；micro4×2验收峰值约10.97GiB。协议/实际恢复10项检查通过。',
  '原赢家及全部历史源权重保留。每臂best/last、所有完整节点计数、配置/采样/初始化/梯度身份均保存；新的非赢家G概率仅在完整76文件本机SHA验证且分析完成后释放。',
  '', '## 复核入口与收尾', '',
  '原始服务器归档位于server_results；最终冻结配方work/final_lock.json，逐图最终计数work/test_progress.json，最终结果work/final_test.json。ops/local_probability.py可离线解析远端绝对路径的lazy融合，保持原index文件不变；真实旧管线样本已通过逐组件SHA及数组精确一致验证。',
  '本报告是已完成数值的自动汇总草稿，仍需主代理核查解释、必要的M误差图库、全部必需本机备份及关机回执。此脚本不会写completion_ready或触发关机。']
 (REPORTS/'最终实验报告_草稿.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('Completed results summarized; review and verified backup are still required.')
if __name__=='__main__':main()
