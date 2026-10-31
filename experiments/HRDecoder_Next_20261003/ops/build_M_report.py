"""CPU-only interpretation of completed paired nodes and inspected saved crops."""
from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analyze_cache import image_bootstrap
from nextlib.decision import best_candidate

NOTES={
 ('EX','new_FP'):('边缘与普通组织误报','可见亮斑周围的红色边缘及细条；增量FP按实际M1与M0差集计数，整幅红色并非全部新增。'),
 ('EX','remaining_FN'):('大片主体漏检','边缘亮色病灶区大部分为黄，仅有少量绿；主体漏检大于边缘修补。'),
 ('EX','rescued_FN'):('局部边缘补回','可见多处TP与周围FN并存；相对M0补回213像素，但仍有大片轮廓未覆盖。'),
 ('HE','new_FP'):('普通组织误报','均匀橙色组织上出现两块红色预测；没有视盘/血管GT，不据此推断临床原因。'),
 ('HE','remaining_FN'):('大片与整颗漏检','多块大面积标注为黄，仅下方局部出现TP；高误差由主体漏检支撑。'),
 ('HE','rescued_FN'):('内部与边缘少量补回','小块内部TP增加，外围及其他标注仍为FN；没有形成稳定完整覆盖。'),
 ('SE','new_FP'):('普通亮色组织误报','亮色组织中两个红色小块；这里记录视觉位置，不把明亮组织自动标为视盘。'),
 ('SE','remaining_FN'):('小中病灶整颗漏检','多块黄色区域几乎没有TP，另有局部已识别区域。'),
 ('SE','rescued_FN'):('轮廓局部补回','一个较大病灶主体已有TP，旁边数块仍为FN；可见边缘红带与黄区并存。'),
 ('MA','new_FP'):('血管邻域与普通组织误报','边缘/血管邻近可见数个红色小点；没有血管GT，血管邻域属于人工外观观察。'),
 ('MA','remaining_FN'):('微小整颗漏检','多处微小黄色标注未识别，部分病灶仅中心为TP；微小病灶问题仍存在。'),
 ('MA','rescued_FN'):('少量微小补回','一处小TP伴随多处未识别的小黄色标注；增量补回26像素，不能代表整体微小病灶改善。'),
}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def fmt(x):return f'{100*x:.4f}%'
def main():
 local=ROOT/'server_results';out=ROOT/'reports'
 histories={a:read(local/f'work/M_seed42/{a}/history.json') for a in ['M0','M1']}
 assert all([x['update'] for x in h]==[0,1000,2000] for h in histories.values())
 decision=read(local/'reports/adaptive_budget_decision.json')
 assert decision['status']=='PAIRED_COST_STOP_AT_2000'
 manual=list(csv.DictReader((local/'reports/M_error_gallery/manual_review.csv').open(encoding='utf-8-sig')))
 assert len(manual)==12
 for row in manual:row['error_type'],row['notes']=NOTES[(row['lesion_class'],row['selection'])]
 with (out/'M_manual_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(manual[0]));w.writeheader();w.writerows(manual)
 paired=[];bucket_delta={}
 for i,node in enumerate([0,1000,2000]):
  for mode in ['solo','fixed_half']:
   a,b=histories['M0'][i][mode],histories['M1'][i][mode]
   ac={x['sid']:x['counts'] for x in a['per_image']};bc={x['sid']:x['counts'] for x in b['per_image']}
   assert set(ac)==set(bc) and len(ac)==76
   ids=sorted(ac);ci=image_bootstrap([bc[x] for x in ids],[ac[x] for x in ids])
   paired.append(dict(update=node,output=mode,delta=b['mIoU_lesion4']-a['mIoU_lesion4'],image_uncertainty=ci))
   bucket_delta[f'{node}_{mode}']=(np.asarray(b['lesion_buckets'])-np.asarray(a['lesion_buckets'])).tolist()
 lines=['# M阶段：完整配对与共同成本止损','',
  '两臂在0/1000/2000步均完成76张原坐标val。最佳固定半融合分别为M0 42.3552%、M1 42.3459%，均发生在1000步，仍低于旧完整部署43.4877%；2000步分别回落至42.2027%、42.1643%。真实RGB细节支路在当前配方和已完成预算内没有带来可投入下一种子的部署增益。',
  '', '|更新数|M0单模|M1单模|M0半融合|M1半融合|半融合M1−M0|','|---:|---:|---:|---:|---:|---:|']
 for i,node in enumerate([0,1000,2000]):
  a,b=histories['M0'][i],histories['M1'][i]
  lines.append(f'|{node}|{fmt(a["solo"]["mIoU_lesion4"])}|{fmt(b["solo"]["mIoU_lesion4"])}|{fmt(a["fixed_half"]["mIoU_lesion4"])}|{fmt(b["fixed_half"]["mIoU_lesion4"])}|{100*(b["fixed_half"]["mIoU_lesion4"]-a["fixed_half"]["mIoU_lesion4"]):+.4f}点|')
 lines += ['', '原方案预算为8000步，常规共同止损门槛设在4000步。本轮根据用户“最快发现有效路径、无希望自适应止损”的授权，提前在M1完整2000步结果出现前声明了更严格的成本门槛：两臂各自历史最佳均落后旧部署至少1个百分点、最近完整区间没有上升、2000步与旧部署的按图配对95%区间上界均低于0。三个条件两臂全部满足，因此共同在2000步结束。此结论是预算取舍，不宣称网络已经收敛，也不能证明更长训练绝不改善。',
  'M0/M1在2000步相对旧部署的按图区间分别为[−2.9332, −0.0175]、[−2.9214, −0.0359]个百分点。图像bootstrap不代表训练种子复核；未触发约+0.5点的第二种子门槛。',
  '', '|节点/输出|M1−M0|按图配对95%区间（百分点）|','|---|---:|---:|']
 for row in paired:
  lo,hi=row['image_uncertainty']['percentile95']
  lines.append(f'|{row["update"]}/{row["output"]}|{100*row["delta"]:+.4f}|[{100*lo:+.4f}, {100*hi:+.4f}]|')
 lines += ['', '## 微小病灶与误差检查','',
  '按GT连通域面积≤16、17–64、65–256、257–1024、>1024原始像素分桶。下表是共同最佳节点1000步单模M1相对M0的漏检像素/整颗漏检个数变化，负数表示减少。完整所有类别、节点与单模/半融合桶计数见M_analysis_summary.json。',
  '', '|类别|面积≤16：FN像素／整颗漏检|17–64|65–256|257–1024|>1024|','|---|---:|---:|---:|---:|---:|']
 delta=np.asarray(bucket_delta['1000_solo'])
 for c,name in enumerate(['EX','HE','SE','MA']):
  lines.append('|'+name+'|'+'|'.join(f'{delta[c,k,3]:+d}／{delta[c,k,1]:+d}' for k in range(5))+'|')
 lines += ['', '已人工审阅12张按真实误差差集/漏检总数挑选的原坐标裁块，各类各含新增FP、剩余FN、补回FN。大面积/整颗漏检仍然突出；成功图主要显示少量局部或边缘补回，同时存在普通组织和血管邻域误报。图中颜色表示M1全部TP/FP/FN，不表示每个着色像素都由细节支路新增；类别和增量排名分别记录在M_manual_review.csv及原始exact_error_selection.json。没有血管/视盘GT，视觉分类只作人工位置描述。',
  '', '审阅入口：[EX](M_contact_sheets/EX.png)、[HE](M_contact_sheets/HE.png)、[SE](M_contact_sheets/SE.png)、[MA](M_contact_sheets/MA.png)。',
  '', '## 身份、数据曝光与工程收尾','',
  '两臂更新2000、清单cursor16000一致；1000→2000区间四类中心前景出现次数一致，均为[3725,5060,1437,4718]。两臂各有1次AMP溢出跳步，均重放原批次，没有转移采样cursor。micro4×2、FP32损失、归一化、冻结D真实增强全图缓存、模型源SHA均保持。',
  '首次收尾守卫把identity.json中既有的统计兼容注解也拿来与checkpoint核心身份比较，触发保护恢复。随后只修复守卫：该注解逐字段验证、核心身份保持完全相等；模型checkpoint未改。旧队列在恢复期间已打印M0 2100/4000，属于未完成验证且未入选的少量额外计算；停止时保留的last仍严格为2000步，两臂最终比较都只使用完整0/1k/2k节点，未将这段额外更新冒充配对成果。',
  'best均为1000步，last均为2000步；选择与止损回执、源/协议/清单SHA和全部完整节点报告保存。后续仅对val晋级的G管线进行一次冻结最终test，不追加机制消融或旧模型延长训练。']
 (out/'M_阶段结论.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 summary=dict(status='REVIEWED_PAIRED_COMPLETE',paired=paired,lesion_bucket_deltas_M1_minus_M0=bucket_delta,
  visual_review=dict(status='REVIEWED',images=12,selection='Actual saved probability pixel differences, no new inference',
  ontology='Appearance descriptions; no vessel or optic-disc ground truth',annotation='Full candidate TP/FP/FN overlay'),
  budget_decision=decision,model_source_unchanged=True)
 (out/'M_analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(status=summary['status'],paired=paired),ensure_ascii=True))
if __name__=='__main__':main()
