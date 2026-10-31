"""Record the agent's inspection of all 24 existing error crops; no image modification."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'server_results/reports/G_error_gallery/manual_review.csv'
rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
observations=[
 ('病灶边缘/普通组织','多发亮区的预测边缘扩张，图内也有散在漏检；不是明显视盘或黑边误报。'),
 ('病灶边缘','亮斑周边有红色外扩；黑边在画面中，主要红区贴近病灶而非进入黑边。'),
 ('血管邻近/视盘邻近','视盘周围血管与组织可见少量独立红区；不能据此确定器官级误报归因。'),
 ('血管邻近/病灶边缘','红区位于血管邻近及部分绿色斑块边缘，剩余小块漏检同时存在。'),
 ('整块区域漏检','大片黄色GT区域仅少量绿色岛，超出小范围轮廓修补。'),
 ('整颗区域漏检/视盘邻近','靠近视盘的一个黄色标注区域几乎没有绿色覆盖。'),
 ('边缘补回','多发小亮区绿色边缘与黄色剩余部分共存，变化主要有轮廓修补。'),
 ('边缘补回','多个亮斑主体已经预测，局部轮廓补回；红色外扩同时出现。'),
 ('整块区域漏检','大片黄色区域几乎未覆盖，图像偏暗且背景病灶密集。'),
 ('病灶边缘/区域漏检','大块绿色主体外仍有宽黄色带；红色外扩与漏检并存。'),
 ('普通组织/血管邻近','明显血管附近的普通组织存在孤立红色小块，不位于黑边。'),
 ('血管邻近/病灶边缘','红区处于血管及亮区邻近，另有绿色主体边缘外扩。'),
 ('整颗区域漏检','血管邻近多个黄色斑块基本无绿色覆盖。'),
 ('病灶边缘','大块绿色主体周围仍有宽黄色边缘，另有局部整块漏检。'),
 ('边缘补回','绿色主体外仍有黄色边缘，补回未解决全部漏检。'),
 ('边缘补回/视盘邻近','视盘邻近一个绿色主体和黄色周边，残余轮廓误差明显。'),
 ('普通组织/血管邻近','稀疏小红点出现在纹理与细血管邻近，有些和可见暗点重合；不判断标注对错。'),
 ('普通组织/血管邻近','小红块和黄色漏检并存，血管及其他类别亮斑都在画面中。'),
 ('整颗小区域漏检','大量离散黄色小块，只有少数绿色点，不能概括成边界误差。'),
 ('整颗小区域漏检','离散黄色小标注基本未被覆盖，邻近血管与亮斑；黑边不改变评价范围。'),
 ('小区域补回/剩余漏检','存在少量绿色小块，但画面仍有散在黄色和红色点。'),
 ('小区域补回/剩余漏检','可见绿色小点及散在黄色小标注，整体剩余漏检依然明显。'),
 ('小区域补回/剩余漏检','仅少量小绿色区域；未解决全部离散小区域。'),
 ('小区域补回/剩余漏检','血管周围有极小绿色与黄色区域，不能据此宣称微小病灶普遍解决。'),
]
# The source sheet groups each class as new_FP, remaining_FN, rescued_FN.
mapping={
 ('EX','new_FP'):[0,1],('EX','remaining_FN'):[4,5],('EX','rescued_FN'):[6,7],
 ('HE','new_FP'):[2,3],('HE','remaining_FN'):[8,9],('HE','rescued_FN'):[14,15],
 ('SE','new_FP'):[10,11],('SE','remaining_FN'):[12,13],('SE','rescued_FN'):[14,15],
 ('MA','new_FP'):[16,17],('MA','remaining_FN'):[18,19],('MA','rescued_FN'):[22,23],
}
# Correct HE rescued-region observations are specific to the two inspected large dark regions.
overrides={('007-6734-400','HE','rescued_FN'):('主体/边缘补回','大块病灶主体绿色，但宽黄色外沿和红色外扩仍在。'),
 ('007-3567-200','HE','rescued_FN'):('主体/边缘补回','多个暗区主体绿色，周边黄色残留明显，同时有局部红色外扩。')}
seen={}
for row in rows:
 key=(row['lesion_class'],row['selection']);ix=seen.get(key,0);seen[key]=ix+1
 category,note=overrides.get((row['sid'],*key),observations[mapping[key][ix]])
 row['error_type']=category;row['notes']=note
target=ROOT/'reports/G_manual_review.csv'
with target.open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(ROOT/'reports/G_visual_review.json').write_text(json.dumps(dict(status='complete',images_inspected=24,
 reviewer='Codex visual inspection',candidate='G highres fixed25',baseline='current complete pipeline',
 selection='actual per-pixel new_FP, remaining_FN, rescued_FN; top two images per class and category',
 limits='Overlay displays all candidate TP/FP/FN, not just changed pixels. Categories describe crop context; no vessel/disc ground truth or clinical judgment.',
 findings='Both large residual regions and lesion-boundary errors remain. Small rescues coexist with new false positives. Does not support a guarantee from calibration or detail alone.',
 table=str(target),gallery=str(source.parent/'index.html')),ensure_ascii=False,indent=2),encoding='utf-8')
print(target)
