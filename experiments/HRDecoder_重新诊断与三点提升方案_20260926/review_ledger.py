from pathlib import Path
import json
root=Path(__file__).resolve().parent
d=json.loads((root/'history_ledger_workbuddy_unreviewed.json').read_text(encoding='utf-8'))
recs=d['records']
reviews={
'HIST-BASELINE':'历史基准；仅自定义454/76/227划分内可比。ImageNet初始化以运行配置为准，不沿用“随机/ImageNet”含混表述。',
'HIST-E1-Tversky':'只否定实际运行的早期配置；不能否定Tversky族。',
'HIST-SE-4-combo':'含同一copy-paste标签问题，不能据此否定正确实现的粘贴增强。',
'B':'B相对E4的主要test改善为EX/HE；不能说来自EX/SE。MA下降。',
'H4':'H4单独接D时MA=25.2207(val)；42.6686是接入完整集成后的口径，二者不得拼成一次评估。首层权重大小也不足以证明RGB的因果贡献。',
'H5/H7':'最终冠军test约44.2115%；h7 JSON与后来四通道重算有末位差异，应保留各自原值及出处。',
'R1-B':'两臂各自最佳41.75−41.67=+0.08pp；逐节点最大差按报告四舍五入数为+0.11pp，不是+0.08pp。',
'U0-REAL':'原始加载日志明确340个编码器键加载成功；官方ImageNet ConvNeXt-S初始化。12000步末点仍上升，不能否定充分训练后的架构。',
'H0':'H0_stopped6000_metadata.tar.gz内有train.log及3000/6000验证JSON；不能说本地没有日志、未训练。无本地最终权重不等于没有执行。',
'H1':'只找到计划，无外部病灶预训练完成证据。',
'U1':'只找到计划，无外部病灶预训练完成证据。',
'FC-EXPERT-COMP':'固定逐类路由43.0645%，+0.3951pp。OR是可执行并集，不是GT oracle；跨实验套用CI宽度无效。',
'refiner_v1':'Stage1仅8张图的过拟合检查，不能当454图正式训练的泛化结果。Stage2本地完整断点epoch19/12597步；恢复时误套fresh-start identity检查，随后仍有旧失败标记风险，未取得正式末轮结果。另发现训练补零/GN与推理不补零不一致，详见CPU诊断。',
'R6-P2-M1':'从零训练的当前小专家失败，不能推出所有高分辨率专家不可行。',
'R6-P1-HE':'本次小头未改善，不足以证明HE不能用校正头或所有非MA头无效。',
'AUDIT-PHASE-R0':'记录数据可用；图像bootstrap针对给定两个模型，不包含重新训练随机性。',
}
out=[]
for rec in recs:
    r=dict(rec)
    r.pop('can_negate',None)
    r['review']='证据限定为所列配置、节点及来源，不外推整个方法族。'
    if r['id'] in reviews:r['review']=reviews[r['id']]
    if r['id']=='U0-REAL':
        r['init']='官方ImageNet ConvNeXt-S权重，运行日志loaded=340/missing=[]；导出阶段禁止外部初始化不等于训练阶段没加载'
    if r['id']=='H0':
        r['status']='trained_stopped6000_local_metadata_present'
        r['steps']='6000；本地有日志与验证JSON，未找到H0完整checkpoint'
        r['change']='HRDecoder-W48 DDR-only对照；实际执行与旧占位配置分开记录'
    if r['id']=='R1-B':r['test']='未测；各自最佳差+0.08pp；报告四节点逐点最大差约+0.11pp'
    if r['id']=='H4':r['val']='D后MA头25.2207%；完整集成接头mIoU42.6686%（不同评估组合）'
    if r['id']=='refiner_v1':
        r['steps']='Stage1 8图200ep；Stage2实际完整保存ep19/step12597，计划30ep未完整验收'
        r['val']='Stage1 8图过拟合模型41.9112%，不代表Stage2正式模型；未取得Stage2最终val'
        r['status']='interrupted_with_resume_guard_failure_incomplete'
    if r['id']=='ROUTE':r['status']='evaluation_complete_no_gain'
    if r['id']=='AUDIT-PLANNED-T0/T1/U0/E1/V1/J0/J1/C1/F1':
        r['status']='superseded_plan_snapshot'
        r['review']='旧RUN_PLAN的NOT_RUN已过时。WorkBuddy 9月17日最终报告及T0/T1原始计数证明多臂已执行，见新增分行。'
    out.append(r)

src=r'C:\Users\GuanXingRen\WorkBuddy\2026-09-15-22-20-12\审查轮_数据与实验设计整理_20260917.md'
extra=[
('AUDIT-T0v2','旧前景采样；3361参数MA头，512输入/256中心监督；8000步','MA峰值25.4468@1000，终点24.7038；四类终点42.3525','未单列','原始T0计数JSON+报告§3.1/§10.1'),
('AUDIT-T1v2','2均匀+1MA阳性+1基准误报的采样；其余匹配T0','MA峰值25.4733@4000，终点24.7868；四类终点42.3733','MA27.5344；合成mIoU43.9311','原始T1计数JSON+报告§3.1/§10.5'),
('AUDIT-U0v2','1948033参数多尺度MA校正网；8000步，区别于9月20日ConvNeXt U0','MA终点24.7509；四类42.3643','MA27.5153；合成mIoU43.9263','报告§10.1/§10.5'),
('AUDIT-E1-T0','T0从8000步配对续训4000步','MA24.7878；四类42.3735','未单列','报告§10.1'),
('AUDIT-E1-T1','T1配对续训4000步','MA24.7552；四类42.3654','MA27.5796；合成mIoU43.9424','报告§10.1/§10.5'),
('AUDIT-V1','T0/T1分别种子1/2/3，共6次，种子2/3追加四臂','报告估计MA跨seed标准差0.001905（比例）','未单列','报告§11；不把这个方差套到别的实验test差值'),
('AUDIT-J0','D@20000起点，只训decoder，固定5e-5，8000步','40.50@约4000；40.48@约8000','未单列','报告§3.3；原始远端路径未本轮访问'),
('AUDIT-J1','同J0，额外解冻stage4，BN冻结','40.417@约4000；40.448@约8000','未单列','报告§3.3；梯度证据不等于充分训练'),
('AUDIT-C1/F1','16项MA融合候选；锁定U0/T1各0.5','MA24.8429；合成四类42.3873','MA27.5061；合成四类43.9240','报告§10.3-10.5；非MA三类为历史常数，非此次四通道实测'),
('AUDIT-T0/T1-FIRST','第一遍脚本val_eval定义未调用，重跑为v2','没有正式验证曲线','未测','报告§5.1；工程失败，不当作方法结果'),
]
for id,change,val,test,ev in extra:
    out.append(dict(date='2026-09-17',id=id,family='audit_0916_followup',change=change,val=val,test=test,status='reported_executed',source=src+'；'+ev,review='百分数单位。除T0/T1有本地原始计数支持外，本行主要为该最终报告的事实摘录；未把报告因果/显著性结论照抄。'))

out.extend([
dict(date='2026-09-16',id='D-BN-STUDENT',family='MA_head',change='对D学生BN统计重估',steps='推理诊断',val='40.363（EX43.90/HE41.77/SE53.96/MA21.83）',test='未测',status='complete',source='HRDecoder_MA残差头与集成_20260915-16/02_原始数据/eval_json/E_0916__eval_VAL_D_renorm_stu.json',review='低于D原val41.615。否定当前重估配方，不等于所有BN策略无效。'),
dict(date='2026-09-16',id='D-BN-ALL',family='MA_head',change='对D全部分支BN统计重估',steps='推理诊断',val='25.286（EX32.52/HE29.45/SE36.05/MA3.12）',test='未测',status='complete',source='HRDecoder_MA残差头与集成_20260915-16/02_原始数据/eval_json/E_0916__eval_VAL_D_renorm_all.json',review='大幅退化。不能把它作为继续重估BN即可提高的证据。'),
dict(date='2026-09-16',id='D-MULTISCALE',family='MA_head',change='D推理尺度2048/2560',steps='推理',val='未单列',test='42.797（EX54.70/HE41.85/SE50.09/MA24.55）',status='complete',source='HRDecoder_MA残差头与集成_20260915-16/02_原始数据/eval_json/E_0916__eval_SCALE_D_2048_2560.json',review='较D42.576有小幅增加；不能推广为所有多尺度集成都提升。'),
dict(date='2026-09-16',id='DE-FLIP-MULTISCALE',family='MA_head',change='D/E翻转与双尺度融合',steps='推理',val='未单列',test='42.805（EX53.98/HE44.36/SE48.61/MA24.27）',status='complete',source='HRDecoder_MA残差头与集成_20260915-16/02_原始数据/eval_json/E_0916__eval_ENS_DE_flip_2scale.json',review='低于同轮D/E单尺度翻转43.183。'),
])

result={'date':'2026-09-26','scope':'历史家族台账，含已执行/计划/工程失败，非独立重跑证明','records':out,
'global_corrections':['WorkBuddy初稿漏读了9月17日已执行T/U/E/V/J/C/F队列，现补入。','D/E/B/E4/H4均有本地推理用权重资产；缺完整优化器日志不等于冠军无法复现，详见asset_presence.json。','“可否否定路线”粗标签已移除；未显著不等于等效，也不等于全部路线失败。','未将全部文件索引等同于逐文件内容核验；未知和缺失留为未知。']}
(root/'history_ledger_reviewed.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

def fmt(x):
    if isinstance(x,(dict,list)):x=json.dumps(x,ensure_ascii=False)
    return str(x).replace('|',' / ').replace('\n',' ')
lines=['# 历史实验台账：事实与证据边界','',
'2026-09-26。原始数据不重跑；每行保留来源与缺项。“完成”仅指所列运行，并非方法有效。数值列保留来源原单位，0.44与44%须按行辨别；新增9月17日分行统一用百分数。',
'','WorkBuddy原始答复存于delegation；以下为关键错误经审核后的台账。旧RUN_PLAN的未运行状态不能覆盖较晚的真实结果。',
'','|实验ID|日期|动机/改法|训练/节点|val原记值|test原记值|状态|来源|审核限定|',
'|---|---|---|---|---|---|---|---|---|']
for r in out:
    lines.append('|'+ '|'.join(fmt(r.get(k,'未记载')) for k in ('id','date','change','steps','val','test','status','source','review'))+'|')
lines+=['','逐图计数与来源文件仍以原目录为准。候选阈值、同一模型不同TTA和集成权重未全部展开成“新训练”，完整文件索引见local_file_catalog.jsonl。',
'','冠军链：E4@36000 → B@3000 → D@2000 / E@1500；D×3+E×2+B+E4+翻转，再接H4 MA头。H5/H7为最终管线结果，不能称为某个单模型的训练成绩。',
'','本次确认的遗漏/冲突处理见主报告；未拿到原始文件的报告数字不会升级为独立核验结果。']
(root/'02_历史实验台账_审核版.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
report=root.parent/'HRDecoder_MA残差头与集成_20260915-16/01_结果报告/最终报告_两轮实验汇总.md'
txt=report.read_text(encoding='utf-8')
(root/'02b_集成与尺度候选原表.md').write_text('# 9月15日至16日推理候选原表\n\n以下完整保留来源报告的22行test及5行val表；重复别名和单次复现不当作独立训练。这里只搬录已有历史结果，不重新进行test选型。\n\n来源：'+str(report)+'\n\n'+txt[txt.index('## 一、'):],encoding='utf-8')
print('reviewed records',len(out))
