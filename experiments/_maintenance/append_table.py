# -*- coding: utf-8 -*-
import copy
import docx
from docx.oxml.ns import qn

DOC = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
BAK = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

doc = docx.Document(DOC)
bak = docx.Document(BAK)
body = doc.element.body
sect = body.find(qn("w:sectPr"))

print("current tables:", len(doc.tables), [len(t.rows) for t in doc.tables])
print("backup tables:", len(bak.tables), [len(t.rows) for t in bak.tables])

# 1) restore table 2 (15 rows) into the empty trailing table
src_tbl2 = bak.tables[1]._tbl
empty_tbl = doc.tables[-1]._tbl
if len(doc.tables[-1].rows) == 0:
    new_tbl2 = copy.deepcopy(src_tbl2)
    empty_tbl.addprevious(new_tbl2)
    body.remove(empty_tbl)
    print("restored table2")

print("tables now:", [len(t.rows) for t in doc.tables])

ROWS = [
    ("记录时间", "实验组名", "主要条件", "结果及判断"),
    ("2026-08 第一轮", "SE 粘贴增强", "病灶库只存SE掩膜；图像整块覆盖、标签取最大值；40k。", "验证39.7%@32k；测试39.67%。合成不一致的缺陷使“粘贴无效”不能成立。"),
    ("2026-08 第一轮", "SE 滑窗过采样", "对SE图块过采样，其余同基线；40k。", "验证39.6%@20k；测试40.07%；SE未升。"),
    ("2026-08 第一轮", "SE 粘贴加滑窗", "两法叠加；40k。", "验证40.4%@40k；测试40.79%；仍低于42.20%。"),
    ("2026-08 第一轮", "SE 损失权重扫描", "逐像素逆频率权重1.5倍、2倍、3倍与自适应。", "测试40.90%/40.31%/38.50%/38.76%；权重越强越差。"),
    ("2026-08 中旬", "E1 Tversky", "Dice换Tversky，α=0.3与0.7两版；4k。", "全正预测崩溃后停止；未测test。"),
    ("2026-08 中旬", "E1v2 CE+Dice", "Dice换CE+Dice；40k。", "验证39.970%@12k；测试39.43%。"),
    ("2026-08 中旬", "E2+E3", "可学习LR/HR融合加ASPP-lite上下文；40k。", "验证39.206%@20k；测试39.98%；调阈值40.86%。"),
    ("2026-08 中旬", "E4 细节分支", "stride4 RGB浅支路，零初始化门控；40k。", "验证38.766%@36k；测试41.63%；MA召回33.54→39.32、精度46.17→38.44、IoU 24.11→24.13%。"),
    ("2026-08 中旬", "E4v2", "E4加0.3权重辅助监督；40k。", "验证38.667%@36k；测试40.06%。"),
    ("2026-08 中旬", "E4+CE+Dice", "细节结构与组合损失；40k。", "验证39.871%@32k；测试41.08%。"),
    ("2026-08 中旬", "combo 三臂", "三组组合臂，预算未跑满。", "仅得35.666%@4k、37.782%@16k、37.089%@12k；无test，不计结果。"),
    ("2026-09-11", "A 语义条件MA残差", "E4@36k起点；残差零初始化；MA辅助0.02；冻结BN；8k。", "验证38.632%@1k、末次38.265%；未超起点，未测test。"),
    ("2026-09-11", "B 协议对照", "同起点同辅助损失，不加新分支；8k。", "验证39.205%@3k；测试41.92%；增益来自EX/HE，MA反降。"),
    ("2026-09-11", "逐类路由", "按验证IoU逐类选E4或B通道。", "验证39.90%；测试41.82%；HE排名在test反转，2.76点收益变损失。"),
    ("2026-09-11", "E4+B 等权平均", "0.5/0.5概率平均，阈值0.5，无TTA。", "验证39.54%；测试42.35%；较基线高0.15个百分点。"),
    ("2026-09-13", "D 局部学生", "B@3000初始化；仅Dice；1024裁块；20k。", "验证41.615%@2000；补充测试42.58%。"),
    ("2026-09-13", "L-PLAN（plus3）", "Dice加0.1 Lovasz，仅建配置。", "未启动；后由审查轮L臂另行执行。"),
    ("2026-09-15", "E（B起点MA头）", "零初始化残差头，仅训新头。", "验证40.255%@1500；测试42.559%。"),
    ("2026-09-15", "E2（D起点MA头）", "同结构换D起点。", "测试42.544%；验证未记载。"),
    ("2026-09-15", "E3 第三MA臂", "第三个MA残差臂。", "仅0字节smoke日志；未完成，不计负结果。"),
    ("2026-09-15", "L Lovasz对照", "D配方仅换Dice加0.1 Lovasz，best@6000。", "验证41.056%；测试41.777%，翻转41.645%。"),
    ("2026-09-16", "集成矩阵", "D/E/B/E4/L共22条test配置，概率级融合。", "最好D乘3加B加E4为43.149%；D+L为42.923%；翻转与多尺度低于单尺度翻转43.183%。"),
    ("2026-09-16", "BN统计重估", "对D学生、D全分支重估BN。", "验证40.363%/25.286%；后者崩塌，否定该配方。"),
    ("2026-09-16", "D 多尺度与翻转组合", "D多尺度2048/2560；D/E翻转加双尺度。", "测试42.797%/42.805%；均低于同轮最佳组合。"),
    ("2026-09-16", "H3 MA融合权重扫描", "w取0.5至1.0共5点，无训练。", "MA提高0.0047、mIoU提高0.0012；可忽略。"),
    ("2026-09-17", "H6/H9/H11 非MA校正头", "HE与SE头、BCE主导HE头、BCE主导MA头。", "HE 0.4356/0.4382低于0.4473；SE 0.5144低于0.5508；MA 0.2510低于0.2522。"),
    ("2026-09-17", "H8 延长头训练", "6000步、lr 2e-3，主干冻结。", "验证MA 0.2569；测试MA 0.2858，完整mIoU未升。"),
    ("2026-09-17", "H12/H13 整图监督", "整图级监督及其test确认。", "H12验证MA 0.2565、测试0.2803；H13 44.05%低于44.21%。"),
    ("2026-09-17", "P5 种子复核", "H4头seed 1/2/3。", "验证MA 0.2510/0.2537/0.2514；散布约0.0027。"),
    ("2026-09-17", "T0/T1 采样臂", "T0旧前景采样；T1加MA阳性与基准误报窗口；8k。", "MA峰值25.4468/25.4733，终点24.7038/24.7868；两臂差异很小。"),
    ("2026-09-17", "U0v2 多尺度MA网", "1,948,033参数校正网；8k。", "验证MA终点24.7509；四类42.3643%。"),
    ("2026-09-17", "E1-T0/T1 续训", "两臂各配对续训4k。", "四类42.3735%/42.3654%；MA 24.7878/24.7552。"),
    ("2026-09-17", "J0/J1 解码续训", "J0只训decoder；J1再解冻stage4；8k。", "J0约40.50%、J1约40.42%；均无提升。"),
    ("2026-09-17", "C1/F1 MA融合候选", "16项MA融合候选，锁定U0与T1各0.5。", "MA 24.8429%；合成四类42.3873%。"),
    ("2026-09-17", "R1-A/R1-B 融合感知", "纯学生Dice对0.5学生加0.5参考Dice；4k。", "最佳0.4167/0.4175；差0.08个百分点，低于0.3门槛。"),
    ("2026-09-18", "R5 第六成员与融合监督", "E0加BNEW；V1纯融合w=1；V2课程；E2换seed；E3入冠军。", "0.4273/0.4278；V1 0.4187；V2 0.4181；seed43 0.4178；E3仅提高0.0006。"),
    ("2026-09-18", "R6-P1 非MA校正头", "HE头；EX头因缓存预算到期跳过。", "HE 0.4360低于0.4376；EX未完成。"),
    ("2026-09-18", "R6-P2/R6b MA专家", "512平方Dice加BCE；256平方加Tversky；从零训练。", "patch IoU峰值0.0220/0.0563；α=0.3时MA 0.2438/0.2431，低于0.2597。"),
    ("2026-09-18", "R6-P3 终局对照", "冠军推理复算，不训练。", "验证0.4267、测试0.4421，与P0一致。"),
    ("2026-09-19", "refiner_v1", "冠军先验MA/HE残差头；Stage1仅8图200轮。", "Stage1 41.9112%为过拟合；Stage2第12,597步身份断链，未完成。"),
    ("2026-09-19", "C_S 条件采样", "C普通混合裁剪；S类平衡实例裁剪；4k。", "C 39.7974/39.1729，S 38.6455/39.8320；MIX验证42.8657%；test C@2k 39.8104、MIX 43.8064。"),
    ("2026-09-20", "U0-REAL", "ConvNeXt-S加U-Net，12k，ce加dice。", "验证38.4464%@12k；3k 17.48升至12k 38.45，曲线仍升。"),
    ("2026-09-20", "H0 DDR-only", "HRDecoder-W48完整训练对照。", "实际6k停止；验证36.9226%（41.88/43.52/41.45/20.84）。"),
    ("2026-09-20", "H1/U1 初始计划", "外部病灶预训练配置。", "仅locked-intent，未训练；不计结果。"),
    ("2026-09-20", "互补性分析", "8模型乘76图逐像素互补。", "固定逐类路由43.0645%（高0.3951）；oracle不可部署。"),
    ("2026-10-02", "U1/H1/A0 外部监督", "IDRiD栅格化48/6；HACDR骨干；2.4k至12k。", "U1外部27.1605%、DDR 36.6934%；H1外部42.9998%、DDR 35.5950%@9k；A0 9.6781%@6k后停。"),
    ("2026-10-02", "三折融合与锁定test", "候选权重0/0.125/0.25/0.5，统一U0 0.25。", "OOF 43.2483%；固定验证43.4877%；锁定test 44.7809%（高0.5699）。"),
    ("2026-10-03", "U0 高分辨率续训", "2048控制臂2k；2048/2560混合臂4k。", "单模39.0864/39.6065；融合43.3663%（低0.1214）/43.7328%（高0.2451）。"),
    ("2026-10-03", "G 高分辨率固定25%", "验证最高候选做一次冻结test。", "验证43.8679%；test 44.6345%（低0.1464）；区间负0.4094至正0.0948含0。"),
    ("2026-10-03", "M0/M1 冻结特征适配", "全局语义adapter；加stride1-2 RGB跳连；2k。", "单模41.0373/41.0441；半融合42.3552/42.3459；落后旧部署超1点。"),
    ("2026-10-04", "Q_DDR", "DINOv3 ViT-B/16加DPT，头部与最后4块；16k。", "单模37.8937%；固定25%为44.3557%；test 45.7216%。"),
    ("2026-10-04", "Q_MIX/E_NEG", "4DDR加2IDRiD加2MAPLES；负类语义；24k。", "MIX未胜；E_NEG 37.0094%、融合44.1643%，未超R。"),
    ("2026-10-05", "F4/F12 全主干追加", "从Q16k追加12k，分别解冻4块与12块。", "单模与融合38.7411/44.3069%、40.2418/44.5888%；F12成四臂父模型。"),
    ("2026-10-06", "C 臂（续训对照）", "F12起点，统一日程加EMA，32k。", "20k 41.0244%/45.1762%；32k 41.1071%/45.1949%。"),
    ("2026-10-06", "S 臂（空间交互）", "块0/3/6/9注入、2/5/8/11提取，32k。", "20k 42.2703%/45.3218%；32k 42.1521%/45.2922%；锁定test 45.8604%。"),
    ("2026-10-06", "D 臂（TJDR监督）", "独立source头，前20k额外监督，32k。", "20k 41.9082%/45.1010%；32k 41.6133%/45.1045%。"),
    ("2026-10-06", "SD 臂（联合）", "空间交互加监督包，32k。", "20k 41.3796%/44.8832%；32k 42.0312%/45.1389%；交互差为负。"),
    ("2026-10-08", "机制重构v3", "语义解码加零残差细节，计划40k。", "20k主动停止；best 37.9882%@8k；末次15.0509%，EX/SE归零。"),
    ("2026-10-09", "CET", "训练目标改造，完成40k。", "40k 38.3138%；best 40.4876%@32k；低于同预算38.835%。"),
    ("2026-10-09", "SACD", "动作纠错，20k后转配对诊断。", "20k 37.1848%。"),
    ("2026-10-09", "Residual", "detach连续FP32残差，8k。", "8k 37.2124%；私有收益0.2782低于0.5门槛，拒绝。"),
    ("2026-10-09", "Expert", "独立私有分割加可学习logit融合，4k。", "4k 27.7377%；MA 2.8132%。"),
    ("2026-10-09", "Adapter", "1/4网格直接多尺度纠错，4k。", "4k 30.4442%；只差floor 0.0316仍被拒。"),
    ("2026-10-09", "Detail", "1/2细节纠错网格。", "12k 36.7358%后16k跌至13.0748%（低23.66）；20k 15.2038%。"),
    ("2026-10-09", "Foundation", "基础阳性梯度保护加教师蒸馏，4k。", "4k 31.6795%；MA 12.8671%，私有收益为负。"),
    ("2026-10-09", "Aligned", "私有BCE/KL按全GT正负平衡，4k。", "4k 31.7823%；SE跌超2个百分点。"),
    ("2026-10-10", "Semantic", "尺度保留拼接加raw-logit旁路，4k。", "4k 32.8890%；私有收益不足1点，HE/MA仍有收益。"),
    ("2026-10-10", "IoUCorrection", "私有Dice换binary Lovasz hinge，4k。", "4k 26.2690%；四类精度全升、召回全降。"),
    ("2026-10-16", "NativePyramid v3", "2048直入W48加四尺度分类与细节支路，40k。", "40k 39.4256%；best 40.2770%@36k；MA 20.50%；Z减S为负0.0566。"),
    ("2026-10-17", "DualScale/JointContext", "后续架构设计与CPU核对。", "未实现、未训练；不计结果。"),
]


def set_cell_text(tc, text):
    ps = tc.findall(qn("w:p"))
    for p in ps[1:]:
        tc.remove(p)
    p = ps[0]
    # capture template run properties before clearing
    tmpl_rPr = None
    tmpl_r = p.find(qn("w:r"))
    if tmpl_r is not None and tmpl_r.find(qn("w:rPr")) is not None:
        tmpl_rPr = copy.deepcopy(tmpl_r.find(qn("w:rPr")))
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    r = p.makeelement(qn("w:r"), {})
    t = p.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    if tmpl_rPr is not None:
        r.append(copy.deepcopy(tmpl_rPr))
    r.append(t)
    p.append(r)


# 2) build table 3 from the restored table2 as template
tbl2 = doc.tables[-1]._tbl
header_tr = copy.deepcopy(tbl2.findall(qn("w:tr"))[0])
body_tr = copy.deepcopy(tbl2.findall(qn("w:tr"))[1])

new_tbl = copy.deepcopy(tbl2)
for tr in new_tbl.findall(qn("w:tr")):
    new_tbl.remove(tr)

for i, row in enumerate(ROWS):
    tr = copy.deepcopy(header_tr if i == 0 else body_tr)
    tcs = tr.findall(qn("w:tc"))
    for j, tc in enumerate(tcs[:4]):
        set_cell_text(tc, row[j])
    new_tbl.append(tr)

# 3) append heading + note + table before sectPr
def add_para_before_sect(text, style=None, bold=False, size=None, font="Arial Unicode MS"):
    p = doc.add_paragraph()
    if style:
        try:
            p.style = doc.styles[style]
        except KeyError:
            p.style = doc.styles[style.lower()]
    run = p.add_run(text)
    run.font.name = font
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rFonts.set(qn(attr), font)
    if bold:
        run.bold = True
    if size:
        run.font.size = docx.shared.Pt(size)
    return p

add_para_before_sect("实验表续（同学侧补充记录）", style="Heading 2", bold=True)
add_para_before_sect("本表补记正文与表一、表二未收录的同学侧实验。时间栏为记录阶段，不是逐组精确启动时刻；“报告所述”表示仅见同学侧报告、未由我方独立复算；“未完成”不计为算法负结果。指标口径同表一。")
sect.addprevious(new_tbl)
add_para_before_sect("")

doc.save(DOC)
print("saved. tables:", [len(t.rows) for t in docx.Document(DOC).tables])
