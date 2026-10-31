# -*- coding: utf-8 -*-
"""删除新增的第六章与表三，并把九月以来缺失的实验按时间线并入原表一/表二。"""
import copy
import re
import docx
from docx.oxml.ns import qn

DOC = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"

doc = docx.Document(DOC)
body = doc.element.body

# ---------- 1) 删除第六章（标题 -> “以上记录不改变正文的结论”段） ----------
ps = [c for c in body if c.tag == qn("w:p")]
start_el = end_el = None
for p in ps:
    t = "".join(n.text or "" for n in p.iter(qn("w:t"))).strip()
    if t == "六 同学侧补充实验记录":
        start_el = p
    if start_el is not None and t.startswith("以上记录不改变正文的结论"):
        end_el = p
        break
assert start_el is not None and end_el is not None, "section 6 not found"

children = list(body)
i0 = children.index(start_el)
i1 = children.index(end_el)
removed = 0
for el in children[i0:i1 + 1]:
    body.remove(el)
    removed += 1
print("removed section6 elements:", removed)

# ---------- 2) 删除表三及其标题、说明 ----------
drop_texts = ("实验表续（同学侧补充记录）",)
note_prefix = "本表补记正文与表一、表二未收录"
to_drop = []
for p in [c for c in body if c.tag == qn("w:p")]:
    t = "".join(n.text or "" for n in p.iter(qn("w:t"))).strip()
    if t in drop_texts or t.startswith(note_prefix):
        to_drop.append(p)
for p in to_drop:
    body.remove(p)
print("dropped heading/note:", len(to_drop))

if len(doc.tables) > 2:
    body.remove(doc.tables[2]._tbl)
    print("table3 removed")

# 清理表三后面多余的空段（保留一个空段在文末即可）
tail = [c for c in body if c.tag == qn("w:p")]
if tail:
    last = tail[-1]
    if not "".join(n.text or "" for n in last.iter(qn("w:t"))).strip():
        prev = last.getprevious()
        if prev is not None and prev.tag == qn("w:tbl"):
            body.remove(last)
            print("removed trailing empty p")

print("tables now:", [len(t.rows) for t in doc.tables])

# ---------- 3) 重建表格 ----------
def set_cell_text(tc, text):
    ps = tc.findall(qn("w:p"))
    for p in ps[1:]:
        tc.remove(p)
    p = ps[0]
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


def date_key(text):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})-(\d{2})", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), 1)
    return (9999, 12, 31)


def rebuild(table, new_rows):
    tbl = table._tbl
    trs = tbl.findall(qn("w:tr"))
    header, body_rows = trs[0], trs[1:]
    template = copy.deepcopy(body_rows[-1])
    items = []
    for tr in body_rows:
        first = tr.findall(qn("w:tc"))[0]
        txt = "".join(n.text or "" for n in first.iter(qn("w:t")))
        items.append((date_key(txt), 0, tr))
    for row in new_rows:
        tr = copy.deepcopy(template)
        tcs = tr.findall(qn("w:tc"))
        for j, val in enumerate(row[:4]):
            set_cell_text(tcs[j], val)
        items.append((date_key(row[0]), 1, tr))
    items.sort(key=lambda x: (x[0], x[1]))
    for tr in trs:
        tbl.remove(tr)
    tbl.append(header)
    for _, _, tr in items:
        tbl.append(tr)
    return len(items)


T1_NEW = [
    ("2026-09-11", "A（语义MA残差）", "相对E4@36k：只加“主干语义+原图细节→MA残差”分支（末层零初始化）与0.02 MA辅助损失；冻结BN、lr 5e-4、8k。", "验证38.632%@1k、末次38.265%，未超起点38.766%；未测test。"),
    ("2026-09-11", "B（协议对照）", "相对A：只删去残差分支，起点、损失、冻结与8k预算全部相同。", "验证39.205%@3k；测试41.92%（EX52.65/HE44.37/SE48.91/MA21.74）；增益在EX/HE，MA反降。"),
    ("2026-09-11", "ROUTE（逐类路由）", "相对B：不训练，按验证IoU逐类选E4或B的通道。", "验证39.90%；测试41.82%；HE在验证与测试排名反转，2.76个百分点收益变损失。"),
    ("2026-09-11", "ENS（E4+B等权）", "相对ROUTE：取消逐类选择，四类统一0.5/0.5概率平均，阈值0.5、无TTA。", "验证39.54%；测试42.35%；较基线42.20%高0.15个百分点。"),
    ("2026-09-13", "D（局部学生）", "相对B@3000：改为1024裁块训练、推理与冻结B全图概率各半；损失只保留Dice；20k。", "验证41.615%@2k、末次40.476%；补充测试42.58%。"),
    ("2026-09-15", "E（B起点MA头）", "相对B：只加7通道MA后融合头（3361参数、末层零初始化），主干与参考全冻结。", "验证40.255%@1500；测试42.559%。"),
    ("2026-09-15", "E2（D起点MA头）", "相对E：只把底座从B换成D，头结构与训练方式不变。", "测试42.544%；验证未记载。"),
    ("2026-09-15", "E3（第三MA臂）", "相对E2：只更换第三组初始化/数据配置（原文未细列）。", "仅0字节smoke日志，未完成；不计负结果。"),
    ("2026-09-15", "L（Dice+0.1Lovasz）", "相对D：只把损失从Dice换成Dice+0.1 Lovasz，其余同D；best@6000。", "验证41.056%；测试41.777%，翻转TTA 41.645%。"),
    ("2026-09-16", "集成矩阵（22条）", "相对D单模：只改融合成员、权重与是否翻转/多尺度，全部不重训。", "最高D×3+E×2+B+E4+翻转43.229%；D×3+B+E4 43.149%；D+L 42.923%；D翻转42.717%。"),
    ("2026-09-16", "D-BN-STU（学生BN重估）", "相对D推理：只重估学生分支BN统计，权重与融合不变。", "验证40.363%，低于D的41.615%。"),
    ("2026-09-16", "D-BN-ALL（全分支重估）", "相对D-BN-STU：只把重估范围扩到全部分支。", "验证25.286%，崩溃；该配方被否定。"),
    ("2026-09-16", "D-MS（多尺度）", "相对D推理：只把输入尺度从2048改为2048/2560双尺度。", "测试42.797%，略高于D的42.576%。"),
    ("2026-09-16", "DE-FLIP-2S（翻转+双尺度）", "相对D-MS：再加入水平翻转TTA与E成员。", "测试42.805%，低于同轮单尺度翻转43.183%。"),
    ("2026-09-16", "AUDIT-R0（融合分解）", "无训练：对student/reference/blend做逐图计数分解与bootstrap。", "student 39.335%、reference 39.205%、blend 41.615%；ΔmIoU +0.87pp，区间[0.57,1.18]。"),
    ("2026-09-16", "H3（MA权重扫描）", "相对冠军融合：只扫MA通道权重0.5/0.6/0.7/0.8/1.0，不训练。", "MA最高+0.0047、mIoU+0.0012，可忽略。"),
    ("2026-09-16", "H6（HE/SE校正头）", "相对H4：只把校正目标从MA换成HE或SE，头结构与冻结不变。", "HE 0.4356<0.4473；SE 0.5144<0.5508，均被拒。"),
    ("2026-09-17", "H8（延长头训练）", "相对H4：只把步数2000→6000、学习率1e-3→2e-3。", "验证MA 0.2569；测试MA 0.2858，仍低于冠军0.2866。"),
    ("2026-09-17", "H9（BCE主导HE头）", "相对H6的HE头：只把损失配比改为BCE/Dice=0.9/0.1。", "HE 0.4382<0.4473。"),
    ("2026-09-17", "H11（BCE主导MA头）", "相对H4：只把损失配比改为BCE主导。", "MA 0.2510<0.2522。"),
    ("2026-09-17", "H12/H13（整图监督）", "相对H4：只把crop级监督换成整图级监督，H13为独立test确认。", "H12验证MA 0.2565、测试0.2803；H13整体44.05%<44.21%。"),
    ("2026-09-17", "P5（三种子复核）", "相对H4：只换随机种子为1/2/3。", "MA 0.2510/0.2537/0.2514，散布0.0027。"),
    ("2026-09-17", "T0/T1（采样对照）", "相对D：T0沿用旧前景采样；T1只改为“2均匀+1MA阳性+1基准误报窗”。", "报告所述：MA峰值25.4468/25.4733，终点24.7038/24.7868；本地无原始run，不计结论。"),
    ("2026-09-17", "J0/J1（解码续训）", "相对D@20k：J0只训decoder；J1只额外解冻stage4；均8k。", "报告所述：J0约40.50%、J1约40.42%，均无提升。"),
    ("2026-09-17", "U0v2（多尺度MA网）", "相对H4：只把3361参数头换成194万参数多尺度校正网，8000步。", "报告所述：MA终点24.7509%，四类42.3643%。"),
    ("2026-09-17", "C1/F1（MA融合16候选）", "相对冠军：只在MA通道做16组融合权重并锁定U0/T1各0.5。", "报告所述：MA 24.8429%、合成四类42.3873%。"),
    ("2026-09-17", "R1-A（纯学生监督）", "相对D@2000：只换损失为Dice(student)，训练4000更新。", "验证0.4152/0.4162/0.4167/0.4131（1k至4k）。"),
    ("2026-09-17", "R1-B（融合感知）", "相对R1-A：只把损失换成0.5 Dice(stu)+0.5 Dice(0.5stu+0.5ref)。", "最佳0.4175；较A高0.08pp，低于0.3pp门槛。"),
    ("2026-09-18", "R5-E0（第六成员）", "相对冠军：只把R1-B学生作为第六成员加入，权重1或2，不重训。", "验证0.4273/0.4278/0.4272（冠军0.4267）。"),
    ("2026-09-18", "R5-E1-V1（纯融合w=1）", "相对R1-B：只把融合权重从0.5固定为1.0，3000步。", "验证0.4187@3000，过G1门。"),
    ("2026-09-18", "R5-E1-V2（课程w）", "相对V1：只把固定w=1改为0→0.5→1.0课程。", "验证0.4181，低于V1。"),
    ("2026-09-18", "R5-E2（换seed）", "相对V1：只换随机种子42→43。", "验证0.4178，未复现V1。"),
    ("2026-09-18", "R5-E3（学生入冠军）", "相对R5-E0：只把新增成员从BNEW换成V1学生。", "验证0.4273，仅+0.0006，未过门。"),
    ("2026-09-18", "R6-P0（冠军复现）", "无训练：按原配方复算并做逐图错误画像。", "验证0.4267、测试0.4421。"),
    ("2026-09-18", "R6-P1-HE（HE校正头）", "相对H4：只把校正目标换成HE，主干与融合冻结。", "0.4360<0.4376，拒绝。"),
    ("2026-09-18", "R6-P1-EX（EX校正头）", "相对P1-HE：只把目标换成EX。", "缓存建到50/76预算到期，跳过，未完成。"),
    ("2026-09-18", "R6-P2-M1（RGB MA专家）", "相对冠军：不加后融合头，改为从RGB零训练512²MA专家，再做α融合。", "patch IoU峰值0.0220；α=0.3时MA 0.2438<0.2597。"),
    ("2026-09-18", "R6b（256²+Tversky）", "相对P2-M1：只把窗口512→256、损失改Tversky(0.3,0.7)、2000步。", "patch best 0.0563；α=0.3时MA 0.2431（低1.66pp）。"),
    ("2026-09-18", "R6-P3（终局对照）", "无训练：冠军再推理一次，验证复现性。", "验证0.4267、测试0.4421，与P0一致。"),
]

T2_NEW = [
    ("2026-09-19", "C_S-C（普通混合裁剪）", "相对D@2000：只把采样换成普通混合裁剪，4000步。", "C@2k 39.7974%、C@4k 39.1729%。"),
    ("2026-09-19", "C_S-S（类平衡实例裁剪）", "相对C：只把采样换成类平衡实例裁剪。", "S@2k 38.6455%、S@4k 39.8320%；S@4k比C@4k高0.66pp。"),
    ("2026-09-19", "MIX-C（冠军+C@2k）", "相对冠军：只把25%成员换成C@2k，不重训。", "验证42.8657%；测试43.8064%，比冠军低0.4052pp。"),
    ("2026-09-19", "refiner_v1（先验残差头）", "相对冠军：加冠军先验MA/HE残差头；Stage1仅8图200轮。", "Stage1 41.9112%为过拟合；Stage2第12597步身份断链，未完成。"),
    ("2026-09-20", "U0-REAL（ConvNeXt+U-Net）", "相对W48：换ConvNeXt-S+U-Net、ce+dice、整图、12k。", "验证38.4464%@12k（3k 17.48升至12k 38.45，仍上升）。"),
    ("2026-09-20", "H0（W48 DDR-only）", "相对B0：fresh ImageNet W48、2048输入、eff batch8、计划12k。", "6k停止；验证36.9226%（EX41.88/HE43.52/SE41.45/MA20.84）。"),
    ("2026-09-20", "FC-EXPERT-COMP（互补性）", "无训练：8个模型乘76图逐像素互补与面积分桶。", "逐类argmax 43.0645%（高0.3951）；oracle不可部署。"),
    ("2026-10-02", "U1（ConvNeXt外部预训练）", "相对U0：只加IDRiD多边形栅格化48/6外部预训练2.4k，再迁移DDR。", "外部dev 27.1605%；DDR验证36.6934%，低于U0。"),
    ("2026-10-02", "H1（W48外部预训练）", "相对H0：只加同一外部预训练2.4k，再跑DDR 12k。", "外部dev 42.9998%；DDR最佳35.5950%@9k、末次35.4032%。"),
    ("2026-10-02", "A0（HACDR骨干）", "相对H0：只把W48换成HACDR官方骨干并接项目解码。", "best 9.6781%@6k，HE/MA检出塌陷；6197步停止。"),
    ("2026-10-03", "U0_cont（2048续训）", "相对U0@12k：只把基准长边固定2048继续2000步。", "单模39.0864%；固定融合43.3663%（低0.1214）。"),
    ("2026-10-03", "U0_HR（2048/2560混合）", "相对U0_cont：只把基准长边改为2048/2560随机。", "单模39.6065%；固定融合43.7328%（高0.2451）。"),
    ("2026-10-03", "G（固定25%最高候选）", "相对旧部署：只把25%成员换成U0_HR@1000，冻结后一次test。", "验证43.8679%；test 44.6345%（低0.1464），区间[-0.4094,+0.0948]含0。"),
    ("2026-10-03", "M0（全局语义适配）", "相对U0_HR：加全局语义adapter，2000步。", "单模41.0373%；半融合42.3552%。"),
    ("2026-10-03", "M1（M0+RGB跳连）", "相对M0：只加stride1/2真实RGB跳连。", "单模41.0441%；半融合42.3459%，未优于M0。"),
    ("2026-10-04", "Q_DDR（DINOv3分支）", "相对冠军：只加DINOv3 ViT-B/16+DPT分支（头部与最后4块），16k。", "单模37.8937%；固定25% 44.3557%；test 45.7216%。"),
    ("2026-10-04", "Q_MIX（加外部数据）", "相对Q_DDR：只把数据改为4DDR+2IDRiD+2MAPLES，24k。", "未胜出，未进入融合。"),
    ("2026-10-04", "E_NEG（负类语义）", "相对Q：只加负类语义监督，24k。", "单模37.0094%；融合44.1643%，低于Q。"),
    ("2026-10-05", "F4（解冻4块）", "相对Q@16k：全主干追加12k，只解冻4块。", "单模38.7411%；融合44.3069%。"),
    ("2026-10-05", "F12（解冻12块）", "相对F4：只把解冻4块改为12块。", "单模40.2418%；融合44.5888%；成为四臂父模型。"),
    ("2026-10-06", "四臂C/D/SD（S臂见表内S20）", "相对F12：C=只续训+EMA；D=加TJDR独立头；SD=空间交互+D；S=空间交互（即S20来源）。", "C 20k 41.0244/45.1762；D 41.9082/45.1010；SD 41.3796/44.8832；均低于S的42.2703/45.3218。"),
    ("2026-10-08", "Arch-v3（机制重构）", "相对W48基线：语义解码+零残差细节支路，计划40k。", "20k主动停止；best 37.9882%@8k；末次15.0509%，EX/SE归零。"),
    ("2026-10-09", "CET（训练目标）", "相对v3：只把训练目标换成CET（GT+KD+aux）。", "完成40k 38.3138%；best 40.4876%@32k；低于同预算38.835%。"),
    ("2026-10-09", "SACD（动作纠错）", "相对CET：只把监督换成动作/纠错读出，20k。", "20k 37.1848%。"),
    ("2026-10-09", "Residual（连续残差）", "相对SACD：去掉动作KL，改detach连续FP32残差。", "8k 37.2124%；私有收益0.2782<0.5门，拒绝。"),
    ("2026-10-09", "Expert（私有分割）", "相对Residual：改独立私有分割+可学习logit融合。", "4k 27.7377%；MA 2.8132%。"),
    ("2026-10-09", "Adapter（1/4网格）", "相对Expert：改1/4网格直接多尺度纠错。", "4k 30.4442%；差floor 0.0316被拒。"),
    ("2026-10-09", "Detail（1/2细节网格）", "相对Adapter：只把1/4网格改为1/2细节网格。", "12k 36.7358%后16k跌至13.0748%（低23.66），20k 15.2038%。"),
    ("2026-10-09", "Foundation（阳性梯度保护）", "相对Detail：加基础GT平衡BCE与GT一致教师蒸馏。", "4k 31.6795%；MA 12.8671%，私有收益为负。"),
    ("2026-10-09", "Aligned（正负组平衡）", "相对Foundation：只把私有BCE/KL改为按全GT正负0.5/0.5。", "4k 31.7823%；SE跌超2个百分点。"),
    ("2026-10-10", "Semantic（语义拼接）", "相对Aligned：改尺度保留拼接+raw-logit旁路+分支裁剪。", "4k 32.8890%；私有收益不足1点，HE/MA仍有收益。"),
    ("2026-10-10", "IoU（Lovasz hinge）", "相对Semantic：只把私有Dice换成binary Lovasz hinge。", "4k 26.2690%；四类精度全升、召回全降。"),
    ("2026-10-16", "NativePyramid v3", "相对W48基线：2048直入+四尺度分类+细节支路+新解码器，40k。", "40k 39.4256%；best 40.2770%@36k；MA 20.50%；Z减S为负0.0566。"),
]

n1 = rebuild(doc.tables[0], T1_NEW)
n2 = rebuild(doc.tables[1], T2_NEW)
print("table1 body rows:", n1, "table2 body rows:", n2)

doc.save(DOC)
d = docx.Document(DOC)
print("final tables:", [len(t.rows) for t in d.tables])
print("paras:", len(d.paragraphs))
