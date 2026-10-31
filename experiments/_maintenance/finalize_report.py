# -*- coding: utf-8 -*-
"""按导师可读标准重做：删掉两段防御性概述，正文一二三四五补入失败实验，
表格去掉重复项、压缩到与原文同等长度。"""
import copy
import re
import docx
from docx.oxml.ns import qn

DOC = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
BAK = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

doc = docx.Document(DOC)
bak = docx.Document(BAK)
body = doc.element.body


def ptext(el):
    return "".join(n.text or "" for n in el.iter(qn("w:t"))).strip()


# ---------- 1) 删除上一版插入的两段 ----------
for p in [c for c in body if c.tag == qn("w:p")]:
    t = ptext(p)
    if t.startswith("在系统成型之前") or t.startswith("同学侧在 9 月还平行"):
        body.remove(p)
print("removed old bridge paragraphs")

paras = doc.paragraphs


def append_sentence(p, text):
    last_r = p.runs[-1]
    r = copy.deepcopy(last_r._r)
    for tnode in r.findall(qn("w:t")):
        r.remove(tnode)
    t = r.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    p._p.append(r)


def insert_after(anchor, text):
    new_p = copy.deepcopy(anchor)
    runs = new_p.findall(qn("w:r"))
    rPr = None
    if runs and runs[0].find(qn("w:rPr")) is not None:
        rPr = copy.deepcopy(runs[0].find(qn("w:rPr")))
    for r in runs:
        new_p.remove(r)
    r = new_p.makeelement(qn("w:r"), {})
    if rPr is not None:
        r.append(copy.deepcopy(rPr))
    t = new_p.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    new_p.append(r)
    anchor.addnext(new_p)


# ---------- 2) 正文补写 ----------
P_S1 = ("主线之外，后期还对 SE 增强、MA 细节分支、逐类路由、外部预训练与十余种纠错头做过尝试，"
        "均未超过当时的最好系统，逐条列在附录表一、表二。")

P2A = ("在 H5 之前，同学侧先做了两版 MA 后融合头：E 接在 B 上、E2 接在 D 上，测试分别为 42.559% 和 42.544%；"
       "随后比较按验证集逐类选模型的 41.82% 与两模型等权平均的 42.35%，等权平均成为第一个超过基线的方案。"
       "D 用 1024 裁块训练学生网络，推理时与冻结 B 的整图概率各半融合，验证 41.615%、测试 42.58%，"
       "之后的集成都以它为起点。再往后，MA 权重扫描、集成成员增减、多尺度与翻转组合都只在 42 附近，"
       "直到引入 ConvNeXt S 与 DINOv3 分支才越过 44%。")

P2B = ("跨架构分支并非一次到位。U1 用 IDRiD 多边形栅格化的 48 张训练、6 张验证做外部预训练，外部 dev 27.16%，"
       "迁移到 DDR 后验证 36.69%；H1 用同一套数据预训练 W48，外部 dev 43.00%，迁移后最好只有 35.60%；"
       "A0 换 HACDR 骨干，6k 时 9.68%，HE 与 MA 几乎检不出。高分辨率续训 U0_cont 与 U0_HR 的融合验证"
       "为 43.37% 与 43.73%，按此选出的 G 在测试上为 44.63%，低于原管线。M0、M1 的全局语义适配与 "
       "RGB 跳连单模 41.04%，也没有超过原点。")

P3 = ("同学侧同期还比较过损失与训练策略。L 把 D 的 Dice 换成 Dice+0.1 Lovasz，验证 41.056%、测试 41.777%；"
      "H8 把 MA 头训练延长到 6000 步并把学习率提高到 2e-3，测试 MA 0.2858，低于冠军的 0.2866；"
      "H9、H11 把 HE、MA 头的损失改成 BCE 主导，分别为 0.4382 和 0.2510，都低于原配方。"
      "H12 改成整图级监督，测试确认 44.05%，低于 44.21%；P5 用三个种子重训 H4 头，"
      "MA 为 0.2510、0.2537、0.2514。采样方面，T0 与 T1 的 MA 峰值为 25.4468% 和 25.4733%；"
      "J0、J1 只训解码器或额外解冻 stage4，验证约 40.50% 和 40.42%；U0v2 把 3361 参数的头换成"
      "194 万参数的多尺度校正网，MA 终点 24.7509%。这些改动都没有超过各自的起点。")

P4 = ("同学侧 10 月的 11 条候选架构在 4k 筛查中全部未超过基线，唯一跑完 40k 的 CET 为 38.31%，"
      "也低于同预算的原模型；这支持把下一步放在评测协议，而不是继续叠加模块。")

P5 = ("附录两张表收录双方从 6 月到 10 月的全部实验，包括未达预期的组。")

anchor_s1 = anchor_s2 = anchor_s3 = anchor_s4 = anchor_s5 = None
for p in doc.paragraphs:
    t = p.text
    if t.startswith("中期计划提出多架构横向比较"):
        anchor_s1 = p
    if t.startswith("同学随后引入 ConvNeXt S"):
        anchor_s2 = p
    if t.startswith("R03 专门检查成熟权重的继续训练"):
        anchor_s3 = p
    if t.startswith("本期数值主要来自项目自定义"):
        anchor_s4 = p
    if t.startswith("中期数据取自 2026 年 6 月 6 日"):
        anchor_s5 = p
assert all([anchor_s1, anchor_s2, anchor_s3, anchor_s4, anchor_s5])

append_sentence(anchor_s1, P_S1)
insert_after(anchor_s2._p, P2B)   # 先插后一段
insert_after(anchor_s2._p, P2A)
insert_after(anchor_s3._p, P3)
append_sentence(anchor_s4, P4)
append_sentence(anchor_s5, P5)

# ---------- 3) 重建表格 ----------
def set_cell_text(tc, text):
    ps = tc.findall(qn("w:p"))
    for p in ps[1:]:
        tc.remove(p)
    p = ps[0]
    rPr = None
    r0 = p.find(qn("w:r"))
    if r0 is not None and r0.find(qn("w:rPr")) is not None:
        rPr = copy.deepcopy(r0.find(qn("w:rPr")))
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    r = p.makeelement(qn("w:r"), {})
    if rPr is not None:
        r.append(copy.deepcopy(rPr))
    t = p.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
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


def rebuild(table, keep_set, new_rows):
    tbl = table._tbl
    trs = tbl.findall(qn("w:tr"))
    header, body_rows = trs[0], trs[1:]
    kept = []
    for tr in body_rows:
        cells = tr.findall(qn("w:tc"))
        key = tuple("".join(n.text or "" for n in c.iter(qn("w:t"))) for c in cells)
        if key in keep_set:
            kept.append(tr)
    template = copy.deepcopy(body_rows[-1])
    items = [(date_key("".join(n.text or "" for n in tr.findall(qn("w:tc"))[0].iter(qn("w:t")))), 0, tr) for tr in kept]
    for row in new_rows:
        tr = copy.deepcopy(template)
        for j, val in enumerate(row):
            set_cell_text(tr.findall(qn("w:tc"))[j], val)
        items.append((date_key(row[0]), 1, tr))
    items.sort(key=lambda x: (x[0], x[1]))
    for tr in trs:
        tbl.remove(tr)
    tbl.append(header)
    for _, _, tr in items:
        tbl.append(tr)
    return len(kept), len(new_rows)


def orig_keys(tbl):
    return {tuple("".join(n.text or "" for n in c.iter(qn("w:t"))) for c in tr.findall(qn("w:tc")))
            for tr in tbl.findall(qn("w:tr"))[1:]}


K1 = orig_keys(bak.tables[0]._tbl)
K2 = orig_keys(bak.tables[1]._tbl)

T1_NEW = [
    ("2026-09-11", "语义引导MA残差（A）", "从E4@36k起，只加MA残差分支与0.02辅助损失。", "验证38.632%@1k，末次38.265%，未超起点。"),
    ("2026-09-11", "无分支对照（B）", "与A同起点同损失，只删残差分支。", "验证39.205%@3k；测试41.92%。"),
    ("2026-09-11", "逐类路由（E4/B）", "不训练，按验证IoU逐类选E4或B的通道。", "验证39.90%；测试41.82%，HE排名反转。"),
    ("2026-09-11", "等权融合（E4+B）", "取消逐类选择，改0.5/0.5概率平均。", "验证39.54%；测试42.35%，首次超过基线。"),
    ("2026-09-13", "局部学生（D）", "从B@3000起，改1024裁块训练，推理与B各半。", "验证41.615%@2k；测试42.58%。"),
    ("2026-09-15", "B起点MA头（E）", "在B上加7通道MA头，只训新头。", "验证40.255%@1500；测试42.559%。"),
    ("2026-09-15", "D起点MA头（E2）", "同结构，只把底座换成D。", "测试42.544%。"),
    ("2026-09-15", "第三MA臂（E3）", "同结构，换第三组初始化。", "训练未完成，无验证结果。"),
    ("2026-09-15", "Dice+0.1Lovasz（L）", "在D上只换损失为Dice+0.1 Lovasz。", "验证41.056%；测试41.777%。"),
    ("2026-09-16", "集成与TTA矩阵", "在D上只改融合成员、权重与翻转。", "最高43.229%；D×3+B+E4 43.149%。"),
    ("2026-09-16", "学生BN重估", "只重估D学生分支BN统计。", "验证40.363%，低于41.615%。"),
    ("2026-09-16", "全分支BN重估", "把重估扩到全部分支。", "验证25.286%，崩溃。"),
    ("2026-09-16", "MA权重扫描（H3）", "只扫MA通道融合权重0.5至1.0。", "MA+0.0047，mIoU+0.0012。"),
    ("2026-09-16", "HE/SE校正头（H6）", "把H4的MA头改成HE或SE。", "HE 0.4356、SE 0.5144，均低于融合值。"),
    ("2026-09-17", "延长头训练（H8）", "H4头改6000步、学习率2e-3。", "测试MA 0.2858，低于冠军0.2866。"),
    ("2026-09-17", "BCE主导HE头（H9）", "HE头损失改BCE/Dice=0.9/0.1。", "HE 0.4382，低于0.4473。"),
    ("2026-09-17", "BCE主导MA头（H11）", "MA头损失改BCE主导。", "MA 0.2510，低于0.2522。"),
    ("2026-09-17", "整图监督（H12/H13）", "把裁块监督改成整图级监督。", "H13测试44.05%，低于44.21%。"),
    ("2026-09-17", "三种子复核（P5）", "H4头只换种子1/2/3。", "MA 0.2510/0.2537/0.2514。"),
    ("2026-09-17", "采样对照（T0/T1）", "T1在T0上加MA阳性与基准误报窗。", "MA峰值25.4468%/25.4733%。"),
    ("2026-09-17", "解码续训（J0/J1）", "J0只训decoder，J1再解冻stage4。", "验证约40.50%/40.42%，无提升。"),
    ("2026-09-17", "多尺度MA网（U0v2）", "把3361参数头换成194万参数校正网。", "MA 24.7509%，四类42.3643%。"),
    ("2026-09-17", "MA融合16候选（C1/F1）", "只在MA通道做16组融合权重。", "MA 24.8429%，四类42.3873%。"),
    ("2026-09-17", "纯学生监督（R1-A）", "从D@2000起只用Dice学生损失，4000步。", "验证0.4167@3k。"),
    ("2026-09-17", "融合感知监督（R1-B）", "把参考概率并入学生损失。", "验证0.4175，比A高0.08pp。"),
    ("2026-09-18", "第六成员（R5-E0）", "把R1-B学生作为第六成员加入冠军。", "验证0.4273至0.4278。"),
    ("2026-09-18", "纯融合监督（R5-E1-V1）", "融合权重固定1.0，训练3000步。", "验证0.4187。"),
    ("2026-09-18", "课程权重（R5-E1-V2）", "权重改为0至0.5至1.0课程。", "验证0.4181，低于V1。"),
    ("2026-09-18", "第二种子（R5-E2）", "V1换seed 43。", "验证0.4178，未复现。"),
    ("2026-09-18", "学生入冠军（R5-E3）", "把V1学生替换BNEW加入冠军。", "验证0.4273，仅高0.0006。"),
    ("2026-09-18", "冠军复现（R6-P0/P3）", "按原配方复算并做错误画像。", "验证0.4267；测试0.4421。"),
    ("2026-09-18", "HE校正头（R6-P1-HE）", "把H4的MA头改成HE头。", "0.4360，低于0.4376。"),
    ("2026-09-18", "RGB MA专家（R6-P2-M1）", "从RGB零训练512² MA专家再融合。", "patch IoU 0.0220；MA 0.2438。"),
    ("2026-09-18", "256²MA专家（R6b）", "窗口改256，损失改Tversky。", "patch 0.0563；MA 0.2431。"),
]

T2_NEW = [
    ("2026-09-19", "普通采样（C_S-C）", "在D@2000上只换普通混合裁剪。", "C@2k 39.7974%、C@4k 39.1729%。"),
    ("2026-09-19", "类平衡采样（C_S-S）", "只换类平衡实例裁剪。", "S@4k 39.8320%，比C高0.66pp。"),
    ("2026-09-19", "冠军加C@2k（25%）", "只把25%成员换成C@2k。", "验证42.8657%；测试43.8064%。"),
    ("2026-09-19", "冠军先验残差头（refiner_v1）", "在冠军上加MA/HE残差头，仅8图200轮。", "Stage1 41.9112%，Stage2中断。"),
    ("2026-09-20", "异构U-Net（U0-REAL）", "ConvNeXt-S+U-Net、整图、12k。", "验证38.4464%@12k。"),
    ("2026-09-20", "W48 DDR-only（H0）", "fresh W48、2048、eff batch8，计划12k。", "6k停止，验证36.9226%。"),
    ("2026-09-20", "8模型互补分析", "8个模型乘76图逐像素互补。", "逐类argmax 43.0645%。"),
    ("2026-10-02", "ConvNeXt外部预训练（U1）", "IDRiD 48/6预训练2.4k后迁移DDR。", "外部dev 27.1605%；DDR 36.6934%。"),
    ("2026-10-02", "W48外部预训练（H1）", "W48同法预训练后迁移DDR。", "外部dev 42.9998%；DDR 35.5950%。"),
    ("2026-10-02", "HACDR骨干（A0）", "W48换HACDR官方骨干。", "best 9.6781%@6k，HE/MA塌陷。"),
    ("2026-10-03", "2048续训（U0_cont）", "U0长边固定2048继续2k。", "融合验证43.3663%。"),
    ("2026-10-03", "2048/2560混合（U0_HR）", "长边改2048/2560混合，4k。", "融合验证43.7328%。"),
    ("2026-10-03", "高分辨率25%候选（G）", "把25%成员换成U0_HR@1000。", "验证43.8679%；测试44.6345%。"),
    ("2026-10-03", "语义适配（M0）", "在U0_HR上加全局语义adapter。", "单模41.0373%；半融合42.3552%。"),
    ("2026-10-03", "RGB跳连（M1）", "在M0上再加stride1/2 RGB跳连。", "单模41.0441%；42.3459%。"),
    ("2026-10-04", "DINOv3第二种子（Q_DDR）", "Q_DDR换seed 2024重训16k。", "测试45.6771%，两种子均值45.6993%。"),
    ("2026-10-04", "外部数据版（Q_MIX）", "Q_DDR数据加2 IDRiD与2 MAPLES，24k。", "未超过Q_DDR。"),
    ("2026-10-04", "负类语义（E_NEG）", "在Q上加负类语义监督，24k。", "融合44.1643%，低于44.3557%。"),
    ("2026-10-05", "解冻4块（F4）", "Q@16k后全主干追加12k，解冻4块。", "融合44.3069%。"),
    ("2026-10-05", "解冻12块（F12）", "解冻12块。", "融合44.5888%，成为四臂父模型。"),
    ("2026-10-06", "四臂C/D/SD", "从F12起：C续训加EMA；D加TJDR头；SD为空间交互加D。", "C 45.1762%、D 45.1010%、SD 44.8832%。"),
    ("2026-10-08", "机制重构（Arch-v3）", "语义解码加零残差细节，计划40k。", "20k停止，best 37.9882%，末次15.05%。"),
    ("2026-10-09", "CET训练目标", "训练目标改为GT+KD+aux。", "40k 38.3138%，best 40.4876%@32k。"),
    ("2026-10-09", "动作纠错（SACD）", "监督改为动作纠错读出，20k。", "20k 37.1848%。"),
    ("2026-10-09", "连续残差（Residual）", "去动作KL，改detach连续残差。", "8k 37.2124%。"),
    ("2026-10-09", "私有分割（Expert）", "独立私有分割加可学习logit融合。", "4k 27.7377%，MA 2.8132%。"),
    ("2026-10-09", "1/4网格纠错（Adapter）", "1/4网格直接多尺度纠错。", "4k 30.4442%。"),
    ("2026-10-09", "1/2细节网格（Detail）", "1/2细节纠错网格。", "12k 36.7358%，16k崩至13.0748%。"),
    ("2026-10-09", "阳性梯度保护（Foundation）", "加基础GT平衡BCE与教师蒸馏。", "4k 31.6795%。"),
    ("2026-10-09", "正负组平衡（Aligned）", "私有BCE/KL按正负0.5/0.5。", "4k 31.7823%。"),
    ("2026-10-10", "语义拼接（Semantic）", "尺度保留拼接加raw-logit旁路。", "4k 32.8890%。"),
    ("2026-10-10", "Lovasz纠错（IoU）", "私有Dice换Lovasz hinge。", "4k 26.2690%。"),
    ("2026-10-16", "原生金字塔（NativePyramid v3）", "2048直入W48加四尺度分类与细节支路，40k。", "40k 39.4256%，best 40.2770%@36k。"),
]

print("t1", rebuild(doc.tables[0], K1, T1_NEW))
print("t2", rebuild(doc.tables[1], K2, T2_NEW))

doc.save(DOC)
d = docx.Document(DOC)
print("paras", len(d.paragraphs), "tables", [len(t.rows) for t in d.tables])
