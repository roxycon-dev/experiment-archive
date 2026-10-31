# -*- coding: utf-8 -*-
"""在第二、三节末尾各补一段同学侧负结果概述，保持原有正文格式。"""
import copy
import docx
from docx.oxml.ns import qn

DOC = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"

T2_TEXT = ("在系统成型之前，同学侧还试过几条中间路线：先做只改 MA 的后融合残差头（E、E2），"
           "再比较逐类路由与等权概率平均（ROUTE、ENS），最后才转向跨架构分支。这些尝试的单模型或集成结果"
           "多停留在 41.8% 至 42.6%，只有把异构模型整体纳入概率系统后才越过 44%。它们的共同点是改动集中在"
           "输出端或融合端，主干与训练目标没有同步变化，因此收益随系统规模放大而衰减。逐条条件与数值见附录表一、表二。")

T3_TEXT = ("同学侧在 9 月还平行做过一组损失与训练策略的对照，结果同样为负：MA 后融合头换损失配比（H9、H11）、"
           "把头的训练从 2000 步延长到 6000 步（H8）、把裁块监督改成整图监督（H12、H13）、更换采样顺序（T0、T1）、"
           "只训解码器或额外解冻 stage4（J0、J1），都没有超过各自起点；集成成员从 5 个增加到 6 个（R5-E0）"
           "也只多 0.06 个百分点。唯一例外是 H4 这种只改单一通道、参数量仅 3361 的窄头，而它一旦推广到 HE、SE 就失效。"
           "这些结果与 R02 至 R04 指向同一结论：在 76 张验证图上，小于 1 个百分点的差异无法稳定分辨。")


def insert_after(anchor, text):
    new_p = copy.deepcopy(anchor)
    runs = new_p.findall(qn("w:r"))
    tmpl_rPr = None
    if runs:
        r0 = runs[0]
        if r0.find(qn("w:rPr")) is not None:
            tmpl_rPr = copy.deepcopy(r0.find(qn("w:rPr")))
        for r in runs:
            new_p.remove(r)
    # 清掉可能的书签/批注引用
    for tag in ("w:bookmarkStart", "w:bookmarkEnd", "w:commentRangeStart", "w:commentRangeEnd"):
        for el in new_p.findall(qn(tag)):
            new_p.remove(el)
    r = new_p.makeelement(qn("w:r"), {})
    if tmpl_rPr is not None:
        r.append(copy.deepcopy(tmpl_rPr))
    t = new_p.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    new_p.append(r)
    anchor.addnext(new_p)


doc = docx.Document(DOC)
paras = doc.paragraphs

anchor2 = None
anchor3 = None
for p in paras:
    if p.text.startswith("同学随后引入 ConvNeXt S"):
        anchor2 = p
    if p.text.startswith("R03 专门检查成熟权重的继续训练"):
        anchor3 = p
assert anchor2 is not None and anchor3 is not None

insert_after(anchor3._p, T3_TEXT)   # 先插后面的，避免锚点失效
insert_after(anchor2._p, T2_TEXT)

doc.save(DOC)

d = docx.Document(DOC)
print("paras:", len(d.paragraphs), "tables:", [len(t.rows) for t in d.tables])
for p in d.paragraphs:
    if p.text.startswith("在系统成型之前") or p.text.startswith("同学侧在 9 月"):
        print("OK:", p.text[:40], "...", "style=", p.style.name)
