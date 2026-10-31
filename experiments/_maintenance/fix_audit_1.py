# -*- coding: utf-8 -*-
"""按审查结论修复：重复行、时间顺序、事实表述、标题字体、文末段落。"""
import copy
import re
import docx
from docx.oxml.ns import qn

DOC = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
BLUE = "F4F7FA"


d = docx.Document(DOC)


def cell_set(cell, text):
    ps = cell.paragraphs
    for p in ps[1:]:
        p._p.getparent().remove(p._p)
    p = ps[0]
    rPr = None
    if p.runs and p.runs[0]._r.find(qn("w:rPr")) is not None:
        rPr = copy.deepcopy(p.runs[0]._r.find(qn("w:rPr")))
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    r = p._p.makeelement(qn("w:r"), {})
    if rPr is not None:
        r.append(copy.deepcopy(rPr))
    t = p._p.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    p._p.append(r)


def find_row(table, name):
    for row in table.rows:
        if row.cells[1].text.strip() == name:
            return row
    return None


def date_key(text):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})-(\d{2})", text)
    if m:
        return (int(m.group(1)), int(m.group(2)), 5)
    return (9999, 12, 31)


def resort(table):
    tbl = table._tbl
    trs = tbl.findall(qn("w:tr"))
    header, body = trs[0], trs[1:]
    items = []
    for idx, tr in enumerate(body):
        txt = "".join(n.text or "" for n in tr.findall(qn("w:tc"))[0].iter(qn("w:t")))
        items.append((date_key(txt), idx, tr))
    items.sort(key=lambda x: (x[0], x[1]))
    for tr in trs:
        tbl.remove(tr)
    tbl.append(header)
    for _, _, tr in items:
        tbl.append(tr)


def band(table, first_white=True):
    for idx, row in enumerate(table.rows[1:]):
        want = (idx % 2 == 1) if first_white else (idx % 2 == 0)
        for c in row.cells:
            tcPr = c._tc.tcPr
            if tcPr is None:
                continue
            for old in tcPr.findall(qn("w:shd")):
                tcPr.remove(old)
            if want:
                shd = tcPr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): BLUE})
                tcPr.insert_element_before(shd, "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign", "w:hideMark")


t1, t2 = d.tables[0], d.tables[1]

# 1) 时间标注：H4 与 H5 归位
r = find_row(t1, "H4 校正头")
if r is not None:
    cell_set(r.cells[0], "2026-09-16")
r = find_row(t1, "H5 共同系统")
if r is not None:
    cell_set(r.cells[0], "2026-09-17 至 18")

# 2) 删除与 H5 完全同值的“冠军复现”
r = find_row(t1, "冠军复现（R6-P0/P3）")
if r is not None:
    t1._tbl.remove(r._tr)

# 3) 删除与 R03 S2 完全同值的 R04 L0，L2 改为直接对 S2 配对
r = find_row(t2, "R04 S2 P0 L0")
if r is not None:
    t2._tbl.remove(r._tr)
r = find_row(t2, "R04 S2 P0 L2")
if r is not None:
    cell_set(r.cells[2], "与 R03 S2 配对，只把损失换成 L2。")

# 4) 措辞与口径
r = find_row(t2, "R03 S3b v2")
if r is not None:
    cell_set(r.cells[3], "最佳/末次 41.580%/41.540%；解冻后与 S2 最佳持平。")

r = find_row(t1, "HE 校正头复核（R6-P1）")
if r is not None:
    cell_set(r.cells[2], "在 9 月 18 日复核 HE 头，其余同 H6。")

r = find_row(t2, "8 模型互补分析")
if r is not None:
    cell_set(r.cells[2], "8 个模型 × 76 图逐像素互补。")

r = find_row(t1, "第六成员（R5-E0）")
if r is not None:
    cell_set(r.cells[3], "验证 42.73%–42.78%。")

# 5) 表一按日期重排
resort(t1)
band(t1, first_white=True)
band(t2, first_white=False)

# 6) 正文事实修正
fixes = [
    ("再往后，MA 权重扫描、集成成员增减、多尺度与翻转组合都只在 42 附近，直到引入 ConvNeXt S 与 DINOv3 分支才越过 44%。",
     "再往后，MA 权重扫描、集成成员增减、多尺度与翻转组合都只在 42 附近；H5 的四模型融合把测试推到 44.21%，再加上 ConvNeXt S 与 DINOv3 分支后升到 45.86%。"),
    ("同学侧 10 月的 11 条候选架构在 4k 筛查中全部未超过基线，唯一跑完 40k 的 CET 为 38.31%，也低于同预算的原模型；这支持把下一步放在评测协议，而不是继续叠加模块。",
     "同学侧 10 月的 11 条候选架构都在 4k 或 20k 筛查中停止，唯一跑完 40k 的 CET 为 38.31%，低于同预算原模型的 38.84%；这支持把下一步放在评测协议，而不是继续叠加模块。"),
    ("R04 统一起点、数据、种子、冻结策略和训练日程后，L2 为 41.57%、L0 为 41.58%，早期小幅优势未复现。",
     "R04 统一起点、数据、种子、冻结策略和训练日程后，L2 为 41.57%，与 S2 的 41.58% 持平，早期小幅优势未复现。"),
    ("R03 的 S0 至 S3 是逐级加严的冻结策略", "R03 的 S0 至 S3 是逐步调整学习率与冻结范围的策略"),
    ("H3 至 H13 是接在融合输出后的校正头及其变体", "H3 至 H13 是融合后处理的一组实验，包括 MA 权重扫描与各类校正头"),
]
for old, new in fixes:
    for p in d.paragraphs:
        hit = False
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new)
                hit = True
        if hit:
            break

# 7) 附录标题补回显式字体
for p in d.paragraphs:
    if p.text.strip() == "附录 按时间线整理的实验表":
        for r in p.runs:
            rPr = r._r.get_or_add_rPr()
            rf = rPr.find(qn("w:rFonts"))
            if rf is None:
                rf = rPr.makeelement(qn("w:rFonts"), {})
                rPr.insert(0, rf)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(a), "Arial Unicode MS")
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]

# 8) 表二后补一个收尾段落
body = d.element.body
sect = body.find(qn("w:sectPr"))
last_tbl = [c for c in body if c.tag == qn("w:tbl")][-1]
nxt = last_tbl.getnext()
if nxt is None or nxt.tag != qn("w:p"):
    p = body.makeelement(qn("w:p"), {})
    last_tbl.addnext(p)

d.save(DOC)
d2 = docx.Document(DOC)
print("tables", [len(t.rows) for t in d2.tables], "paras", len(d2.paragraphs))
for i, r in enumerate(d2.tables[0].rows):
    if i < 3 or 12 <= i <= 20:
        print(i + 1, r.cells[0].text, "|", r.cells[1].text)
