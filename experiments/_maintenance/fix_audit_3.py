# -*- coding: utf-8 -*-
"""第三轮：口径标注、冠军改称 H5、去除重复升级链、标题顿号。"""
import copy
import docx
from docx.oxml.ns import qn

DOC = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
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


def row_by_name(table, name):
    for row in table.rows:
        if row.cells[1].text.strip() == name:
            return row
    return None


t1, t2 = d.tables[0], d.tables[1]

cell_set(row_by_name(t1, "HE/SE 校正头（H6，首次）").cells[3],
         "验证 HE 43.56%、SE 51.44%，均低于两路融合的 44.73%/55.08%。")
cell_set(row_by_name(t1, "BCE 主导 HE 头（H9）").cells[3], "验证 HE 43.82%，低于融合的 44.73%。")
cell_set(row_by_name(t1, "BCE 主导 MA 头（H11）").cells[3], "验证 MA 25.10%，低于 H4 的 25.22%。")
cell_set(row_by_name(t1, "延长头训练（H8）").cells[3], "测试 MA 28.58%，低于 H5 的 28.66%。")
cell_set(row_by_name(t1, "三种子复核（P5）").cells[3], "验证 MA 25.10%/25.37%/25.14%。")
cell_set(row_by_name(t1, "MA 权重扫描（H3）").cells[3], "验证 MA +0.47、mIoU +0.12 个百分点。")
cell_set(row_by_name(t1, "集成与 TTA 矩阵").cells[3],
         "测试最高 43.229%；D×3+B+E4 43.149%；翻转 42.717%、多尺度 42.797%、双尺度翻转 42.805%。")
cell_set(row_by_name(t1, "采样对照（T0/T1）").cells[3],
         "验证 MA 峰值 25.4468%/25.4733%；续训后四类 42.3735%/42.3654%。")
cell_set(row_by_name(t1, "多尺度 MA 网（U0v2）").cells[3], "验证 MA 24.7509%，四类 42.3643%。")
cell_set(row_by_name(t1, "MA 融合 16 候选（C1/F1）").cells[3], "验证 MA 24.8429%，四类 42.3873%。")
cell_set(row_by_name(t1, "HE 校正头复核（R6-P1）").cells[3], "验证 43.60%，低于同轮基线 43.76%。")
cell_set(row_by_name(t1, "RGB MA 专家（R6-P2-M1）").cells[3], "patch IoU 2.20%；验证 MA 24.38%。")
cell_set(row_by_name(t1, "256² MA 专家（R6b）").cells[3], "patch 5.63%；验证 MA 24.31%。")
cell_set(row_by_name(t1, "学生入冠军（R5-E3）").cells[1], "学生入 H5（R5-E3）")
cell_set(row_by_name(t2, "冠军加 C@2k（25%）").cells[1], "H5 加 C@2k（25%）")
cell_set(row_by_name(t2, "冠军先验残差头（refiner_v1）").cells[1], "H5 先验残差头（refiner_v1）")

REP = [
    ("再往后，MA 权重扫描、集成成员增减、多尺度与翻转组合多数落在 42 至 43 之间；H5 的四模型融合把测试推到 44.21%，再加上 ConvNeXt S 与 DINOv3 分支后升到 45.86%。",
     "再往后，MA 权重扫描、集成成员增减、多尺度与翻转组合多数落在 42 至 43 之间，H5 的四模型融合才把测试推到 44.21%。"),
    ("其 MA IoU 比原预处理 L0 高约 1.12 个百分点",
     "其 MA IoU 比原预处理对照组 L0 高约 1.12 个百分点"),
    ("测试 MA 28.58%，低于冠军的 28.66%", "测试 MA 28.58%，低于 H5 的 28.66%"),
    ("分别为 43.82% 和 25.10%，都低于 H4 的对应通道。",
     "HE 为 43.82%、MA 为 25.10%，都低于各自基线 44.73% 与 25.22%。"),
    ("P5 用三个种子重训 H4 头，MA 为", "P5 用三个种子重训 H4 头，验证 MA 为"),
    ("refiner_v1 是冠军先验残差头的第一版", "refiner_v1 是 H5 先验残差头的第一版"),
    ("三 损失预处理和训练策略的结果解释", "三 损失、预处理和训练策略的结果解释"),
]
for old, new in REP:
    hit = False
    for p in d.paragraphs:
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new)
                hit = True
                break
        if hit:
            break
    if not hit:
        print("MISS:", old[:26])

# 整段替换 P17（其余对照组）
P17_OLD_START = "其余对照组一并说明："
P17_NEW = ("未单独成段的对照如下：E3 训练未跑完；学生 BN 重估验证 40.363%，全分支重估 25.286%；"
           "C1/F1 的 16 组 MA 融合验证 24.8429%；R1-A 与 R1-B 验证 41.67% 与 41.75%，"
           "R5 五组验证 41.78% 至 42.78%，R6 的 HE 头验证 43.60%，512² 与 256² 两个 MA 专家的 patch IoU 为 2.20% 与 5.63%；"
           "普通采样 4k 验证 39.17%、类平衡采样 4k 验证 39.83%，后者高 0.66 个百分点，H5 加 C@2k 的测试为 43.81%；"
           "U0 异构模型验证 38.45%、H0 验证 36.92%，8 模型互补分析的最好逐类组合验证 43.06%。"
           "DINOv3 分支的外部数据版未胜出，负类语义版验证 44.16%，解冻 4 块与 12 块验证 44.31% 与 44.59%，"
           "四臂中的 C、D、SD 验证 45.18%、45.10%、44.88%。")

# 整段替换 P19（11 条候选）
P19_NEW = ("同学的 11 条候选架构中，除 CET 跑满 40k 外，其余 10 条都在 4k 至 20k 停止；"
           "CET 固定 40k 终点为 38.31%，低于原模型 40k 的 38.84%；"
           "10 月 16 日的原生金字塔 40k 为 39.43%，与历史 B0 的 39.39% 基本持平。"
           "这支持把下一步放在评测协议，而不是继续叠加模块。")

for p in d.paragraphs:
    is17 = p.text.startswith(P17_OLD_START)
    is19 = p.text.startswith("同学的 11 条候选架构")
    if not (is17 or is19):
        continue
    new_text = P17_NEW if is17 else P19_NEW
    runs = p.runs
    for r in runs[1:]:
        r._r.getparent().remove(r._r)
    runs[0].text = new_text
    print("replaced:", new_text[:18])

d.save(DOC)
d2 = docx.Document(DOC)
print("paras", len(d2.paragraphs), "tables", [len(t.rows) for t in d2.tables])
