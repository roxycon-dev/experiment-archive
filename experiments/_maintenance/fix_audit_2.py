# -*- coding: utf-8 -*-
"""第二轮：修正口径歧义、补全正文未提及的对照、修表格标签。"""
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

row_by_name(t1, "H5 共同系统").cells[3]
cell_set(row_by_name(t1, "H5 共同系统").cells[3],
         "验证 mIoU 42.6686%；测试 44.2110%（EX 53.86/HE 44.51/SE 49.82/MA 28.66）。共同成果只计一次。")
cell_set(row_by_name(t1, "HE/SE 校正头（H6，首次）").cells[3],
         "HE 43.56%、SE 51.44%，均低于融合后的 44.73%/55.08%。")
cell_set(row_by_name(t1, "BCE 主导 HE 头（H9）").cells[3], "HE 43.82%，低于融合的 44.73%。")
cell_set(row_by_name(t1, "BCE 主导 MA 头（H11）").cells[3], "MA 25.10%，低于 H4 的 25.22%。")
cell_set(row_by_name(t1, "HE 校正头复核（R6-P1）").cells[3], "43.60%，低于同轮基线 43.76%。")
cell_set(row_by_name(t1, "256²MA 专家（R6b）").cells[1], "256² MA 专家（R6b）")
cell_set(row_by_name(t1, "Dice+0.1Lovasz（L）").cells[1], "Dice+0.1 Lovasz（L）")
cell_set(row_by_name(t2, "四臂对照（续训·监督·联合）").cells[3],
         "C 45.1762%、D 45.1010%、SD 44.8832%（S 臂即 S20）。")

# 正文逐处替换
REP = [
    ("等权平均成为第一个超过基线的方案",
     "等权平均的测试为 42.35%，首次超过 42.20% 的基线"),
    ("之后的集成都以它为起点",
     "09-15 起的 MA 头与 H5 都以它为起点"),
    ("集成成员增减、多尺度与翻转组合都只在 42 附近",
     "集成成员增减、多尺度与翻转组合多数落在 42 至 43 之间"),
    ("按此选出的 G 在测试上为 44.63%，低于原管线",
     "按此选出的 G 在测试上为 44.63%，低于 10 月 2 日锁定的 44.78% 管线"),
    ("H12 改成整图级监督，测试确认 44.05%，低于 44.21%",
     "H12 改成整图级监督，H13 的测试确认为 44.05%，低于 44.21%"),
    ("以 EX、HE、SE、MA 四个独立 sigmoid 通道计算病灶 mIoU",
     "以 EX（硬性渗出）、HE（出血）、SE（软性渗出）、MA（微动脉瘤）四个独立 sigmoid 通道计算病灶 mIoU"),
    ("同学侧 10 月的 11 条候选架构都在 4k 或 20k 筛查中停止，唯一跑完 40k 的 CET 为 38.31%，低于同预算原模型的 38.84%；这支持把下一步放在评测协议，而不是继续叠加模块。",
     "同学侧 10 月的 11 条候选架构（机制重构、CET、SACD、Residual、Expert、Adapter、Detail、Foundation、Aligned、Semantic、IoU）都在 4k 至 20k 停止；CET 跑满 40k 为 38.31%，低于该轮同预算原模型的 38.84%，原生金字塔 40k 为 39.43%，也低于既有系统。这支持把下一步放在评测协议，而不是继续叠加模块。"),
    ("H3 至 H13 是融合后处理的一组实验，包括 MA 权重扫描与各类校正头",
     "H3、H4、H6、H8、H9、H11、H12/H13 是融合后处理的一组实验，包括 MA 权重扫描与各类校正头，E2、E3 是同结构的换底座与第三组初始化"),
    ("P2 同时改变了裁剪、缩放、参考分支输入和逆变换",
     "P2 同时改变了裁剪、缩放、参考分支输入、逆变换与损失（squared Dice 改为 Dice）"),
    ("D 与 B 的概率平均构成低成本训练对照。H5 则进一步融合 D、E、B、E4 四个已有模型",
     "D 与 B 的概率平均构成低成本训练对照。E4 是带细节分支的早期版本，H5 则进一步融合 D、E、B、E4 四个已有模型"),
    ("MA 细节分支、逐类路由、外部预训练与十余种纠错头做过尝试",
     "MA 细节分支、逐类路由、外部预训练与多组校正头做过尝试"),
]
for old, new in REP:
    done = False
    for p in d.paragraphs:
        for r in p.runs:
            if old in r.text:
                r.text = r.text.replace(old, new)
                done = True
                break
        if done:
            break
    if not done:
        print("MISS:", old[:30])

# 新增一段：把正文未提到的对照组一次交代
anchor = None
for p in d.paragraphs:
    if p.text.startswith("同学侧同期还比较过损失与训练策略"):
        anchor = p
        break
NEW = ("其余对照组一并说明：E3 训练未跑完；学生 BN 重估为 40.363%，全分支重估崩到 25.286%；"
       "C1/F1 的 16 组 MA 融合为 24.8429%；R1-A 与 R1-B 为 41.67% 与 41.75%，R5 五组在 41.78% 至 42.78% 之间，"
       "R6 的 HE 头为 43.60%，两个 RGB MA 专家的 patch IoU 为 2.20% 与 5.63%；"
       "普通采样与类平衡采样为 39.79% 与 39.83%，冠军加 C@2k 为 43.81%；"
       "U0 异构模型 38.45%、H0 36.92%，8 模型互补分析得到的最好逐类组合为 43.06%。"
       "DINOv3 分支的外部数据版未胜出，负类语义版为 44.16%，解冻 4 块与 12 块为 44.31% 与 44.59%，"
       "四臂中的 C、D、SD 为 45.18%、45.10%、44.88%。")
if anchor is not None:
    np = copy.deepcopy(anchor._p)
    runs = np.findall(qn("w:r"))
    rPr = None
    if runs and runs[0].find(qn("w:rPr")) is not None:
        rPr = copy.deepcopy(runs[0].find(qn("w:rPr")))
    for r in runs:
        np.remove(r)
    r = np.makeelement(qn("w:r"), {})
    if rPr is not None:
        r.append(copy.deepcopy(rPr))
    t = np.makeelement(qn("w:t"), {})
    t.text = NEW
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    np.append(r)
    anchor._p.addnext(np)

d.save(DOC)
d2 = docx.Document(DOC)
print("paras", len(d2.paragraphs), "tables", [len(t.rows) for t in d2.tables])
