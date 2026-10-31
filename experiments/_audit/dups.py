# -*- coding: utf-8 -*-
import sys, io, re, collections, itertools
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.oxml.ns import qn
from lxml import etree

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
P_OLD = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

def txt(el): return ''.join(n.text or '' for n in el.iter(qn('w:t')))

# --- OLD body order + heading run props
old = Document(P_OLD)
print("=== OLD body order ===")
pi=0; ti=0
for ch in old.element.body.iterchildren():
    tag = ch.tag.split('}')[1]
    if tag=='p':
        pPr = ch.find(qn('w:pPr')); s=None
        if pPr is not None:
            ps = pPr.find(qn('w:pStyle')); s = ps.get(qn('w:val')) if ps is not None else None
        runs = ch.findall(qn('w:r'))
        rf = None
        if runs:
            rPr = runs[0].find(qn('w:rPr'))
            if rPr is not None:
                e = rPr.find(qn('w:rFonts'))
                if e is not None: rf = dict((k.split('}')[1],v) for k,v in e.attrib.items())
        print(f"  P{pi:03d} [{s}] rFonts={rf} {txt(ch).strip()[:40]!r}")
        pi+=1
    elif tag=='tbl':
        print(f"  >>> TABLE {ti} <<<"); ti+=1

new = Document(P_NEW)
rows=[]
for ti, tb in enumerate(new.tables):
    for ri, tr in enumerate(tb._tbl.findall(qn('w:tr'))):
        if ri==0: continue
        c=[txt(tc).strip() for tc in tr.findall(qn('w:tc'))]
        rows.append((ti+1, ri, c))

print("\n=== 实验组名重复 ===")
names = collections.defaultdict(list)
for t,r,c in rows:
    names[c[1]].append((t,r))
for n,v in names.items():
    if len(v)>1: print(f"  组名 {n!r} 出现 {len(v)} 次: {v}")

print("\n=== 完全相同行（四列全同） ===")
sigs = collections.defaultdict(list)
for t,r,c in rows:
    sigs[tuple(c)].append((t,r))
for s,v in sigs.items():
    if len(v)>1: print(f"  {v} : {s}")

def nums(s):
    return re.findall(r'\d+\.\d+|\d+', s)

print("\n=== 结果列数值集合高度重合 ===")
res = [(t,r,c[1],c[3]) for t,r,c in rows]
for (t1,r1,n1,d1),(t2,r2,n2,d2) in itertools.combinations(res,2):
    if n1==n2 and d1==d2:
        print(f"  [全等] 表{t1}R{r1} {n1} == 表{t2}R{r2} {n2}"); continue
    a=set(nums(d1)); b=set(nums(d2))
    if a and b:
        inter=a&b
        j=len(inter)/len(a|b)
        if j>=0.75:
            print(f"  [雷同 {j:.0%}] 表{t1}R{r1} {n1!r} <-> 表{t2}R{r2} {n2!r}")
            print(f"        {d1[:110]}")
            print(f"        {d2[:110]}")

print("\n=== 组名包含关系（同一实验被拆两行）===")
for (t1,r1,n1,_),(t2,r2,n2,_) in itertools.combinations(res,2):
    if n1==n2: continue
    if n1 in n2 or n2 in n1:
        print(f"  表{t1}R{r1} {n1!r}  ⊂/⊃  表{t2}R{r2} {n2!r}")
