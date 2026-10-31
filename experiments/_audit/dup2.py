# -*- coding: utf-8 -*-
import sys, io, re, itertools, difflib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.oxml.ns import qn

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
def txt(e): return ''.join(n.text or '' for n in e.iter(qn('w:t')))
d = Document(P_NEW)
rows=[]
for ti,tb in enumerate(d.tables):
    for ri,tr in enumerate(tb._tbl.findall(qn('w:tr'))):
        if ri==0: continue
        c=[txt(tc).strip() for tc in tr.findall(qn('w:tc'))]
        rows.append(dict(t=ti+1, r=ri, name=c[1], cond=c[2], res=c[3]))

def pct(s):
    return [round(float(x), 4) for x in re.findall(r'\d+\.\d+', s)]

print("=== 结果数值集合相同(归一化) ===")
for a,b in itertools.combinations(rows,2):
    pa, pb = set(pct(a['res'])), set(pct(b['res']))
    if pa and pb and pa == pb:
        print(f"  表{a['t']}R{a['r']} {a['name']!r}  ==  表{b['t']}R{b['r']} {b['name']!r}")
        print(f"      {a['res']}")
        print(f"      {b['res']}")
    else:
        inter = pa & pb
        if len(inter) >= 2 and len(inter)/max(len(pa),len(pb)) >= 0.6:
            print(f"  [重叠{len(inter)}/{max(len(pa),len(pb))}] 表{a['t']}R{a['r']} {a['name']!r} <-> 表{b['t']}R{b['r']} {b['name']!r}  {sorted(inter)}")

print("\n=== 主要条件文本高度相似(>=0.72) ===")
for a,b in itertools.combinations(rows,2):
    s = difflib.SequenceMatcher(None, a['cond'], b['cond']).ratio()
    if s >= 0.72:
        print(f"  {s:.2f}  表{a['t']}R{a['r']} {a['name']!r} :: {a['cond']}")
        print(f"        表{b['t']}R{b['r']} {b['name']!r} :: {b['cond']}")

print("\n=== 组名相似(>=0.7) ===")
for a,b in itertools.combinations(rows,2):
    s = difflib.SequenceMatcher(None, a['name'], b['name']).ratio()
    if s >= 0.7:
        print(f"  {s:.2f} 表{a['t']}R{a['r']} {a['name']!r} <-> 表{b['t']}R{b['r']} {b['name']!r}")
