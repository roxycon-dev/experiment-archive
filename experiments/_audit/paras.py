# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.oxml.ns import qn

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
P_OLD = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

doc = Document(P_NEW)
body = doc.element.body
print("=== BODY ORDER ===")
pi = 0; ti = 0
for ch in body.iterchildren():
    tag = ch.tag.split('}')[1]
    if tag == 'p':
        st = ch.find(qn('w:pPr'))
        s = None
        if st is not None:
            ps = st.find(qn('w:pStyle'))
            s = ps.get(qn('w:val')) if ps is not None else None
        t = ''.join(n.text or '' for n in ch.iter(qn('w:t'))).strip()
        print(f"  P{pi:03d} [{s}] {t[:60]!r}")
        pi += 1
    elif tag == 'tbl':
        print(f"  >>> TABLE {ti} <<<")
        ti += 1
    elif tag == 'sectPr':
        print("  [sectPr]")

print()
print("=== PARAGRAPH DETAIL (pPr + first run) ===")
def show(p, i):
    pPr = p._p.find(qn('w:pPr'))
    info = {'style': p.style.name}
    if pPr is not None:
        for k in ('jc','spacing','ind','keepNext','keepLines','widowControl','numPr','outlineLvl'):
            e = pPr.find(qn('w:'+k))
            if e is not None:
                info[k] = {kk.split('}')[1]: vv for kk, vv in e.attrib.items()} or 'present'
        rPr = pPr.find(qn('w:rPr'))
        if rPr is not None:
            info['pPr.rPr'] = [ (x.tag.split('}')[1], dict((k.split('}')[1],v) for k,v in x.attrib.items())) for x in rPr]
    runs = p._p.findall(qn('w:r'))
    rinfo = []
    for r in runs[:3]:
        rPr = r.find(qn('w:rPr'))
        if rPr is None: rinfo.append(None); continue
        rinfo.append([(x.tag.split('}')[1], dict((k.split('}')[1],v) for k,v in x.attrib.items())) for x in rPr])
    print(f"P{i:03d} nRuns={len(runs)} {info}")
    for j, rr in enumerate(rinfo):
        print(f"     run{j}: {rr}")

for i, p in enumerate(doc.paragraphs):
    show(p, i)
