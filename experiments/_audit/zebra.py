# -*- coding: utf-8 -*-
import sys, io, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.oxml.ns import qn

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
P_OLD = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

def txt(el): return ''.join(n.text or '' for n in el.iter(qn('w:t')))

def rowinfo(tr):
    trPr = tr.find(qn('w:trPr'))
    h = None; cant = False; hdr = False
    if trPr is not None:
        he = trPr.find(qn('w:trHeight'))
        if he is not None: h = (he.get(qn('w:val')), he.get(qn('w:hRule')))
        cant = trPr.find(qn('w:cantSplit')) is not None
        hdr = trPr.find(qn('w:tblHeader')) is not None
    cells = tr.findall(qn('w:tc'))
    fills = []
    fonts = set(); szs = set(); bolds = set(); cols = set(); jcs = set(); vas = set(); mars = set(); bds = set()
    for tc in cells:
        tcPr = tc.find(qn('w:tcPr'))
        if tcPr is not None:
            shd = tcPr.find(qn('w:shd'))
            fills.append(shd.get(qn('w:fill')) if shd is not None else 'NONE')
            va = tcPr.find(qn('w:vAlign'))
            vas.add(va.get(qn('w:val')) if va is not None else None)
            mar = tcPr.find(qn('w:tcMar'))
            mars.add(tuple(sorted((c.tag.split('}')[1], c.get(qn('w:w'))) for c in mar)) if mar is not None else None)
            bd = tcPr.find(qn('w:tcBorders'))
            bds.add(tuple(sorted((c.tag.split('}')[1], c.get(qn('w:val')), c.get(qn('w:sz')), c.get(qn('w:color'))) for c in bd)) if bd is not None else None)
        else:
            fills.append('NONE')
        for p in tc.findall(qn('w:p')):
            pPr = p.find(qn('w:pPr'))
            jc = pPr.find(qn('w:jc')) if pPr is not None else None
            jcs.add(jc.get(qn('w:val')) if jc is not None else None)
            for r in p.findall(qn('w:r')):
                rPr = r.find(qn('w:rPr'))
                if rPr is None: continue
                rf = rPr.find(qn('w:rFonts'))
                if rf is not None: fonts.add((rf.get(qn('w:ascii')), rf.get(qn('w:eastAsia'))))
                s = rPr.find(qn('w:sz'))
                if s is not None: szs.add(s.get(qn('w:val')))
                b = rPr.find(qn('w:b'))
                bolds.add((b.get(qn('w:val')) or 'on') if b is not None else 'off')
                c = rPr.find(qn('w:color'))
                if c is not None: cols.add(c.get(qn('w:val')))
    return dict(h=h, cant=cant, hdr=hdr, fills=fills, fonts=sorted(map(str,fonts)), sz=sorted(szs),
                b=sorted(bolds), col=sorted(cols), jc=sorted(map(str,jcs)), va=sorted(map(str,vas)),
                mar=len(mars), bd=len(bds), texts=[txt(tc).strip() for tc in cells])

old = Document(P_OLD); new = Document(P_NEW)
old_texts = set()
for tb in old.tables:
    for tr in tb._tbl.findall(qn('w:tr')):
        for tc in tr.findall(qn('w:tc')): old_texts.add(txt(tc).strip())

for ti, tb in enumerate(new.tables):
    trs = tb._tbl.findall(qn('w:tr'))
    print("="*96)
    print(f"TABLE {ti} rows={len(trs)}")
    exp_start = 'NONE' if ti == 0 else 'F4F7FA'
    bad = []
    for ri, tr in enumerate(trs):
        I = rowinfo(tr)
        isnew = not any(t in old_texts for t in I['texts'] if t)
        uniq = set(I['fills'])
        shd = uniq.pop() if len(uniq) == 1 else 'MIXED:' + ','.join(sorted(map(str, set(I['fills']))))
        if ri == 0:
            exp = 'HDR'
        else:
            exp = 'NONE' if (ri % 2 == 1) == (ti == 0) else 'F4F7FA'
        ok = (ri == 0 and shd == '233A54') or (ri > 0 and shd == exp)
        flag = '' if ok else f'  <== 斑马纹不符 期望{exp}'
        if not ok: bad.append(ri)
        print(f"R{ri:02d} {'NEW' if isnew else '   '} shd={shd:<6} h={I['h']} cant={int(I['cant'])} hdr={int(I['hdr'])} "
              f"f={I['fonts']} sz={I['sz']} b={I['b']} c={I['col']} jc={I['jc']} va={I['va']} "
              f"marTypes={I['mar']} bdTypes={I['bd']} | {I['texts'][1][:22]}{flag}")
    print("斑马纹异常行:", bad if bad else "无")
