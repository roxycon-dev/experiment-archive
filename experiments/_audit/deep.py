# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.oxml.ns import qn

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
P_OLD = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

def tcs_of(tr): return tr.findall(qn('w:tc'))
def txt(el): return ''.join(n.text or '' for n in el.iter(qn('w:t')))

def run_sig(ps):
    """signature of all runs in a paragraph: list of (font,sz,b,color)"""
    out = []
    for r in ps.findall(qn('w:r')):
        rPr = r.find(qn('w:rPr'))
        s = {'font': None, 'sz': None, 'b': 'off', 'color': None}
        if rPr is not None:
            rf = rPr.find(qn('w:rFonts'))
            if rf is not None:
                s['font'] = (rf.get(qn('w:ascii')), rf.get(qn('w:eastAsia')), rf.get(qn('w:hAnsi')))
            sz = rPr.find(qn('w:sz'))
            if sz is not None: s['sz'] = sz.get(qn('w:val'))
            b = rPr.find(qn('w:b'))
            if b is not None: s['b'] = b.get(qn('w:val')) or 'on'
            c = rPr.find(qn('w:color'))
            if c is not None: s['color'] = c.get(qn('w:val'))
        out.append(s)
    return out

def p_sig(p):
    pPr = p.find(qn('w:pPr'))
    s = {'jc': None, 'spacing': None, 'ind': None, 'pStyle': None, 'rPr': None}
    if pPr is not None:
        jc = pPr.find(qn('w:jc'))
        if jc is not None: s['jc'] = jc.get(qn('w:val'))
        sp = pPr.find(qn('w:spacing'))
        if sp is not None:
            s['spacing'] = {k: sp.get(qn('w:'+k)) for k in ('before','after','line','lineRule','beforeLines','afterLines')}
        ind = pPr.find(qn('w:ind'))
        if ind is not None: s['ind'] = {k: v for k, v in ind.attrib.items()}
        st = pPr.find(qn('w:pStyle'))
        if st is not None: s['pStyle'] = st.get(qn('w:val'))
        rPr = pPr.find(qn('w:rPr'))
        if rPr is not None:
            rr = {'font': None, 'sz': None, 'b': None, 'color': None}
            rf = rPr.find(qn('w:rFonts'))
            if rf is not None: rr['font'] = (rf.get(qn('w:ascii')), rf.get(qn('w:eastAsia')), rf.get(qn('w:hAnsi')))
            sz = rPr.find(qn('w:sz'))
            if sz is not None: rr['sz'] = sz.get(qn('w:val'))
            b = rPr.find(qn('w:b'))
            if b is not None: rr['b'] = b.get(qn('w:val')) or 'on'
            c = rPr.find(qn('w:color'))
            if c is not None: rr['color'] = c.get(qn('w:val'))
            s['rPr'] = rr
    return s

def cell_sig(tc):
    tcPr = tc.find(qn('w:tcPr'))
    s = {'shd': None, 'vAlign': None, 'w': None, 'mar': None, 'bd': None, 'gridSpan': None}
    if tcPr is not None:
        shd = tcPr.find(qn('w:shd'))
        if shd is not None: s['shd'] = shd.get(qn('w:fill'))
        va = tcPr.find(qn('w:vAlign'))
        if va is not None: s['vAlign'] = va.get(qn('w:val'))
        w = tcPr.find(qn('w:tcW'))
        if w is not None: s['w'] = (w.get(qn('w:w')), w.get(qn('w:type')))
        mar = tcPr.find(qn('w:tcMar'))
        s['mar'] = {c.tag.split('}')[1]: c.get(qn('w:w')) for c in mar} if mar is not None else None
        bd = tcPr.find(qn('w:tcBorders'))
        s['bd'] = {c.tag.split('}')[1]: (c.get(qn('w:val')), c.get(qn('w:sz'))) for c in bd} if bd is not None else None
        gs = tcPr.find(qn('w:gridSpan'))
        if gs is not None: s['gridSpan'] = gs.get(qn('w:val'))
    ps = tc.findall(qn('w:p'))
    s['nP'] = len(ps)
    s['p'] = [p_sig(p) for p in ps]
    s['runs'] = [run_sig(p) for p in ps]
    return s

def row_sig(tr):
    trPr = tr.find(qn('w:trPr'))
    s = {'h': None, 'hRule': None, 'cantSplit': False, 'tblHeader': False, 'jc': None}
    if trPr is not None:
        h = trPr.find(qn('w:trHeight'))
        if h is not None: s['h'] = (h.get(qn('w:val')), h.get(qn('w:hRule')))
        s['cantSplit'] = trPr.find(qn('w:cantSplit')) is not None
        s['tblHeader'] = trPr.find(qn('w:tblHeader')) is not None
        jc = trPr.find(qn('w:jc'))
        if jc is not None: s['jc'] = jc.get(qn('w:val'))
    return s

old = Document(P_OLD); new = Document(P_NEW)
old_texts = set()
for tb in old.tables:
    for tr in tb._tbl.findall(qn('w:tr')):
        for tc in tcs_of(tr): old_texts.add(txt(tc).strip())

def compact(cs):
    """one-line fingerprint of a cell: font/sz/b/color algn vAlign margins borders width nP"""
    p0 = cs['p'][0] if cs['p'] else {}
    r0 = (cs['runs'][0][0] if cs['runs'] and cs['runs'][0] else {})
    rl = cs['runs'][0] if cs['runs'] else []
    bolds = sorted(set(x['b'] for x in rl))
    szs = sorted(set(str(x['sz']) for x in rl))
    colors = sorted(set(str(x['color']) for x in rl))
    fonts = sorted(set(str(x['font']) for x in rl))
    return (f"vA={cs['vAlign']} w={cs['w']} mar={cs['mar']} bd={cs['bd']} "
            f"jc={p0.get('jc')} sp={p0.get('spacing')} "
            f"PF={p0.get('rPr')} RF={fonts} sz={szs} b={bolds} col={colors} nP={cs['nP']} nR={[len(x) for x in cs['runs']]}")

for ti, tb in enumerate(new.tables):
    print("="*110)
    print(f"TABLE {ti}  rows={len(tb._tbl.findall(qn('w:tr')))}")
    for ri, tr in enumerate(tb._tbl.findall(qn('w:tr'))):
        cs_list = [cell_sig(tc) for tc in tcs_of(tr)]
        texts = [txt(tc).strip() for tc in tcs_of(tr)]
        isnew = not any(t in old_texts for t in texts if t)
        mark = "NEW" if isnew else "   "
        print(f"\nR{ri:02d} [{mark}] {texts[1][:30]!r}  row={row_sig(tr)}")
        for ci, cs in enumerate(cs_list):
            print(f"   c{ci}: {compact(cs)}")
