# -*- coding: utf-8 -*-
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.oxml.ns import qn

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
P_OLD = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def q(t): return qn('w:'+t)

def get_or_none(el, path):
    if el is None: return None
    return el.find(qn('w:'+path))

def run_fmt(r):
    rPr = r.find(qn('w:rPr'))
    d = {}
    if rPr is None: return d
    rf = rPr.find(qn('w:rFonts'))
    if rf is not None:
        d['ascii'] = rf.get(qn('w:ascii'))
        d['ea'] = rf.get(qn('w:eastAsia'))
        d['hAnsi'] = rf.get(qn('w:hAnsi'))
    sz = rPr.find(qn('w:sz'))
    if sz is not None: d['sz'] = sz.get(qn('w:val'))
    b = rPr.find(qn('w:b'))
    if b is not None: d['b'] = b.get(qn('w:val')) or 'on'
    else: d['b'] = 'off'
    c = rPr.find(qn('w:color'))
    if c is not None: d['color'] = c.get(qn('w:val'))
    return d

def para_fmt(p_el):
    pPr = p_el.find(qn('w:pPr'))
    d = {}
    if pPr is None: return d
    jc = pPr.find(qn('w:jc'))
    d['jc'] = jc.get(qn('w:val')) if jc is not None else None
    sp = pPr.find(qn('w:spacing'))
    if sp is not None:
        for k in ('before','after','line','lineRule'):
            d['sp_'+k] = sp.get(qn('w:'+k))
    ind = pPr.find(qn('w:ind'))
    if ind is not None:
        d['ind_left'] = ind.get(qn('w:left'))
        d['ind_first'] = ind.get(qn('w:firstLine'))
    st = pPr.find(qn('w:pStyle'))
    d['pStyle'] = st.get(qn('w:val')) if st is not None else None
    return d

def cell_fmt(tc):
    tcPr = tc.find(qn('w:tcPr'))
    d = {}
    if tcPr is None: return d
    shd = tcPr.find(qn('w:shd'))
    d['shd'] = shd.get(qn('w:fill')) if shd is not None else None
    va = tcPr.find(qn('w:vAlign'))
    d['vAlign'] = va.get(qn('w:val')) if va is not None else None
    tcW = tcPr.find(qn('w:tcW'))
    if tcW is not None:
        d['w'] = tcW.get(qn('w:w')); d['wtype'] = tcW.get(qn('w:type'))
    mar = tcPr.find(qn('w:tcMar'))
    if mar is not None:
        d['mar'] = {c.tag.split('}')[1]: c.get(qn('w:w')) for c in mar}
    else:
        d['mar'] = None
    bd = tcPr.find(qn('w:tcBorders'))
    if bd is not None:
        d['bd'] = {c.tag.split('}')[1]: (c.get(qn('w:val')), c.get(qn('w:sz')), c.get(qn('w:color'))) for c in bd}
    else:
        d['bd'] = None
    gs = tcPr.find(qn('w:gridSpan'))
    if gs is not None: d['gridSpan'] = gs.get(qn('w:val'))
    # paragraph props inside cell (first p)
    ps = tc.findall(qn('w:p'))
    d['nparas'] = len(ps)
    if ps:
        d['pfmt'] = para_fmt(ps[0])
        runs = ps[0].findall(qn('w:r'))
        d['nruns'] = len(runs)
        d['runs'] = [run_fmt(r) for r in runs]
    return d

def row_fmt(tr):
    trPr = tr.find(qn('w:trPr'))
    d = {'trHeight': None, 'hRule': None, 'cantSplit': False, 'tblHeader': False}
    if trPr is None: return d
    h = trPr.find(qn('w:trHeight'))
    if h is not None:
        d['trHeight'] = h.get(qn('w:val')); d['hRule'] = h.get(qn('w:hRule'))
    d['cantSplit'] = trPr.find(qn('w:cantSplit')) is not None
    d['tblHeader'] = trPr.find(qn('w:tblHeader')) is not None
    return d

def table_meta(tbl):
    t = tbl._tbl
    tblPr = t.find(qn('w:tblPr'))
    meta = {}
    if tblPr is not None:
        st = tblPr.find(qn('w:tblStyle'))
        meta['style'] = st.get(qn('w:val')) if st is not None else None
        wdt = tblPr.find(qn('w:tblW'))
        if wdt is not None: meta['tblW'] = (wdt.get(qn('w:w')), wdt.get(qn('w:type')))
        lay = tblPr.find(qn('w:tblLayout'))
        if lay is not None: meta['layout'] = lay.get(qn('w:type'))
        mar = tblPr.find(qn('w:tblCellMar'))
        if mar is not None:
            meta['cellMar'] = {c.tag.split('}')[1]: c.get(qn('w:w')) for c in mar}
        bd = tblPr.find(qn('w:tblBorders'))
        if bd is not None:
            meta['borders'] = {c.tag.split('}')[1]: (c.get(qn('w:val')), c.get(qn('w:sz')), c.get(qn('w:color'))) for c in bd}
        look = tblPr.find(qn('w:tblLook'))
        if look is not None: meta['look'] = dict(look.attrib)
    grid = t.find(qn('w:tblGrid'))
    meta['grid'] = [g.get(qn('w:w')) for g in grid.findall(qn('w:gridCol'))] if grid is not None else None
    meta['nrows'] = len(t.findall(qn('w:tr')))
    return meta

def row_signature(tbl, ri):
    tr = tbl._tbl.findall(qn('w:tr'))[ri]
    tc_list = tr.findall(qn('w:tc'))
    return {
        'row': row_fmt(tr),
        'cells': [cell_fmt(tc) for tc in tc_list],
        'texts': [''.join(n.text or '' for n in tc.iter(qn('w:t'))) for tc in tc_list],
    }

for path, tag in [(P_OLD, 'OLD'), (P_NEW, 'NEW')]:
    d = Document(path)
    print("#"*80)
    print(tag)
    for ti, tb in enumerate(d.tables):
        meta = table_meta(tb)
        print(f"--- TABLE {ti} grid={meta.get('grid')} style={meta.get('style')} tblW={meta.get('tblW')} layout={meta.get('layout')}")
        print(f"    tblCellMar={meta.get('cellMar')}")
        print(f"    tblBorders={meta.get('borders')}")
        print(f"    tblLook={meta.get('look')} nrows={meta.get('nrows')}")
        for ri in range(len(tb._tbl.findall(qn('w:tr')))):
            sig = row_signature(tb, ri)
            print(f"  R{ri:02d} tr={sig['row']} shd={[c.get('shd') for c in sig['cells']]} va={[c.get('vAlign') for c in sig['cells']]}")
