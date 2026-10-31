# -*- coding: utf-8 -*-
import sys, io, json, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.oxml.ns import qn

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
P_OLD = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

def txt(el): return ''.join(n.text or '' for n in el.iter(qn('w:t')))

def run_tuple(r):
    rPr = r.find(qn('w:rPr'))
    d = {'f': None, 'sz': None, 'b': None, 'c': None, 'other': ()}
    if rPr is None: return d
    rf = rPr.find(qn('w:rFonts'))
    if rf is not None:
        d['f'] = (rf.get(qn('w:ascii')), rf.get(qn('w:eastAsia')), rf.get(qn('w:hAnsi')), rf.get(qn('w:cs')))
    sz = rPr.find(qn('w:sz'))
    if sz is not None: d['sz'] = sz.get(qn('w:val'))
    b = rPr.find(qn('w:b'))
    d['b'] = (b.get(qn('w:val')) or 'on') if b is not None else 'off'
    c = rPr.find(qn('w:color'))
    if c is not None: d['c'] = c.get(qn('w:val'))
    # extra rPr children names to detect stray formatting
    d['other'] = tuple(sorted(x.tag.split('}')[1] for x in rPr))
    return d

def para_tuple(p):
    pPr = p.find(qn('w:pPr'))
    d = {'jc': None, 'sp': None, 'ind': None, 'style': None, 'pPrKids': ()}
    if pPr is not None:
        d['pPrKids'] = tuple(sorted(x.tag.split('}')[1] for x in pPr))
        jc = pPr.find(qn('w:jc'))
        if jc is not None: d['jc'] = jc.get(qn('w:val'))
        sp = pPr.find(qn('w:spacing'))
        if sp is not None:
            d['sp'] = tuple((k, sp.get(qn('w:'+k))) for k in ('before','after','line','lineRule','beforeLines','afterLines'))
        ind = pPr.find(qn('w:ind'))
        if ind is not None: d['ind'] = tuple(sorted(ind.attrib.items()))
        st = pPr.find(qn('w:pStyle'))
        if st is not None: d['style'] = st.get(qn('w:val'))
    runs = [run_tuple(r) for r in p.findall(qn('w:r'))]
    return d, runs

def cell_tuple(tc):
    tcPr = tc.find(qn('w:tcPr'))
    d = {'shd': None, 'va': None, 'w': None, 'mar': None, 'bd': None, 'gs': None, 'kids': ()}
    if tcPr is not None:
        d['kids'] = tuple(sorted(x.tag.split('}')[1] for x in tcPr))
        shd = tcPr.find(qn('w:shd'))
        if shd is not None: d['shd'] = (shd.get(qn('w:val')), shd.get(qn('w:fill')))
        va = tcPr.find(qn('w:vAlign'))
        if va is not None: d['va'] = va.get(qn('w:val'))
        w = tcPr.find(qn('w:tcW'))
        if w is not None: d['w'] = (w.get(qn('w:w')), w.get(qn('w:type')))
        mar = tcPr.find(qn('w:tcMar'))
        d['mar'] = tuple(sorted((c.tag.split('}')[1], c.get(qn('w:w'))) for c in mar)) if mar is not None else None
        bd = tcPr.find(qn('w:tcBorders'))
        d['bd'] = tuple(sorted((c.tag.split('}')[1], c.get(qn('w:val')), c.get(qn('w:sz')), c.get(qn('w:space')), c.get(qn('w:color'))) for c in bd)) if bd is not None else None
        gs = tcPr.find(qn('w:gridSpan'))
        if gs is not None: d['gs'] = gs.get(qn('w:val'))
    ps = tc.findall(qn('w:p'))
    d['paras'] = [para_tuple(p) for p in ps]
    d['hasTbl'] = len(tc.findall(qn('w:tbl'))) > 0
    return d

def row_tuple(tr):
    trPr = tr.find(qn('w:trPr'))
    d = {'h': None, 'cant': False, 'hdr': False, 'kids': ()}
    if trPr is not None:
        d['kids'] = tuple(sorted(x.tag.split('}')[1] for x in trPr))
        h = trPr.find(qn('w:trHeight'))
        if h is not None: d['h'] = (h.get(qn('w:val')), h.get(qn('w:hRule')))
        d['cant'] = trPr.find(qn('w:cantSplit')) is not None
        d['hdr'] = trPr.find(qn('w:tblHeader')) is not None
    tc_list = tr.findall(qn('w:tc'))
    d['cells'] = [cell_tuple(tc) for tc in tc_list]
    d['ncell'] = len(tc_list)
    d['texts'] = [txt(tc).strip() for tc in tc_list]
    return d

old = Document(P_OLD); new = Document(P_NEW)
old_texts = set()
old_rows_by_text = {}
for tb in old.tables:
    for tr in tb._tbl.findall(qn('w:tr')):
        for tc in tr.findall(qn('w:tc')):
            t = txt(tc).strip()
            old_texts.add(t)

def body_key(rt):
    """format fingerprint independent of text and zebra shading"""
    out = [('row', rt['h'], rt['cant'], rt['hdr'], rt['ncell'], rt['kids'])]
    for c in rt['cells']:
        out.append(('cell', c['va'], c['w'], c['mar'], c['bd'], c['gs'], c['kids'], c['hasTbl'], len(c['paras'])))
        for pd, runs in c['paras']:
            out.append(('p', pd['jc'], pd['sp'], pd['ind'], pd['style'], pd['pPrKids'], len(runs)))
            for r in runs:
                out.append(('r', r['f'], r['sz'], r['b'], r['c'], r['other']))
    return tuple(map(str, out))

for ti, tb in enumerate(new.tables):
    trs = tb._tbl.findall(qn('w:tr'))
    print("="*100)
    print(f"TABLE {ti} rows={len(trs)}")
    keys = collections.Counter()
    rows = []
    for ri, tr in enumerate(trs):
        rt = row_tuple(tr)
        isnew = not any(t in old_texts for t in rt['texts'] if t)
        k = body_key(rt)
        rows.append((ri, isnew, k, rt))
        if ri > 0: keys[k] += 1
    print("distinct body fingerprints (non-header rows):", len(keys))
    # dominant key
    dom, domc = keys.most_common(1)[0]
    print("dominant count:", domc)
    for ri, isnew, k, rt in rows:
        if ri == 0: continue
        if k != dom:
            print(f"\n>>> R{ri:02d} {'NEW' if isnew else 'OLD'} group={rt['texts'][1][:34]!r} DIFFERS from dominant")
            # find first differing element
            for a, b in zip(k, dom):
                if a != b:
                    print("    got:", a)
                    print("    exp:", b)
    # zebra
    print("\n-- zebra shades --")
    for ri, isnew, k, rt in rows:
        s = [c['shd'] for c in rt['cells']]
        fills = set(x[1] if x else 'NONE' for x in s)
        print(f"R{ri:02d} {'NEW' if isnew else 'OLD '} shades={sorted(map(str,fills))} uniform={len(fills)==1}")
