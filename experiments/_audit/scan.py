# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.shared import Pt

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
P_OLD = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

def dump(p, tag):
    d = Document(p)
    print("="*70)
    print(tag, "paragraphs=", len(d.paragraphs), "tables=", len(d.tables))
    print("-"*70)
    for i, para in enumerate(d.paragraphs):
        t = para.text.strip()
        st = para.style.name
        print(f"[P{i:03d}] style={st!r} len={len(t)} :: {t[:80]}")
    print("-"*70)
    for ti, tb in enumerate(d.tables):
        print(f"[TABLE {ti}] rows={len(tb.rows)} cols={len(tb.columns)}")
        for ri, row in enumerate(tb.rows):
            cells = [c.text.strip().replace("\n","\\n")[:24] for c in row.cells]
            print(f"  R{ri:02d} h={row.height} | " + " | ".join(cells))

for p, tag in [(P_OLD, "ORIGINAL"), (P_NEW, "NEW")]:
    dump(p, tag)
