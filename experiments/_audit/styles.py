# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from docx import Document
from docx.oxml.ns import qn
from lxml import etree

P_NEW = r"D:\wechat\xwechat_files\wxid_7rzu6lw5zzod22_9c3c\msg\file\2026-10\26大创人工智能眼底-后期检查报告-20261021.docx"
P_OLD = r"D:\dataset\experiment\_maintenance\26大创报告_原始备份_20261021.docx"

def xml(e):
    return etree.tostring(e, encoding='unicode')

for path, tag in [(P_OLD,'OLD'), (P_NEW,'NEW')]:
    doc = Document(path)
    print("="*70); print(tag)
    st = doc.styles.element
    dd = st.find(qn('w:docDefaults'))
    if dd is not None:
        print("docDefaults:", xml(dd).replace(' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"','')[:1200])
    for s in st.findall(qn('w:style')):
        sid = s.get(qn('w:styleId'))
        nm = s.find(qn('w:name'))
        nmv = nm.get(qn('w:val')) if nm is not None else None
        if nmv in ('Normal','heading 1','heading 2','Title'):
            print(f"-- id={sid} name={nmv}")
            body = xml(s).replace(' xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"','')
            print("   ", body[:800])
