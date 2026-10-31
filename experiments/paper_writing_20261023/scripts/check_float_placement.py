from pathlib import Path
import re,json,hashlib
import pymupdf
root=Path(__file__).resolve().parents[1]
records=[]
for folder,heading in [('paper','REFERENCES'),('paper_zh','参考文献')]:
    pdf=root/folder/'main.pdf';doc=pymupdf.open(pdf)
    refs=[i+1 for i,p in enumerate(doc) if heading in [x.strip() for x in p.get_text().splitlines()]]
    assert refs,folder
    refpage=min(refs)
    aux=(root/folder/'main.aux').read_text(encoding='utf-8')
    labels={key:int(page) for key,page in re.findall(r'\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}',aux)}
    floats={k:v for k,v in labels.items() if k.startswith(('fig:','tab:'))}
    assert len(floats)==8,(folder,floats)
    assert all(v<refpage for v in floats.values()),(folder,refpage,floats)
    assert labels['fig:qualitative']<labels['sec:discussion'],(folder,labels)
    records.append({'document':folder,'pages':len(doc),'float_pages':floats,'discussion_page':labels['sec:discussion'],'references_start_page':refpage,'all_floats_before_references':True,'figure3_before_discussion':True,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()})
(root/'review/float_placement_check.json').write_text(json.dumps({'status':'PASS','documents':records},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(records,ensure_ascii=False))
