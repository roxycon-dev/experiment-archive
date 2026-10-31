"""Export both language versions as a root-entry portable TeX project."""
from pathlib import Path
import hashlib,json,re,zipfile,shutil
root=Path(__file__).resolve().parents[1]
members={};graphics=set()
for folder,lang in [('paper','en'),('paper_zh','zh')]:
    for p in sorted((root/folder).rglob('*')):
        if not p.is_file() or p.suffix not in {'.tex','.bib','.cls','.bst'}:continue
        name=p.relative_to(root/folder).as_posix()
        if lang=='zh':
            if p.suffix!='.tex':continue
            name=name.replace('sections/','sections_zh/').replace('tables_r4/','tables_r4_zh/')
            if '/' not in name:name=p.stem+'_zh.tex'
        if p.suffix=='.tex':
            s=p.read_text(encoding='utf-8')
            for g in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',s):graphics.add((root/folder/g).resolve())
            s=s.replace('../figures/','figures/')
            if lang=='zh':s=s.replace('sections/','sections_zh/').replace('tables_r4/','tables_r4_zh/').replace(r'\input{training_table}',r'\input{training_table_zh}')
            members[name]=s.encode('utf-8')
        else:members[name]=p.read_bytes()
    for name in ('main','supplement'):
        members[name+('_zh' if lang=='zh' else '')+'.pdf']=(root/folder/(name+'.pdf')).read_bytes()
for p in sorted(graphics):
    assert p.is_file(),p
    members['figures/'+p.name]=p.read_bytes()
    svg=p.with_suffix('.svg')
    if svg.is_file():members['figures/'+svg.name]=svg.read_bytes()
for p in sorted((root/'evidence').glob('revision4_*.csv')):
    if p.name=='revision4_file_inventory.csv':continue
    members['table_data/'+p.name]=p.read_bytes()
for src,name in [('figures/figure_sources.json','provenance/earlier_figure_sources.json'),('figures/revision4_figure_manifest.json','provenance/current_evidence_figures.json'),('literature/venue_sources/template_provenance.json','provenance/template.json'),('literature/module_original_sources.md','provenance/module_original_sources.md')]:members[name]=(root/src).read_bytes()
members['revision_notes/defensive_writing_review.md']=(root/'review/revision5_defensive_audit.md').read_bytes()
members['revision_notes/paper_story.md']=(root/'paper_story.md').read_bytes()
members['revision_notes/author_revision_R6.md']=(root/'review/revision6_resolution.md').read_bytes()
members['revision_notes/review_revision_R7.md']=(root/'review/revision7_resolution.md').read_bytes()
members['README.txt']=("""HRDecoder augmentation — English and Chinese editable manuscript package

English paper: main.tex / main.pdf. Compiler: pdfLaTeX.
Chinese paper: main_zh.tex / main_zh.pdf. Compiler: XeLaTeX.
Technical companions: supplement.tex and supplement_zh.tex; use the corresponding compiler.
Bibliography: references.bib, compiled with BibTeX.

Overleaf: upload the ZIP, select main.tex with pdfLaTeX OR main_zh.tex with XeLaTeX.
English locally: pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex.
For example: xelatex main_zh.tex; bibtex main_zh; xelatex main_zh.tex; xelatex main_zh.tex.

Edit sections/ for English and sections_zh/ for Chinese. Numerical table fragments are in
tables_r4/ and tables_r4_zh/; their audited source values are included under table_data/.
All graphics needed for compilation are included. Used evidence figures also include SVG
sources where available; generated conceptual diagrams are raster illustrations.

The supplementary experiments have been incorporated. Author/affiliation, funding,
conflict and ethics declarations remain explicitly pending according to the authors.
The technical companions support author review; they are not designated as admissible
ordinary TMI text/figure supplements. The current main manuscript contains core methods,
training and evaluation protocols. No journal submission has been made.

IEEEtran files retain their original license. No model weights, full datasets, private
credentials or third-party research-paper PDFs are included. Compilation does not require
the original research workspace. Evidence regeneration may require the archived experiment
assets; the ready-made manuscript tables and figures do not.
""").encode('utf-8')
target=root/'HRDecoder_bilingual_source.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in sorted(members.items()):z.writestr(name,data)
shutil.copy2(target,root/'HRDecoder_manuscript_source.zip')
record={'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'members':sorted(members),'member_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in members.items()},'scope':'Portable bilingual papers and companions, used graphics, SVG evidence assets, shared bibliography and source table data; root main.tex and main_zh.tex.'}
(root/'source_package_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'{target.name}: {len(members)} files, {target.stat().st_size:,} bytes')
