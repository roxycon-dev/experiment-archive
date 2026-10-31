"""Compile four exported documents in isolation; compare all rendered pages."""
from pathlib import Path
import zipfile,subprocess,shutil,json,hashlib
import pymupdf
r=Path(__file__).resolve().parents[1];target=r/'review/package_compile_r7';target.mkdir(exist_ok=True)
with zipfile.ZipFile(r/'HRDecoder_bilingual_source.zip') as z:z.extractall(target)
logs=[];comparisons={}
for stem,folder,compiler,original in [('main','paper','pdflatex','main'),('supplement','paper','pdflatex','supplement'),('main_zh','paper_zh','xelatex','main'),('supplement_zh','paper_zh','xelatex','supplement')]:
    command=[shutil.which(compiler),'-interaction=nonstopmode','-halt-on-error',stem+'.tex']
    commands=[command]
    if original=='main':commands.append([shutil.which('bibtex'),stem])
    commands.extend([command]*2)
    for cmd in commands:
        p=subprocess.run(cmd,cwd=target,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        logs.append('COMMAND '+repr(cmd)+'\nEXIT '+str(p.returncode)+'\n'+p.stdout+'\n'+p.stderr)
        if p.returncode:raise RuntimeError(logs[-1])
    a=pymupdf.open(r/folder/(original+'.pdf'));b=pymupdf.open(target/(stem+'.pdf'))
    assert len(a)==len(b),(stem,len(a),len(b))
    equal=[x.get_pixmap().samples==y.get_pixmap().samples for x,y in zip(a,b)]
    assert all(equal),('Page rendering differs',stem,equal)
    comparisons[stem]={'pages':len(a),'all_pages_pixel_identical':all(equal)}
(r/'review/package_compile_r7.log').write_text('\n'.join(logs),encoding='utf-8')
record={'status':'PASS','archive_sha256':hashlib.sha256((r/'HRDecoder_bilingual_source.zip').read_bytes()).hexdigest(),'render_comparisons':comparisons,'external_workspace_paths_required':False,'log':'review/package_compile_r7.log'}
(r/'review/package_compile_r7.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record,indent=2))
