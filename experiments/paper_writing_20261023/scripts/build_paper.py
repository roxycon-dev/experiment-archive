"""Build both complete language versions and their technical companions."""
from pathlib import Path
import shutil,subprocess
root=Path(__file__).resolve().parents[1]
for folder,compiler in [('paper','pdflatex'),('paper_zh','xelatex')]:
    latex=shutil.which(compiler);bibtex=shutil.which('bibtex')
    if not latex or not bibtex:raise SystemExit('Required TeX tools unavailable')
    subprocess.run([latex,'--version'],check=True)
    for target in ('main','supplement'):
        commands=[[latex,'--enable-installer','-interaction=nonstopmode','-halt-on-error','-file-line-error',target+'.tex']]
        if target=='main':commands.append([bibtex,target])
        commands.extend([commands[0]]*2)
        for args in commands:
            print('RUN',folder,args,flush=True)
            subprocess.run(args,cwd=root/folder,check=True,timeout=240)
print('Built English and Chinese main manuscripts and companions',flush=True)
