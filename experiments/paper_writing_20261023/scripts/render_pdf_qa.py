"""Render all four actual EN/CN PDFs and retain final source/page hashes for QA."""
from pathlib import Path
import argparse, json, hashlib, shutil, subprocess
from datetime import datetime, timezone
import pymupdf
from PIL import Image, ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument('--revision',default='R5')
args=parser.parse_args()
assert args.revision.isalnum(), 'Use an alphanumeric revision name.'
root=Path(__file__).resolve().parents[1]
out=root/'review'/('pdf_render_'+args.revision.lower());out.mkdir(parents=True,exist_ok=True)
poppler=shutil.which('pdftoppm')
assert poppler, 'Use the available bundled Poppler executable for final QA.'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
records=[]
for folder,name,label in (('paper','main','main_en'),('paper_zh','main','main_zh'),
                           ('paper','supplement','supplement_en'),('paper_zh','supplement','supplement_zh')):
    src=root/folder/f'{name}.pdf'; before=sha(src);doc=pymupdf.open(src)
    prefix=out/label
    subprocess.run([poppler,'-r','108','-png',str(src),str(prefix)],check=True,capture_output=True)
    thumbs=[];pages=[]
    for i,page in enumerate(doc):
        generated=out/f'{label}-{i+1:0{len(str(len(doc)))}d}.png'
        dst=out/f'{label}_{i+1:02d}.png'
        generated.replace(dst)
        full=Image.open(dst).convert('RGB');im=full.copy();im.thumbnail((370,520))
        cell=Image.new('RGB',(390,550),'#dddddd');cell.paste(im,((390-im.width)//2,25))
        ImageDraw.Draw(cell).text((10,5),f'{label} page {i+1}',fill='black');thumbs.append(cell)
        pages.append({'page':i+1,'png_path':str(dst),'png_sha256':sha(dst),
                      'rgb_pixel_sha256':hashlib.sha256(full.tobytes()).hexdigest(),
                      'pixels':list(full.size),'page_rect':list(page.rect),
                      'extracted_characters':len(page.get_text())})
    cols=3;rows=(len(thumbs)+cols-1)//cols
    contact=Image.new('RGB',(cols*390,rows*550),'#bbbbbb')
    for i,im in enumerate(thumbs):contact.paste(im,((i%cols)*390,(i//cols)*550))
    contact_path=out/f'{label}_contact.png';contact.save(contact_path)
    assert before==sha(src),f'PDF changed during rendering: {src}'
    records.append({'label':label,'path':str(src),'sha256':before,'pages':len(doc),
      'page_render_scale':1.5,'page_render_dpi':108,'page_records':pages,
      'contact_sheet':str(contact_path),'contact_sha256':sha(contact_path),
      'source_sha256':{str(p):sha(p) for p in sorted((root/folder).rglob('*.tex'))}})
manifest={'revision':args.revision.upper(),'rendered_at_utc':datetime.now(timezone.utc).isoformat(),
          'renderer':poppler,'script_sha256':sha(__file__),'documents':records,
          'total_pages':sum(r['pages'] for r in records),'visual_review':'review/pdf_visual_review.md'}
(out/'render_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'total_pages':manifest['total_pages'],'documents':[{k:r[k] for k in ('label','path','sha256','pages')} for r in records]},indent=2))
