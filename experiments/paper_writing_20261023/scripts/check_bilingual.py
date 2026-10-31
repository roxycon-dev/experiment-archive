"""Check deterministic numerical parity and actual bilingual source identities."""
from pathlib import Path
import re,json,hashlib
r=Path(__file__).resolve().parents[1]
checks=[]
number=re.compile(r'(?<![A-Za-z])[-+]?\d+(?:,\d{3})*(?:\.\d+)?')
for p in sorted((r/'paper/tables_r4').glob('*.tex')):
    q=r/'paper_zh/tables_r4'/p.name
    a=number.findall(p.read_text(encoding='utf-8'));b=number.findall(q.read_text(encoding='utf-8'))
    checks.append({'item':'table/'+p.name,'pass':a==b,'numbers':len(a)})
for name in ('framing','method','experiments','results','discussion'):
    a=(r/'paper/sections'/f'{name}.tex').read_text(encoding='utf-8')
    b=(r/'paper_zh/sections'/f'{name}.tex').read_text(encoding='utf-8')
    # Result decimals carry the exact reported scientific values; prose order may differ.
    if name=='results':
        dec=re.compile(r'(?<![A-Za-z])[-+]?\d+\.\d{2,}')
        checks.append({'item':'results reported decimals','pass':sorted(dec.findall(a))==sorted(dec.findall(b))})
    cite=lambda s:sorted(k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',s) for k in group.split(','))
    checks.append({'item':'citations/'+name,'pass':cite(a)==cite(b)})
checks.append({'item':'shared bibliography bytes','pass':(r/'paper/references.bib').read_bytes()==(r/'paper_zh/references.bib').read_bytes()})
record={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL','checks':checks,'scope':'Mechanical numerical/citation parity; independent human semantic review is in revision4_bilingual_method_review.md.','sources':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('paper','paper_zh') for p in (r/folder).rglob('*.tex')}}
(r/'review/bilingual_consistency.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(checks,ensure_ascii=False,indent=2))
raise SystemExit(0 if record['status']=='PASS' else 1)
