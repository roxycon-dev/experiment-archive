import json, pathlib
root = pathlib.Path(r'D:\dataset\experiment')
review = root/'HRDecoder_NativePyramid_20261011'/'review_20261016'
inv = json.loads((review/'inventory_all.json').read_text(encoding='utf-8'))
bysha={}
for x in inv['files']:
    if x.get('sha256'): bysha.setdefault(x['sha256'],[]).append(x)
cat=json.loads((review/'runs_protocol_catalog.json').read_text(encoding='utf-8'))
for run, rec in cat.items():
    src=root/rec['source']
    p=json.loads(src.read_text(encoding='utf-8'))
    identity=p.get('identity',{})
    codes=identity.get('code',{})
    print('\nRUN',run,'PROTO',rec['source'])
    print(' protocolSHA',rec.get('identity_SHA') or p.get('identity_SHA'))
    for name,sha in codes.items():
        matches=bysha.get(sha,[])
        print(name,sha,' | '.join(m['path'] for m in matches[:4]),'matches',len(matches))
