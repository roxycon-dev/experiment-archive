import json,pathlib
root=pathlib.Path(r'D:\dataset\experiment\HRDecoder_ArchitectureDesign_20261008\cache_execution\server_receipts')
for p in sorted(root.rglob('protocol.json')):
    try:j=json.loads(p.read_text(encoding='utf-8'))
    except Exception:continue
    code=j.get('identity',{}).get('code')
    if code: print(p.relative_to(root), 'identity='+str(j.get('identity_SHA') or j.get('protocol_sha256') or ''), 'code='+json.dumps(code,sort_keys=True))
