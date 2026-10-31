"""Read the platform shutdown wrapper without executing it or exposing secrets."""
import ast
from pathlib import Path
import re
import shlex
import sys
from urllib.parse import urlsplit

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from qlib.common import sha,write_json
path=root/'assets/shutdown_control.private'
blob=path.read_bytes()
report={'executed':False,'bytes':len(blob),'sha256':sha(path),'type':'ELF' if blob.startswith(b'\x7fELF') else 'text'}
if report['type']=='text':
    source=blob.decode('utf-8',errors='replace')
    urls=[]
    for raw in re.findall(r'https?://[^\s\x22\x27<>]+',source):
        u=urlsplit(raw)
        parts=['<private>' if len(p)>24 or re.fullmatch(r'[0-9a-fA-F-]{16,}',p) else p for p in u.path.split('/')]
        urls.append({'host':u.hostname,'path':'/'.join(parts),'query_present':bool(u.query)})
    report['url_endpoints_sanitized']=urls
    try:
        tree=ast.parse(source)
        report['imports']=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
        report['call_names']=sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)})
    except SyntaxError:
        report['python_ast']=False
        commands=[]
        for line in source.splitlines():
            try:
                tokens=shlex.split(line,comments=True)
            except ValueError:
                continue
            if tokens and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,31}',tokens[0]):
                commands.append(tokens[0])
        report['shell_command_names']=commands
        report['referenced_script_paths']=[p for p in re.findall(r'/[A-Za-z0-9_.\-/]+\.(?:py|sh)',source)
            if all(len(segment)<64 for segment in p.split('/'))]
write_json(root/'reports/shutdown_control_inspection.json',report)
print(report)
