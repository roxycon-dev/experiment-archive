"""Run mechanical paper checks and preserve exact output; no training or inference."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
skill = Path(r'C:\Users\GuanXingRen\.codex\skills\ai-research-writing-skill\scripts')
checks = [
    ('numeric-main', 'check_numeric_evidence.py', [str(root)]),
    ('numeric-companion', 'check_numeric_evidence.py', [str(root), '--main', 'paper/supplement.tex']),
    ('numeric-chinese', 'check_numeric_evidence.py', [str(root), '--main', 'paper_zh/main.tex']),
    ('numeric-chinese-companion', 'check_numeric_evidence.py', [str(root), '--main', 'paper_zh/supplement.tex']),
    ('chinese-citation-keys', 'check_citations.py', [str(root/'paper_zh/main.tex'), str(root/'paper_zh/references.bib')]),
    ('chinese-build-log', 'parse_build_log.py', [str(root/'paper_zh/main.log')]),
    ('chinese-companion-build-log', 'parse_build_log.py', [str(root/'paper_zh/supplement.log')]),
    ('citation-keys', 'check_citations.py', [str(root/'paper/main.tex'), str(root/'paper/references.bib')]),
    ('citation-lock', 'check_citation_lock.py', [str(root)]),
    ('draft-quality-gate', 'research_quality_gate.py', [str(root)]),
    ('main-build-log', 'parse_build_log.py', [str(root/'paper/main.log')]),
    ('companion-build-log', 'parse_build_log.py', [str(root/'paper/supplement.log')]),
    ('markers', 'check_todos.py', [str(root/'paper/main.tex'),str(root/'paper/sections'),str(root/'paper/supplement.tex'),str(root/'paper_zh/main.tex'),str(root/'paper_zh/sections'),str(root/'paper_zh/supplement.tex')]),
]
parts=[]
failed=[]
env=dict(os.environ,PYTHONUTF8='1')
for label, script, args in checks:
    cmd=[sys.executable,str(skill/script),*args]
    p=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
    parts.append(f'## {label}\ncommand: {cmd!r}\nexit_code: {p.returncode}\n{p.stdout}\n{p.stderr}\n')
    print(label, p.returncode, p.stdout.strip())
    if p.returncode: failed.append(label)
(root/'review/final_checks.log').write_text('\n'.join(parts),encoding='utf-8')
raise SystemExit(bool(failed))
