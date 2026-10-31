"""Verify saved audit numbers and local report links; never run model inference."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'HRDecoder_Ready_20261002'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    compact = next(HERE.glob('03_*.compact.json'))
    raw = json.loads(compact.read_text(encoding='utf-8-sig'))
    errors = []
    for node in raw['eval_nodes']:
        values = []
        for cls, counts in node['per_class'].items():
            denom = counts['tp'] + counts['fp'] + counts['fn']
            value = counts['tp'] / denom
            values.append(value)
            if abs(value - counts['iou']) > 1e-10:
                errors.append([node['run'], node['update'], cls, 'IoU'])
        if abs(sum(values) / 4 - node['miou']) > 1e-10:
            errors.append([node['run'], node['update'], 'mIoU'])

    sources = {SOURCE / 'reports' / 'HRDecoder_完整实验报告_20261003.md'}
    for node in raw['eval_nodes']:
        sources.add(SOURCE / node['source'])
    sources.update((SOURCE / 'trainer').rglob('*.py'))
    sources.update((SOURCE / 'server_results' / 'reports' / 'finish').glob('*.json'))
    sources.add(SOURCE / 'assets' / 'external' / 'PROVENANCE.json')

    missing_links = []
    for report in HERE.glob('*.md'):
        for link in re.findall(r'\]\(<?([^\n)]+?)>?\)', report.read_text(encoding='utf-8-sig')):
            if link.startswith(('https://', 'http://', '#')):
                continue
            p = Path(link.strip('<>'))
            if not p.is_absolute():
                p = report.parent / p
            if not p.exists():
                missing_links.append({'report': report.name, 'target': str(p)})

    result = {
        'scope': 'CPU arithmetic and files only; no GPU, SSH, training, or new test evaluation',
        'evaluation_nodes_recomputed': len(raw['eval_nodes']),
        'metric_errors': errors,
        'missing_local_report_links': missing_links,
        'source_files': [{'path': str(p), 'sha256': sha(p)} for p in sorted(sources) if p.is_file()],
        'missing_source_files': [str(p) for p in sorted(sources) if not p.is_file()],
        'output_files': [{'path': str(p), 'sha256': sha(p)} for p in sorted(HERE.iterdir())
                         if p.is_file() and p.name != 'audit_verification.json'],
    }
    (HERE / 'audit_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_files', 'output_files')}, ensure_ascii=False))
    if errors or missing_links or result['missing_source_files']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
