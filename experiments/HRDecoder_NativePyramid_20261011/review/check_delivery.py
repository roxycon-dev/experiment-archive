"""Check document coverage/links only. No training, model loading or pixel reads."""
import collections
import hashlib
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
project = here.parent
workspace = project.parent
inventory = json.loads((here / 'reading_inventory.json').read_text(encoding='utf-8-sig'))
if isinstance(inventory, dict):
    inventory = inventory.get('entries', inventory.get('documents', []))
expected = {x['primary_path']: x for x in inventory}
rows, reviewers = {}, {}
for file in sorted(here.glob('coverage_*.json')):
    data = json.loads(file.read_text(encoding='utf-8-sig'))
    items = data if isinstance(data, list) else data.get('entries', data.get('documents', []))
    reviewers[file.stem] = dict(collections.Counter(x['actual_read_status'] for x in items))
    for row in items:
        name = row['primary_path']
        if name in rows:
            raise ValueError(f'Duplicate coverage: {name}')
        rows[name] = row
errors = []
for name, entry in expected.items():
    if name not in rows:
        errors.append(f'Missing coverage: {name}')
        continue
    actual = hashlib.sha256((workspace / name).read_bytes()).hexdigest()
    if actual != entry['sha256'] or rows[name]['sha256'] != actual:
        errors.append(f'Changed source identity: {name}')
extra = sorted(set(rows) - set(expected))
if extra:
    errors.append(f'Unexpected coverage entries: {extra}')
proposal = project / '00_最终架构方案.md'
review_summary = project / '01_审查结论与阅读范围.md'
documents = [proposal, review_summary]
for document in documents:
    body = document.read_text(encoding='utf-8')
    for target in re.findall(r'\]\((D:/[^)]+)\)', body):
        if not Path(target).exists():
            errors.append(f'Missing linked file in {document.name}: {target}')
result = {
    'scope': 'Markdown coverage source SHA and delivery document links only; not model or performance verification',
    'inventory_unique': len(expected), 'coverage_unique': len(rows),
    'counts': dict(collections.Counter(x['actual_read_status'] for x in rows.values())),
    'reviewers': reviewers, 'errors': errors,
    'proposal_sha256': hashlib.sha256(proposal.read_bytes()).hexdigest(),
    'review_summary_sha256': hashlib.sha256(review_summary.read_bytes()).hexdigest(),
    'checked_documents': [str(x) for x in documents],
    'status': 'PASS' if not errors and len(expected) == 265 else 'FAIL',
}
(here / 'delivery_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
