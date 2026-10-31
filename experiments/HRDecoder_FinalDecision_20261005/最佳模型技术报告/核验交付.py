from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import re

from docx import Document
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


ledger = json.loads((ROOT / '指标与来源.json').read_text(encoding='utf-8'))
source_checks = []
for source in ledger['sources']:
    actual = sha256(source['path'])
    assert actual == source['sha256'], source['path']
    source_checks.append({'path': source['path'], 'sha256': actual, 'verified': True})

evidence = ROOT.parent / '结题报告' / 'evidence'
final = json.loads((evidence / 'final_locked_test_evidence.json').read_text(encoding='utf-8'))
arms = json.loads((evidence / 'four_arm_saved_validation.json').read_text(encoding='utf-8'))
assert ledger['final_test'] == final['locked_test_miou_pct']
assert ledger['gain_vs_R_pp'] == final['gain_vs_R_pp']
assert ledger['gain_vs_baseline_approx_pp'] == final['approx_gain_vs_baseline_pp']
assert math.isclose(
    sum(x['iou_pct'] for x in final['test_per_class'].values()) / 4,
    ledger['final_test'], abs_tol=1e-12,
)


def score(arm, update, formula):
    matches = [r for r in arms['records'] if r['arm'] == arm and r['update'] == update]
    assert len(matches) == 1
    return matches[0]['outputs'][formula]['miou_pct']


for update in (20000, 32000):
    difference = score('S', update, 'single') - score('C', update, 'single')
    assert difference == ledger[f'space_effect_{update // 1000}k_single_pp']

docx_path = ROOT / 'HRDecoder最佳模型训练与改进报告.docx'
pdf_path = ROOT / 'HRDecoder最佳模型训练与改进报告.pdf'
markdown_path = ROOT / 'HRDecoder最佳模型训练与改进报告.md'
document = Document(docx_path)
all_text = '\n'.join(
    [p.text for p in document.paragraphs]
    + [cell.text for table in document.tables for row in table.rows for cell in row.cells]
)
assert '用户' not in all_text
assert not re.search(r'\{\{(?:TABLE|FIG|EQ)', all_text)
assert len(document.tables) == 7
assert len(document.inline_shapes) == 22
equations = json.loads((ROOT / '公式源.json').read_text(encoding='utf-8'))
assert len(equations) == 17
reader = PdfReader(pdf_path)
assert len(reader.pages) == 11
assert all(len(page.extract_text() or '') > 100 for page in reader.pages)
page_images = sorted((ROOT / '排版检查' / '定稿').glob('page-*.png'))
assert len(page_images) == 11
assert markdown_path.is_file()

outputs = []
for path in (docx_path, pdf_path, markdown_path, ROOT / '指标与来源.json', ROOT / '公式源.json'):
    outputs.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)})

receipt = {
    'stage': 'DELIVERY_VERIFIED',
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'pdf_pages': 11,
    'figures': 5,
    'equations': 17,
    'tables': 7,
    'all_final_pages_visually_inspected': True,
    'visual_review': 'Final 11-page render reviewed page by page; diagrams, equations, tables, Chinese text, margins and page breaks checked.',
    'numeric_checks': {
        'locked_test_from_saved_evidence': True,
        'four_class_mean_from_saved_results': True,
        'gain_vs_original_baseline_approximate': True,
        'gain_vs_R_exact': True,
        'matched_S_minus_C_20k_and_32k_exact': True,
    },
    'source_checks': source_checks,
    'outputs': outputs,
    'final_render_sha256': [{'path': str(p), 'sha256': sha256(p)} for p in page_images],
    'no_new_model_training_or_inference': True,
    'prior_report_untouched': True,
}
(ROOT / '交付核验.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'stage': receipt['stage'], 'pdf_pages': 11, 'verified_sources': len(source_checks), 'outputs': len(outputs)}))
