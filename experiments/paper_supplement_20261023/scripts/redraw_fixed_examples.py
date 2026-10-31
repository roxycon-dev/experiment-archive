"""Render only the already-frozen example selection; never reselect cases."""
import json
from pathlib import Path
from PIL import Image
from analyze_fixed_run import draw_example

ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'results/ddr_test_local_v1'
selection=json.loads((run/'analysis/example_selection.json').read_text(encoding='utf-8'))
first={}
for example in selection['examples']:
    panel=draw_example(example,run,'test',run/'analysis/figures')
    first.setdefault(example['category'],panel)
canvas=Image.new('RGB',(1000,324*len(first)),'white')
for i,category in enumerate(['recovered','still_missed','new_fp']):canvas.paste(first[category],(0,i*324))
canvas.save(run/'analysis/figures/ma_examples_contact.png',dpi=(300,300))
(run/'analysis/example_selection.json').write_text(json.dumps(selection,indent=2,ensure_ascii=False),encoding='utf-8')
(run/'analysis/figures/render_note.json').write_text(json.dumps({'unchanged_selection':True,'resampling':'nearest-neighbor display enlargement',
  'white_box':'preselected target component bounding box plus 3 native pixels; only display annotation',
  'colors':'GT green; predictions orange; contours for visibility only; no metric filtering'},indent=2),encoding='utf-8')
