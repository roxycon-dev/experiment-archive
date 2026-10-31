"""Scalar screen-gate fixtures; never write candidate validation or use test."""
import copy
import json
from pathlib import Path
import tempfile
from screen_gate import screen_gate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
source = ROOT / 'training_runs/residual'
if not source.exists():
    source = ROOT / 'cache_execution/server_receipts/residual/training_runs/residual'
early = json.loads((source / 'val_04000.json').read_text(encoding='utf-8'))
later = json.loads((source / 'val_08000.json').read_text(encoding='utf-8'))
digest = 'synthetic-screen-fixture-not-a-training-identity'


def recompute(row):
    row['IoU_percent'] = [100*t/(t+f+n) for t,f,n in row['counts']]
    row['precision_percent'] = [100*t/(t+f) if t+f else 0 for t,f,n in row['counts']]
    row['recall_percent'] = [100*t/(t+n) for t,f,n in row['counts']]
    row['mIoU_percent'] = sum(row['IoU_percent'])/4


with tempfile.TemporaryDirectory(prefix='expert_gate_fixture_', dir=str(HERE)) as temp:
    work = Path(temp)
    (work/'protocol.json').write_text(json.dumps({'identity_SHA':digest,
        'initialization':'fresh ImageNet synthetic fixture'}))
    (work/'GPU_SMOKE_PASS.json').write_text(json.dumps({'identity_SHA':digest,
        'status':'PASS', 'fresh_initialization_verified':True}))
    a, b = copy.deepcopy(early), copy.deepcopy(later)
    a['completed'], b['completed'] = 2000, 4000
    (work/'val_02000.json').write_text(json.dumps(a))
    (work/'val_04000.json').write_text(json.dumps(b))
    rejected = screen_gate(work, HERE/'screen_protocol.json', digest)
    assert rejected['status']=='REJECTED'
    assert not rejected['decisions']['private_net_gain_over_base']
    improved = copy.deepcopy(b)
    for counts in improved['counts']:
        tp, fp, fn = counts
        gain = max(1, int(fn*.08))
        counts[:] = [tp+gain, int(fp*.85), fn-gain]
    recompute(improved)
    (work/'val_04000.json').write_text(json.dumps(improved))
    passed = screen_gate(work, HERE/'screen_protocol.json', digest)
    assert passed['status']=='PASS', passed
    bad = copy.deepcopy(improved)
    bad['counts'][0][0] += 1
    (work/'val_04000.json').write_text(json.dumps(bad))
    try:
        screen_gate(work, HERE/'screen_protocol.json', digest)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('Corrupted counts accepted')
    (work/'val_04000.json').unlink()
    try:
        screen_gate(work, HERE/'screen_protocol.json', digest)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Missing validation accepted')

receipt = {'status':'PASS', 'formal_updates':0, 'real_candidate_validation_generated':False,
    'fixtures':['insufficient_net_gain_rejected','consistent_counts_pass',
                'corrupt_counts_rejected','missing_point_rejected'],
    'old_gate_unchanged':True,'test_accessed':False,'accuracy_guaranteed':False}
(HERE/'SCREEN_GATE_CPU_PASS.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
