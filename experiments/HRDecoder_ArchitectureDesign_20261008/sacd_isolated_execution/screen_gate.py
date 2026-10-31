"""Two fresh full-val points, practical screening only; no test or architecture search."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'sacd_micro_execution'))
from gate import verify


def screen_gate(work, plan_path, digest):
    work, plan_path = Path(work), Path(plan_path)
    plan = json.loads(plan_path.read_text())['fresh_screen_if_warm_passes']
    root = plan_path.parent.parent
    warm = json.loads((root / 'reports/sacd_micro/MICRO_GATE.json').read_text())
    assert warm['status'] == 'PASS' and warm['full_val_counts_verified']
    smoke = json.loads((work / 'GPU_SMOKE_PASS.json').read_text())
    protocol = json.loads((work / 'protocol.json').read_text())
    assert smoke['status'] == 'PASS' and smoke['identity_SHA'] == protocol['identity_SHA'] == digest
    assert smoke['fresh_initialization_verified'] and protocol['initialization'].startswith('fresh ImageNet')
    assert smoke['all_gradients_finite']
    reference = json.loads((root / 'sacd_execution/val_label_reference.json').read_text())
    early = verify(json.loads((work / 'val_04000.json').read_text()), reference['ground_truth_positive_pixels'])
    later = verify(json.loads((work / 'val_08000.json').read_text()), reference['ground_truth_positive_pixels'])
    assert early['completed'] == 4000 and later['completed'] == 8000
    decisions = {
        '4k_at_least_old4k_plus1pp': early['mIoU_percent'] >= plan['4k_mIoU_at_least'],
        '8k_at_least_old8k_plus1pp': later['mIoU_percent'] >= plan['8k_mIoU_at_least'],
        'gain_4k_to8k_at_least3pp': later['mIoU_percent'] - early['mIoU_percent'] >= plan['8k_mIoU_gain_over_4k_pp_at_least'],
        'every_class_positive': all(x > 0 for x in later['IoU_percent']),
        'HE_IoU_not_falling': later['IoU_percent'][1] >= early['IoU_percent'][1],
        'within_original_deadline_when_granting_extension': time.time() < 1791568800,
    }
    return {'status': 'PASS' if all(decisions.values()) else 'REJECTED',
        'identity_SHA': digest, 'decisions': decisions, 'time_unix': time.time(),
        'full_val_counts_verified': True, 'fresh_initialization_required': True,
        'screen_updates': 8000, 'mIoU_4k': early['mIoU_percent'], 'mIoU_8k': later['mIoU_percent'],
        'criterion_registered_before_results': True, 'test_accessed': False,
        'accuracy_guaranteed': False, 'goal_achieved': False}
