"""Pure metric audit; no GPU, test labels or checkpoint selection."""
import json
import math
from pathlib import Path

CLASSES = ['EX', 'HE', 'SE', 'MA']


def verify(value, positive_pixels):
    assert value['images'] == 76 and value['class_order'] == CLASSES
    assert value['threshold'] == 'strictly > .5'
    for key in ('', 'base_head_metrics'):
        row = value[key] if key else value
        assert len(row['counts']) == 4
        for c, triple in enumerate(row['counts']):
            assert all(type(x) is int and x >= 0 for x in triple)
            tp, fp, fn = triple
            assert tp + fn == positive_pixels[c]
            for field, denominator in [('IoU_percent', tp + fp + fn),
                                       ('precision_percent', tp + fp), ('recall_percent', tp + fn)]:
                calculated = 100 * tp / denominator if denominator else 0.
                assert math.isclose(calculated, row[field][c], abs_tol=1e-8)
        assert math.isclose(sum(row['IoU_percent']) / 4, row['mIoU_percent'], abs_tol=1e-8)
    assert value['teacher_or_GT_input'] is False
    return value


def evaluate_gate(root, plan):
    work = Path(root) / 'training_runs/sacd_micro'
    ref = json.loads((Path(root) / 'sacd_execution/val_label_reference.json').read_text())
    assert ref['status'] == 'PASS'
    original_protocol = json.loads((Path(root) / 'training_runs/sacd/protocol.json').read_text())
    assert ref['metadata_file_SHA256'] == original_protocol['identity']['data_identity_SHA']
    records = {}
    for arm in plan['arms']:
        done = json.loads((work / arm / 'PROBE_COMPLETE.json').read_text())
        smoke = json.loads((work / arm / 'GPU_SMOKE_PASS.json').read_text())
        protocol = json.loads((work / arm / 'protocol.json').read_text())
        assert done['status'] == 'COMPLETE' and smoke['status'] == 'PASS'
        assert done['probe_updates'] == plan['warm_updates_per_arm']
        assert done['identity_SHA'] == smoke['identity_SHA'] == protocol['identity_SHA']
        records[arm] = {n: verify(json.loads((work / arm / ('val_probe_%05d.json' % n)).read_text()),
                                 ref['ground_truth_positive_pixels']) for n in (0, 250, 500)}
    control, modified = records['control'], records['feature_isolated']
    parent_val = json.loads((Path(root) / 'training_runs/sacd/val_20000.json').read_text())
    initial_matches = (control[0]['counts'] == modified[0]['counts'] == parent_val['counts']
                       and control[0]['base_head_metrics']['counts'] == modified[0]['base_head_metrics']['counts']
                       == parent_val['base_head_metrics']['counts'])
    gate = plan['go_gate_all_required']
    gain250 = modified[250]['mIoU_percent'] - control[250]['mIoU_percent']
    gain500 = modified[500]['mIoU_percent'] - control[500]['mIoU_percent']
    old_fp = control[500]['counts'][1][1]
    ratio = modified[500]['counts'][1][1] / old_fp if old_fp else (0. if modified[500]['counts'][1][1] == 0 else float('inf'))
    recall_loss = control[500]['recall_percent'][1] - modified[500]['recall_percent'][1]
    iou_losses = [a - b for a, b in zip(control[500]['IoU_percent'], modified[500]['IoU_percent'])]
    decisions = {
        'same_initial_val_counts_both_heads_and_parent': initial_matches,
        'nonnegative_gain_at250': gain250 >= gate['modified_mIoU_gain_over_control_at_250_pp_at_least'],
        'gain_at500_at_least1pp': gain500 >= gate['modified_mIoU_gain_over_control_at_500_pp_at_least'],
        'HE_FP_ratio_at_most_point8': ratio <= gate['modified_HE_false_positives_ratio_to_control_at_500_at_most'],
        'HE_recall_loss_at_most2pp': recall_loss <= gate['modified_HE_recall_loss_to_control_at_500_pp_at_most'],
        'no_class_IoU_loss_over2pp': max(iou_losses) <= gate['any_class_IoU_loss_to_control_at_500_pp_at_most'],
    }
    return {'status': 'PASS' if all(decisions.values()) else 'REJECTED',
            'purpose': 'Allow or reject one fresh8k screen, not certify40k accuracy',
            'gate_registered_before_results': True, 'decisions': decisions,
            'gain_at250_pp': gain250, 'gain_at500_pp': gain500,
            'initial_mIoU_difference_modified_minus_control_pp': modified[0]['mIoU_percent'] - control[0]['mIoU_percent'],
            'initial_count_equality_required_not_relaxed': True,
            'HE_false_positive_ratio': ratio if math.isfinite(ratio) else None,
            'HE_recall_loss_pp': recall_loss, 'class_IoU_loss_pp': iou_losses,
            'class_order': CLASSES, 'full_val_counts_verified': True,
            'records': records, 'test_accessed': False, 'goal_achieved': False}
