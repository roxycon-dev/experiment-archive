"""Record the observed 20k anomaly and CPU-only diagnostic evidence."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIAG = ROOT / 'cache_execution/server_receipts/reports/diagnostics'
health = json.loads((DIAG / 'checkpoint_health_20000.json').read_text(encoding='utf-8'))
probe = json.loads((DIAG / 'cpu_output_20000_positive_EX_SE.json').read_text(encoding='utf-8'))
assert health['last']['completed'] == 20000 and health['last']['numeric_health'] == 'PASS'
assert probe['status'] == 'COMPLETE' and probe['formal_SGD_updates'] == 0
assert probe['GT_positive_pixels'][0] > 0 and probe['GT_positive_pixels'][2] > 0
assert all(c['finite'] for c in probe['original_resolution_probabilities'])
assert probe['original_resolution_probabilities'][0]['pixels_above_0p5'] == 0
assert probe['original_resolution_probabilities'][2]['pixels_above_0p5'] == 0
record = {'step': 20000, 'mIoU_percent': 15.050912466184402,
          'zero_positive_prediction_classes': ['EX', 'SE'],
          'checkpoint_numeric_health': 'PASS',
          'diagnostic_checkpoint': health['last']['retained_diagnostic_checkpoint'],
          'CPU_FP32_positive_image_probe': str(DIAG / 'cpu_output_20000_positive_EX_SE.json'),
          'CPU_checkpoint_health_report': str(DIAG / 'checkpoint_health_20000.json'),
          'observed_suppression_on_positive_image': True,
          'proven_implementation_defect': False, 'cause': 'UNRESOLVED',
          'protocol_changed': False, 'additional_formal_SGD_updates': 0,
          'budget': 'v3, 2048, batch2, 40000 fixed; continue authorized main run',
          'followup': 'At later scheduled validation, check restoration of EX/SE positive predictions; keep quiet if unchanged and no new validation.',
          'created_at_UTC_plus_8': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}
target = HERE / 'validation_anomaly_20000.json'
target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
state_path = ROOT / 'cache_execution/handover_state.json'
state = json.loads(state_path.read_text(encoding='utf-8'))
state.update(validation_anomaly=record, validation_anomaly_report=str(target))
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
with (HERE / '4090训练启动记录.md').open('a', encoding='utf-8') as stream:
    stream.write('\n20k异常审查：val mIoU15.0509%，EX/SE均无正预测；CPU检查点有限值检查通过，含EX/SE病灶的2048 FP32单图诊断仍输出极低概率。未定位可直接修复的实现错误；保留checkpoint_20000.pt诊断副本、best-val@8k及主训练last.pt。保持原40k协议继续，原因待定。详见validation_anomaly_20000.json及验证指标与趋势.md。\n')
print(json.dumps({'report': str(target), 'numeric_health': 'PASS', 'cause': 'UNRESOLVED', 'protocol_changed': False}))
