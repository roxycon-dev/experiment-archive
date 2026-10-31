import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
state_path=HERE.parent/'cache_execution/handover_state.json'
state=json.loads(state_path.read_text(encoding='utf-8'))
state.update(historical_checkpoint_comparison_required=True,
    historical_comparison_report=str(HERE/'历史检查点对比.md'),
    historical_comparison_data=str(HERE/'historical_checkpoint_comparison.json'),
    comparison_reference='Historical original HRDecoder same-step val; fixed40k38.835%, best32k39.390%; not a newly reproduced paired control',
    heartbeat_check_interval_minutes=45)
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (HERE/'验证指标与趋势.md').open('a',encoding='utf-8') as f:
    f.write('\n历史参照已核对：原baseline4k21.557%，本轮4k29.4859%，差+7.9289个百分点。CE+Dice4k31.161%、combo4k35.666%，仍高于本轮。原baseline8k32.132%、12k34.479%；后续按同节点继续比较。完整证据见[历史检查点对比](历史检查点对比.md)。\n')
