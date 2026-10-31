from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
path=HERE.parent/'cache_execution/handover_state.json'
state=json.loads(path.read_text(encoding='utf-8'))
state.update(stage='RAM_PERFORMANCE_TUNING_SAVED_CHECKPOINT1000',
    latest_user_instruction='Use all available RAM for cache and tune for fastest real training throughput; shut down after completion.',
    formal_training_status='TEMPORARILY_PAUSED_AT_SAVED_1000_FOR_FULL_RAM_CACHE_AND_LOADER_BENCHMARK',
    performance_tuning_in_progress=True,
    full_RAM_cache='454 x 3 FP32 probabilities in shared memfd; ~104.9GiB, persistent NPZ unchanged',
    resume_first='Current agent is tuning. Preserve last.pt@1000 and optimizer/RNG. Do not launch a duplicate process; await RAM_PIPELINE_BENCHMARK_PASS and selected performance_config before explicit resume.',
    server_controller='Previous controller completed repair hold; shutdown_hold.enabled temporarily protects cache tuning. Remove hold and relaunch controller after passing CPU pipeline benchmark.')
path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('RAM_TUNING_STATE_RECORDED')
