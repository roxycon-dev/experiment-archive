"""Estimate deadline margin from archived observations, without polling SSH."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RECEIPTS=ROOT/'cache_execution/server_receipts/sacd'
TZ=timezone(timedelta(hours=8))
DEADLINE=datetime(2026,10,10,2,0,tzinfo=TZ)

def update_eta(snapshot):
    progress,perf=snapshot.get('progress'),snapshot.get('performance')
    if snapshot.get('outcome') or not progress or not perf:return None
    # Warmup/no-action batches must not stand in for the routed auxiliary phase.
    if perf['completed_update']<1600 or perf['window_batches']<100:return None
    observation=datetime.fromtimestamp(snapshot['runtime']['checked_at_unix'],TZ)
    completed=progress['completed']
    mean=perf['mean_batch_seconds']
    assert 0<mean<100 and 0<=completed<=40000
    vals=[json.loads(p.read_text()) for p in (RECEIPTS/'training_runs/sacd').glob('val_*.json')]
    done={v['completed'] for v in vals}
    remaining_points=[s for s in range(4000,40001,4000) if s not in done]
    durations=[v['seconds'] for v in vals]
    val_seconds=max(durations)+2 if durations else 35
    train_seconds=(40000-completed)*mean
    remaining_val_seconds=len(remaining_points)*val_seconds
    save_reserve,test_reserve=120,180
    train_and_validation_end=observation+timedelta(seconds=train_seconds+remaining_val_seconds+save_reserve)
    final_end=train_and_validation_end+timedelta(seconds=test_reserve)
    result={'checked_at':observation.isoformat(),'completed':completed,'mean_step_seconds':mean,
        'performance_update':perf['completed_update'],'remaining_validation_steps':remaining_points,
        'measured_val_seconds':durations,'per_validation_estimate_seconds':val_seconds,
        'checkpoint_save_reserve_seconds':save_reserve,'conditional_final_test_reserve_seconds':test_reserve,
        'estimated_training_and_full_validation_complete':train_and_validation_end.isoformat(),
        'estimated_final_evaluation_complete':final_end.isoformat(),
        'estimated_headroom_before_deadline_seconds':(DEADLINE-final_end).total_seconds(),
        'estimated_training_headroom_before_0157_cutoff_seconds':
            (DEADLINE-timedelta(minutes=3)-train_and_validation_end).total_seconds(),
        'deadline':DEADLINE.isoformat(),'deadline_risk':final_end>=DEADLINE,
        'basis':'Last measured100 routed updates; observed validation maximum plus2s; save120s/test180s. Uncertain extrapolation.',
        'prediction_not_verified_completion':True,
        'stale_observation_minutes_at_calculation':(datetime.now(TZ)-observation).total_seconds()/60}
    (HERE/'ETA_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    state_path=ROOT/'iterative_execution/iteration_state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('active_run')=='sacd':
        state['latest_ETA_analysis']=result
        state['expected_full_train_and_final_evaluation_beijing']=final_end.isoformat()
        state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    snapshot=json.loads((RECEIPTS/'latest_snapshot.json').read_text(encoding='utf-8'))
    print(json.dumps(update_eta(snapshot),ensure_ascii=True))
