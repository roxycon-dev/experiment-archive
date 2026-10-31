"""Record latest user authorization, then start only the bounded staging controller."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'local_preparation'))
from remote import connect
REMOTE = '/root/autodl-tmp/HRDecoder_v3_20261008'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    deployed = json.loads((HERE / 'DEPLOYMENT_CPU_PASS.json').read_text())
    assert deployed['status'] == 'CPU_DEPLOYMENT_PASS'
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    state_path = ROOT / 'iterative_execution/iteration_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    state['latest_user_authorization'] = {
        'recorded_at': stamp, 'micro_diagnostic_authorized': True, 'one_fresh8k_candidate_screen_authorized': True,
        'full40k_continuation_condition': 'registered warm and fresh screens PASS',
        'deadline_extension_condition': 'User explicitly answered: 通过筛查后，可延后至完成',
        'unconditional_full40k_restart_authorized': False, 'additional_candidates_allowed': 1,
        'no_accuracy_guarantee': True}
    state.update(status='SACD_RUNNING_WITH_20K_SCREENING_TRANSITION_STAGED', next_run='sacd_micro_then_conditional_isolated',
                 original_deadline_UTC_plus_8=state['deadline_UTC_plus_8'],
                 conditional_extension_activated=False, another_full_run_authorized=False,
                 screening_plan=str(HERE / 'micro_protocol.json'), screening_monitor=str(HERE / 'monitor_screening.py'))
    write(state_path, state)
    hand_path = ROOT / 'cache_execution/handover_state.json'
    hand = json.loads(hand_path.read_text(encoding='utf-8'))
    hand.update(stage='SACD_RUNNING_SCREENING_TRANSITION_PREPARED',
                latest_user_instruction='One paired500-step diagnostic; if PASS one fresh8k feature-isolated screen. Full40k continuation and deadline extension only after registered screening PASS. Never shutdown or control external diagnoses.',
                active_execution_directory=str(ROOT / 'sacd_execution'),
                active_remote_execution_directory=REMOTE + '/sacd_execution',
                active_remote_run=REMOTE + '/training_runs/sacd', active_remote_reports=REMOTE + '/reports/sacd',
                latest_conditional_authorization=state['latest_user_authorization'],
                local_monitor_script=str(HERE / 'monitor_screening.py'))
    write(hand_path, hand)
    code = '''import json,os,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/HRDecoder_v3_20261008')
reports=root/'reports/sacd_micro';reports.mkdir(exist_ok=True)
assert not (reports/'STAGING_DISPATCHED.json').exists(),'Staging already dispatched'
with open(reports/'controller.log','a',buffering=1) as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-u',str(root/'sacd_micro_execution/controller.py')],cwd=str(root/'sacd_micro_execution'),stdout=log,stderr=log,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4'))
receipt={'status':'STAGING_DISPATCHED','controller_pid':p.pid,'GPU_parent_continues_until20k_save':True,'server_shutdown':False,'full40k_restart_authorized_now':False}
(reports/'STAGING_DISPATCHED.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
'''
    client = connect()
    try:
        _, out, err = client.exec_command('/root/miniconda3/bin/python -c ' + shlex.quote(code))
        raw, error = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(error)
        receipt = json.loads(raw)
    finally:
        client.close()
    write(HERE / 'STAGING_DISPATCHED.json', receipt)
    state['screening_staging_controller'] = receipt; write(state_path, state)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
