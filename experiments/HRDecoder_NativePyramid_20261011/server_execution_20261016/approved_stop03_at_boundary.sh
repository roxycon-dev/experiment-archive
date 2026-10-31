set -eu
test "${ALLOW_ENDING_CURRENT_FORMAL_RUN:-}" = USER_CONFIRMED
BASE=/root/autodl-tmp/nativepyramid_sp_v3_20261016
cd "$BASE"
"$BASE/venv/bin/python" - <<'PY'
import json,os,signal,time,shutil,uuid
from pathlib import Path
from performance04_transition_support import (BASE,OLD_SOURCE,require,read,process,alive,active_group,validate_candidate)
b=BASE;run=b/'run_seed42_launch02'
evidence=validate_candidate()
pid=2562;owner=2160
identity=process(pid);owner_identity=process(owner)
require(identity is not None and owner_identity is not None,'Old owned processes are absent')
require(identity['pgrp']==pid and identity['ppid']==owner,'Old process ownership changed')
command=Path(f'/proc/{pid}/cmdline').read_bytes()
require(b'train.py' in command and str(run).encode() in command,'Different training process')
require(b'controller.py' in Path(f'/proc/{owner}/cmdline').read_bytes()
        and Path(f'/proc/{owner}/cwd').resolve()==b,'Different controller process')
require(read(b/'job_status.json').get('source_sha256')==OLD_SOURCE,'Old source changed')
transaction=b/'receipts'/('approved_stop_transaction_'+uuid.uuid4().hex+'.json')
record={'status':'PREPARING','human_approved_new_trial':True,'old_training_identity':identity,
        'old_controller_identity':owner_identity,'transaction_path':str(transaction),**evidence,'phase_history':[]}
held=owner_held=termination_started=committed=False

def persist(path,value):
    temporary=path.with_suffix('.writing')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)

def phase(status):
    record['status']=status;record['phase_history'].append({'phase':status,'unix_time':time.time()})
    persist(transaction,record)

def wait_stopped(expected):
    deadline=time.monotonic()+10
    while time.monotonic()<deadline:
        current=process(expected['pid'])
        require(current and current['starttime']==expected['starttime'],'Process disappeared or PID reused')
        states=[]
        for path in Path(f"/proc/{expected['pid']}/task").glob('*/status'):
            try:states.append(next(line.split()[1] for line in path.read_text().splitlines() if line.startswith('State:')))
            except FileNotFoundError:pass
        if states and all(s=='T' for s in states):return
        time.sleep(.01)
    raise RuntimeError('Owned thread group did not confirm stopped state')

try:
    phase('PREPARING');deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        p=read(run/'progress.json')
        if p['in_flight_update'] is None and p['executed_updates']>0:
            require(alive(identity),'Training identity no longer alive')
            os.killpg(pid,signal.SIGSTOP);held=True;wait_stopped(identity)
            confirmed=read(run/'progress.json')
            if confirmed['in_flight_update'] is None:break
            os.killpg(pid,signal.SIGCONT);held=False
        time.sleep(.002)
    else:raise RuntimeError('No complete SGD boundary acquired; continue original run')
    require(alive(owner_identity),'Controller identity changed')
    os.kill(owner,signal.SIGSTOP);owner_held=True;wait_stopped(owner_identity)
    p=confirmed;observed=0
    with (run/'train.jsonl').open() as f:
        for line in f:
            row=json.loads(line);observed+=1
            require(row.get('update')==observed and row.get('executed_updates')==observed,
                    'Complete old ledger is not contiguous')
    require(observed==p['executed_updates']==p['completed'],'Old ledger differs from stopped progress')
    require(p['run_identity']['source_sha256']==OLD_SOURCE,'Stopped run source changed')
    record.update(source_sha256=OLD_SOURCE,executed_updates=observed,
        checkpoint_persisted_updates=p['checkpoint_persisted_updates'],in_flight_update=None,
        additional_executed_updates_uncertain=[0,0],complete_ledger_validated=True,
        retained_directory=str(run),old_executed_steps_are_additional_disclosed_budget=True,
        new_trial_must_start_from_ImageNet=True,shutdown_requested=False)
    history=b/('control_attempt03_approved_stop_'+transaction.stem[-12:]);history.mkdir(exist_ok=False)
    for name in ('job_status.json','controller.py','controller_v3_started.json'):shutil.copy2(b/name,history/name)
    phase('STOP_PREPARED_NOT_EXECUTED')
    termination_started=True
    os.killpg(pid,signal.SIGKILL)
    os.kill(owner,signal.SIGKILL)
    deadline=time.monotonic()+15
    while alive(identity) or alive(owner_identity) or active_group(pid):
        if time.monotonic()>deadline:raise RuntimeError('PARTIAL_STOP: old processes have not all exited')
        time.sleep(.05)
    record['resume_forbidden']=True;phase('STOP_CONFIRMED')
    persist(run/'abandoned_for_approved_performance04.json',record)
    # Canonical receipt is the single final atomic commit point. All required
    # mirrors already exist; no fallible state mutation follows this commit.
    persist(b/'receipts/stopped03_for_approved_performance04.json',record)
    committed=True
    print(json.dumps(record),flush=True)
except BaseException as error:
    if not committed:
        record['error']=type(error).__name__+': '+str(error)
        try:phase('PARTIAL_STOP_NO_NEW_TRAINING' if termination_started else 'ABORTED_RECOVERY_PENDING')
        except OSError:pass
    raise
finally:
    if not termination_started:
        recovery_errors=[]
        for target,was_held,is_group in ((owner_identity,owner_held,False),(identity,held,True)):
            try:
                if was_held and alive(target):
                    if is_group:os.killpg(target['pid'],signal.SIGCONT)
                    else:os.kill(target['pid'],signal.SIGCONT)
            except BaseException as error:
                recovery_errors.append(type(error).__name__+': '+str(error))
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            snapshots=[process(target['pid']) for target in (owner_identity,identity)]
            recovered=all(current and current['starttime']==expected['starttime']
                          and current['state'] not in ('T','t','Z')
                          for current,expected in zip(snapshots,(owner_identity,identity)))
            if recovered:break
            time.sleep(.01)
        record['recovery_signal_errors']=recovery_errors
        record['recovery_confirmed']=recovered
        try:phase('ABORTED_CURRENT_RUN_CONTINUES' if recovered else 'ABORTED_RECOVERY_FAILED_NO_NEW_TRAINING')
        except OSError:pass
PY
