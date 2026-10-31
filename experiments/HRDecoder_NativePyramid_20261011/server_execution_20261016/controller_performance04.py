"""Run preflight -> 40k -> eligible test; shut down only after completion.

Setup, preflight, failure and PAUSED states preserve the instance for repair.
Shutdown is armed only after an observed formal SGD update on this source.
"""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

BASE = Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
CODE = BASE/'code_performance04_candidate'
PYTHON = str(BASE/'venv/bin/python')
PREFLIGHT = BASE/'preflight_performance04'
RUN = BASE/'run_seed42_performance04'
ASSET = BASE/'assets/hrnetv2_w48-d2186c55.pth'
IDENTITY = CODE/'preparation_20261016/data_identity.json'
STATUS = BASE/'job_status.json'
SOURCE_SHA = '14e47c3749db5dd28c2938ddbb227aad66b07c40b4fb9d049fdb0564b91c95d0'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.writing')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


state = {'controller_version':4,'started_at':now(),'source_sha256':SOURCE_SHA,
         'shutdown_policy':'confirmed_formal_start_then_successful_completion_only',
         'shutdown_armed':False,'status':'STARTING','formal_training_started':False}


def observe_formal_start(progress):
    if not isinstance(progress,dict):
        return
    updates=progress.get('executed_updates')
    identity=progress.get('run_identity')
    if (type(updates) is int and 1 <= updates <= 40000 and isinstance(identity,dict)
            and identity.get('source_sha256')==SOURCE_SHA):
        if not state.get('formal_training_started'):
            state.update(formal_training_started=True,formal_training_confirmed_at=now(),
                         formal_start_evidence_updates=updates)
        state['shutdown_armed']=True


def stage(name, arguments, log_name):
    state.update(status=name,stage_started_at=now(),child_exit_code=None)
    write(STATUS,state)
    with (BASE/log_name).open('x') as log:
        child=subprocess.Popen([PYTHON,*arguments],cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,
                               start_new_session=True)
        try:
            state['child_pid']=child.pid;write(STATUS,state)
            while child.poll() is None:
                state['observed_at']=now()
                if name=='PREFLIGHT':
                    state['diagnostic_progress']=read(PREFLIGHT/'diagnostic_progress.json')
                if name=='FORMAL_TRAINING':
                    progress=read(RUN/'progress.json')
                    observe_formal_start(progress)
                    state['formal_progress']=None if not progress else {
                        k:progress.get(k) for k in ('status','completed','executed_updates','checkpoint_persisted_updates','in_flight_update')}
                write(STATUS,state)
                time.sleep(5)
        finally:
            if child.poll() is None:
                # Give the owned child a chance to persist its failure/in-flight
                # state before shutdown. Never kill unrelated server workloads.
                try:child.send_signal(signal.SIGINT)
                except ProcessLookupError:pass
                try:
                    child.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    state['child_forced_stop_boundary_uncertain']=True
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        if name=='FORMAL_TRAINING':
            observe_formal_start(read(RUN/'progress.json'))
        state['child_exit_code']=child.returncode
        state['stage_finished_at']=now();write(STATUS,state)
        return child.returncode


def run_job():
    from performance04_transition_support import validate_committed_stop
    transition=validate_committed_stop()
    state['additional_disclosed_old_formal_updates']=transition['executed_updates']
    if not (BASE/'receipts/environment_ready').is_file():
        raise RuntimeError('Environment is not ready')
    if RUN.exists():
        raise RuntimeError('New performance run output must be fresh')
    production=read(BASE/'receipts/performance04_production_parity.json')
    if (not production or production.get('status')!='PASS'
            or production.get('source_sha256')!=SOURCE_SHA or production.get('SGD_updates')!=0
            or production.get('actual_checkpointing') is not False
            or production.get('actual_memory_format')!='channels_last'
            or production.get('all_BN_buffers_unchanged') is not True
            or len(production.get('records',[]))!=8):
        raise RuntimeError('Actual production no-checkpoint micro1 parity has not passed')
    preflight=read(PREFLIGHT/'preflight.json')
    if not preflight or preflight.get('status')!='PASS':
        state['outcome']='PREFLIGHT_NOT_PASSED'
        return
    if preflight.get('source_sha256')!=SOURCE_SHA:
        raise RuntimeError('Preflight ran a different source snapshot')
    if stage('FORMAL_TRAINING',['train.py','--data-root',str(BASE/'data'),'--identity',str(IDENTITY),
             '--pretrained',str(ASSET),'--preflight',str(PREFLIGHT/'preflight.json'),'--output',str(RUN),
             '--micro-batch','1','--workers','8'],'training_performance04.log') != 0:
        state['outcome']='FORMAL_TRAINING_FAILED'
        return
    result=read(RUN/'result.json') or {}
    if result.get('status')=='PAUSED':
        state['outcome']='FORMAL_TRAINING_PAUSED'
        return
    lock=read(RUN/'endpoint_lock.json')
    if not lock or lock.get('completed')!=40000:
        raise RuntimeError('No complete fixed-40k endpoint')
    if lock.get('test_permitted') is True:
        if stage('LOCKED_TEST',['evaluate.py','--lock',str(RUN/'endpoint_lock.json'),
                 '--data-root',str(BASE/'data'),'--identity',str(IDENTITY),'--pretrained',str(ASSET)],'test_performance04.log') != 0:
            state['outcome']='LOCKED_TEST_FAILED'
            return
        state['outcome']='COMPLETE_WITH_LOCKED_TEST'
    else:
        state['outcome']='COMPLETE_VAL_TARGET_NOT_REACHED'


def save_final_result():
    state.update(status='JOB_FINISHED',finished_at=now())
    state['preflight']=read(PREFLIGHT/'preflight.json')
    state['training_result']=read(RUN/'result.json')
    state['test_result']=read(RUN/'test_result.json')
    if state.get('preflight'):
        # Detailed records remain in their own files; summary need not duplicate
        # all 264 discarded-update gradient records and every crop curve.
        state['preflight']={k:state['preflight'].get(k) for k in ('status','source_sha256','formal_updates','test_accessed','error_type','error')}
    write(BASE/'job_result.json',state)
    write(STATUS,state)
    os.sync()


def shutdown_permitted():
    return (state.get('formal_training_started') is True and state.get('shutdown_armed') is True
            and state.get('outcome') in ('COMPLETE_WITH_LOCKED_TEST','COMPLETE_VAL_TARGET_NOT_REACHED'))


def request_shutdown():
    # Guard the actual provider call as well as its caller. No error path may
    # bypass the user's completion-only shutdown policy.
    if not shutdown_permitted():
        state['shutdown_skipped_reason']='task_not_successfully_completed_after_confirmed_formal_start'
        return
    shutdown_script=BASE/'receipts/provider_shutdown_preserve_files.sh'
    state.update(status='SHUTDOWN_REQUESTED',shutdown_requested_at=now(),shutdown_command=f'/bin/bash {shutdown_script}')
    try:
        write(STATUS,state)
        write(BASE/'receipts/shutdown_request.json',{'requested_at':state['shutdown_requested_at'],
              'outcome':state.get('outcome'),'method':state['shutdown_command'],'results_saved_under':str(BASE)})
        os.sync()
    except BaseException as error:
        try:print('Shutdown receipt persistence failed: '+type(error).__name__+': '+str(error),flush=True)
        except BaseException:pass
    try:log=(BASE/'shutdown.log').open('a')
    except OSError:log=subprocess.DEVNULL
    try:
        for attempt in range(1,4):
            try:
                result=subprocess.run(['/bin/bash',str(shutdown_script)],stdout=log,stderr=subprocess.STDOUT,timeout=90)
                if result.returncode==0:
                    return
            except Exception as error:
                if hasattr(log,'write'):
                    try:log.write(f'Attempt {attempt}: {type(error).__name__}: {error}\n');log.flush()
                    except OSError:pass
            time.sleep(5)
    finally:
        if hasattr(log,'close'):log.close()
    state['status']='SHUTDOWN_COMMAND_FAILED'
    try:write(STATUS,state)
    except BaseException:pass


def finish_and_shutdown():
    try:
        save_final_result()
    except BaseException as error:
        state['finalization_error']=type(error).__name__+': '+str(error)
        state['job_outcome_before_finalization_failure']=state.get('outcome')
        state['outcome']='FINALIZATION_FAILED'
        try:print('Finalization failed: '+state['finalization_error'],flush=True)
        except BaseException:pass
        try:write(BASE/'finalization_failure.json',{'error':state['finalization_error'],'outcome':state.get('outcome')})
        except BaseException:pass
    finally:
        if shutdown_permitted():
            request_shutdown()
        else:
            state.update(status='INSTANCE_PRESERVED',
                         shutdown_skipped_reason='task_not_successfully_completed_after_confirmed_formal_start')
            try:write(STATUS,state)
            except BaseException:pass


if __name__=='__main__':
    from performance04_transition_support import validate_committed_stop
    validate_committed_stop()
    # Background shells may inherit SIGINT=SIG_IGN. Explicitly restore handling.
    signal.signal(signal.SIGINT,signal.default_int_handler)
    os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',
                      TMPDIR=str(BASE/'tmp'),HF_HUB_OFFLINE='1')
    # Failure to acquire the exclusive marker must not shut down another owner.
    marker=(BASE/'controller_performance04_started.json').open('x')
    try:
        with marker as stream:
            json.dump({'pid':os.getpid(),'started_at':now(),'source_sha256':SOURCE_SHA},stream)
            stream.flush();os.fsync(stream.fileno())
        run_job()
    except BaseException as error:
        state.update(outcome='CONTROLLER_FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        finish_and_shutdown()
