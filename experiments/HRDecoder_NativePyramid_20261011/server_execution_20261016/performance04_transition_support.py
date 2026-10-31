"""Read-only transition guards. No training, process signals or shutdown."""
import hashlib,json,math,sys
from pathlib import Path
BASE=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
OLD_SOURCE='d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627'
NEW_SOURCE='14e47c3749db5dd28c2938ddbb227aad66b07c40b4fb9d049fdb0564b91c95d0'

def require(condition,message):
    if not condition:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_text())

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def process(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':fields[0],'ppid':int(fields[1]),'pgrp':int(fields[2]),'starttime':int(fields[19])}
    except FileNotFoundError:return None

def alive(identity):
    current=process(identity['pid'])
    return bool(current and current['starttime']==identity['starttime'] and current['state']!='Z')

def active_group(group):
    members=[]
    for path in Path('/proc').iterdir():
        if path.name.isdecimal():
            try:item=process(int(path.name))
            except (OSError,ValueError):continue
            if item and item['pgrp']==group and item['state']!='Z':members.append(item)
    return members

def validate_candidate():
    code=BASE/'code_performance04_candidate';sys.path.insert(0,str(code))
    import train
    require(train.HERE.resolve()==code.resolve(),'Imported a different training module')
    train.configure_runtime()
    digest,_=train.source_identity()
    require(digest==NEW_SOURCE,'Candidate source bytes changed')
    preflight=BASE/'preflight_performance04/preflight.json'
    train.validate_preflight(read(preflight),digest,sha(code/'preparation_20261016/data_identity.json'),
                             train.runtime_identity(),micro_batch=1,workers=8)
    path=BASE/'receipts/performance04_production_parity.json';production=read(path)
    require(production.get('status')=='PASS' and production.get('source_sha256')==NEW_SOURCE
            and production.get('SGD_updates')==0 and production.get('test_accessed') is False
            and production.get('actual_checkpointing') is False
            and production.get('actual_memory_format')=='channels_last'
            and production.get('all_BN_buffers_unchanged') is True,'Actual production parity did not pass')
    records=production.get('records',[])
    expected={(fixture,amp,micro) for fixture in ('zero_residual','nonzero_residual')
              for amp in (False,True) for micro in (1,2)}
    require(len(records)==8 and {(r.get('fixture'),r.get('BF16'),r.get('loss_batch')) for r in records}==expected,
            'Production parity case coverage changed')
    for r in records:
        tolerance={'rtol':.03,'atol':.0005} if r['BF16'] else {'rtol':.0003,'atol':.000003}
        require(r.get('checkpointing') is False and r.get('physical_forward_batch')==1
                and r.get('backward_calls')==2//r['loss_batch']
                and r.get('all_parameter_gradients_compared') is True
                and r.get('tolerance')==tolerance and math.isfinite(r.get('loss',float('nan'))),
                'Production parity record/tolerance changed')
    return {'new_source_sha256':digest,'new_preflight_sha256':sha(preflight),'production_parity_sha256':sha(path)}

def validate_committed_stop():
    r=read(BASE/'receipts/stopped03_for_approved_performance04.json')
    transaction=Path(r.get('transaction_path','')).resolve()
    require(transaction.is_relative_to((BASE/'receipts').resolve())
            and transaction.name.startswith('approved_stop_transaction_'),'Invalid stop transaction path')
    require(read(transaction)==r and read(BASE/'run_seed42_launch02/abandoned_for_approved_performance04.json')==r,
            'Stop transaction or old-prefix mirror is incomplete/inconsistent')
    require(r.get('status')=='STOP_CONFIRMED' and r.get('human_approved_new_trial') is True
            and r.get('source_sha256')==OLD_SOURCE and r.get('new_source_sha256')==NEW_SOURCE
            and r.get('complete_ledger_validated') is True and r.get('executed_updates',0)>0
            and r.get('in_flight_update') is None and r.get('additional_executed_updates_uncertain')==[0,0],
            'No committed, user-approved old-run stop receipt')
    require(not alive(r['old_training_identity']) and not alive(r['old_controller_identity'])
            and not active_group(r['old_training_identity']['pgrp']),'An old controller or training process remains alive')
    evidence=validate_candidate()
    require(all(r.get(k)==v for k,v in evidence.items()),'Candidate evidence changed after approved stop')
    return r
