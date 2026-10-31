"""Nine bounded runs, shared useful RAM cache; explicit identity-checked resume."""
import argparse,gc,os,json,time,traceback
from pathlib import Path
import torch
from qlib.common import read_json,write_json,utc_now,assert_source
from qlib.data import DDR
from experiment import ROOT,val_dataset
from inputs import TJDR,cache_sources
from validation import cache_p0
from train_run import run as train
from parent_f12 import run as parent
from decision import finalize,write_report
from storage import restore_selection
QUEUE=[(arm,42) for arm in ('C','S','D','SD')]+[('PARENT_F12',2024)]+[(arm,2024) for arm in ('C','S','D','SD')]

def main(resume_paused=False):
    runtime_path=ROOT/'configs/runtime.server.json';runtime=read_json(runtime_path);assert_source(ROOT)
    report=ROOT/'reports/campaign.json';lock=ROOT/'reports/campaign.lock'
    if report.exists():
        previous=read_json(report)
        if previous['stage']=='READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN':return
        assert resume_paused,'Existing campaign requires explicit --resume-paused after diagnosis'
    with lock.open('x',encoding='utf-8') as f:json.dump({'pid':os.getpid(),'time':utc_now()},f)
    current=None
    try:
        assert read_json(ROOT/'reports/gpu_readiness.json')['status']=='READY'
        datasets={'DDR':DDR(runtime['ddr_root'],'train',read_json(ROOT/'reports/data_identity.json')['splits']['train']),
                  'TJDR':TJDR(runtime['TJDR_root'],read_json(ROOT/'reports/TJDR_manifest.json'))}
        write_json(report,{'stage':'WARMING_SHARED_CACHE','queue':QUEUE,'time':utc_now()})
        cached,receipt=cache_sources(datasets,runtime['resources']['shared_cache_limit_bytes'],lambda p:print('cache',p,flush=True))
        val=val_dataset(runtime);p0,p0_receipt=cache_p0(val,runtime);shared=(cached,receipt,val,p0,p0_receipt)
        write_json(ROOT/'reports/operational_cache.json',{'training':receipt,'P0':p0_receipt,'shared_across_all_runs':True})
        for arm,seed in QUEUE:
            current=f'{arm}_seed{seed}';runpath=ROOT/'runs'/current
            status=read_json(runpath/'status.json') if (runpath/'status.json').exists() else None
            if status and status['stage'] in ('COMPLETE','SCIENTIFIC_STOP_INCOMPLETE'):
                if arm!='PARENT_F12' or (ROOT/'reports/seed2024_architecture_acceptance.json').exists():continue
            if status and status['stage']=='CURVE_REVIEW_REQUIRED':
                write_json(report,{'stage':'CURVE_REVIEW_REQUIRED','current_run':current,'updated_at':utc_now()});return
            write_json(report,{'stage':'RUNNING','current_run':current,'queue':QUEUE,'updated_at':utc_now()})
            if arm=='PARENT_F12':
                if not status or status['stage']!='COMPLETE':parent(runtime_path,shared,resume=bool(status))
                # The second parent was fixed before any four-arm second-seed run.
                from tests.gpu_acceptance import small_model_gates
                settings=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
                try:
                    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                    acceptance=small_model_gates(runtime,seed=2024)
                    write_json(ROOT/'reports/seed2024_architecture_acceptance.json',{'status':'PASS','tests':acceptance,'time':utc_now()})
                finally:torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=settings
            else:
                if seed==2024:assert read_json(ROOT/'reports/seed2024_architecture_acceptance.json')['status']=='PASS'
                state=train(runtime_path,arm,seed,resume=bool(status),shared=shared)
                if state!='COMPLETE':
                    write_json(report,{'stage':state,'current_run':current,'updated_at':utc_now()});return
            gc.collect();torch.cuda.empty_cache()
        # Training caches are no longer needed during legacy P0 reconstruction.
        # Release useful RAM before the optional fixed test, instead of keeping
        # it allocated just to make the utilization meter read high.
        del shared,cached,datasets,val,p0
        gc.collect();torch.cuda.empty_cache()
        result=finalize();test=None
        if result['test_authorized']:
            write_json(report,{'stage':'WAITING_FOR_SELECTED_CHECKPOINTS','updated_at':utc_now(),'final_selection':result['final_selection']})
            restore_selection(read_json(ROOT/'reports/final_selection_lock.json'))
            from final_test import run as test_run
            test=test_run(ROOT/'reports/final_selection_lock.json',runtime_path)
        write_report(result,test)
        write_json(report,{'stage':'READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN','queue':QUEUE,'final_selection':result['final_selection'],
            'test_authorized':result['test_authorized'],'test_complete':test is not None,'updated_at':utc_now()})
    except BaseException as e:
        write_json(report,{'stage':'TECHNICAL_PAUSE','current_run':current,'reason':f'{type(e).__name__}: {e}',
            'automatic_restart':False,'updated_at':utc_now()});traceback.print_exc();raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-paused',action='store_true');a=p.parse_args();main(a.resume_paused)
