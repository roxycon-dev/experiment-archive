"""Use the TRAIN-transfer interval for the required DDR baseline GPU gate."""
import sys,time,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'reports'))
from qlib.common import read_json,sha,assert_source
import tests.gpu_acceptance as gate
source=assert_source(ROOT)['sha256'];runtime_sha=sha(ROOT/'configs/runtime.server.json')
def reuse_early():
    receipt=read_json(ROOT/'reports/early_gpu_gates.json')
    assert receipt['status']=='PASS' and receipt['source_sha256']==source and receipt['runtime_sha256']==runtime_sha
gate.early=reuse_early
cache=gate.cache_sources
def wait_for_train_then_cache(*args,**kwargs):
    path=ROOT/'reports/parallel_data_ready.json'
    while not path.exists():time.sleep(2)
    assert read_json(path)['stage']=='ALL896_TRAIN_FILES_SHA_VERIFIED'
    return cache(*args,**kwargs)
gate.cache_sources=wait_for_train_then_cache
import parallel_sd_worker as worker
# The original worker waits before acceptance.main. Only this redundant wait
# is bypassed here; acceptance's cache wrapper still requires true TRAIN SHA.
original_acceptance=gate.main
def overlap_main():return original_acceptance()
# Explicitly resume the same worker logic, with a wait-free acceptance entry:
def main():
    import os,torch,cv2,gc,traceback
    from qlib.common import write_json,utc_now
    torch.set_num_threads(1);cv2.setNumThreads(1)
    assert not (ROOT/'reports/driver.lock').exists();assert not (ROOT/'runs/SD_seed42/identity.json').exists()
    lock=ROOT/'reports/driver.lock';write_json(lock,{'pid':os.getpid(),'scope':'SD_seed42_only','time':utc_now()})
    try:
        p=ROOT/'vendor/msda_build/MultiScaleDeformableAttention.so';assert sha(p)==read_json(ROOT/'reports/parallel_kernel_binary.json')['sha256']
        spec=importlib.util.spec_from_file_location('MultiScaleDeformableAttention',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);sys.modules['MultiScaleDeformableAttention']=m
        write_json(ROOT/'reports/driver_state.json',{'stage':'SECOND_NODE_GPU_ACCEPTANCE','pid':os.getpid(),'overlaps_TRAIN_transfer':True,'time':utc_now()})
        reuse_early();original_acceptance()
        from qlib.data import DDR
        from experiment import val_dataset
        from inputs import TJDR,cache_sources
        from validation import cache_p0
        from train_run import run
        runtime_path=ROOT/'configs/runtime.server.json';runtime=read_json(runtime_path)
        datasets={'DDR':DDR(runtime['ddr_root'],'train',read_json(ROOT/'reports/data_identity.json')['splits']['train']),'TJDR':TJDR(runtime['TJDR_root'],read_json(ROOT/'reports/TJDR_manifest.json'))}
        cached,cache_receipt=cache_sources(datasets,runtime['resources']['shared_cache_limit_bytes']);val=val_dataset(runtime);p0,p0_receipt=cache_p0(val,runtime)
        write_json(ROOT/'reports/campaign.json',{'stage':'RUNNING','current_run':'SD_seed42','queue':[['SD',42]],'amendment_sha256':sha(ROOT/'reports/single_seed_parallel_amendment.json'),'time':utc_now()})
        write_json(ROOT/'reports/driver_state.json',{'stage':'PARALLEL_CAMPAIGN','pid':os.getpid(),'time':utc_now()})
        state=run(runtime_path,'SD',42,shared=(cached,cache_receipt,val,p0,p0_receipt));stage='PARALLEL_ASSIGNED_RUNS_COMPLETE_WAITING_FOR_GLOBAL' if state=='COMPLETE' else state
        write_json(ROOT/'reports/campaign.json',{'stage':stage,'current_run':'SD_seed42','queue':[['SD',42]],'time':utc_now()});write_json(ROOT/'reports/driver_state.json',{'stage':stage,'pid':os.getpid(),'time':utc_now()})
    except BaseException as e:
        write_json(ROOT/'reports/driver_state.json',{'stage':'TECHNICAL_PAUSE','reason':str(e),'automatic_restart':False,'pid':os.getpid(),'time':utc_now()});traceback.print_exc();raise
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
