"""Assigned SD42 only; identical frozen science, node-local functional gates."""
import sys,os,json,time,traceback,gc,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,cv2
from qlib.common import read_json,write_json,sha,utc_now,assert_source

def main():
    torch.set_num_threads(1);cv2.setNumThreads(1)
    assert read_json(ROOT/'reports/single_seed_parallel_amendment.json')['seeds']==[42]
    assert not (ROOT/'reports/driver.lock').exists()
    lock=ROOT/'reports/driver.lock';write_json(lock,{'pid':os.getpid(),'time':utc_now(),'scope':'SD_seed42_only'})
    try:
        source=assert_source(ROOT);receipt=read_json(ROOT/'reports/parallel_kernel_binary.json')
        path=ROOT/'vendor/msda_build/MultiScaleDeformableAttention.so';assert sha(path)==receipt['sha256']
        assert torch.__version__=='2.7.1+cu128'
        spec=importlib.util.spec_from_file_location('MultiScaleDeformableAttention',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules['MultiScaleDeformableAttention']=module
        write_json(ROOT/'reports/driver_state.json',{'stage':'SECOND_NODE_GPU_ACCEPTANCE','pid':os.getpid(),'time':utc_now()})
        from tests.gpu_acceptance import early,main as acceptance
        early()
        while not (ROOT/'reports/parallel_data_ready.json').exists():time.sleep(2)
        assert read_json(ROOT/'reports/parallel_data_ready.json')['stage']=='ALL896_TRAIN_FILES_SHA_VERIFIED'
        acceptance()
        from qlib.data import DDR
        from experiment import val_dataset
        from inputs import TJDR,cache_sources
        from validation import cache_p0
        from train_run import run
        runtime_path=ROOT/'configs/runtime.server.json';runtime=read_json(runtime_path)
        datasets={'DDR':DDR(runtime['ddr_root'],'train',read_json(ROOT/'reports/data_identity.json')['splits']['train']),
                  'TJDR':TJDR(runtime['TJDR_root'],read_json(ROOT/'reports/TJDR_manifest.json'))}
        cached,cache_receipt=cache_sources(datasets,runtime['resources']['shared_cache_limit_bytes'])
        val=val_dataset(runtime);p0,p0_receipt=cache_p0(val,runtime)
        write_json(ROOT/'reports/campaign.json',{'stage':'RUNNING','current_run':'SD_seed42','queue':[['SD',42]],'amendment_sha256':sha(ROOT/'reports/single_seed_parallel_amendment.json'),'time':utc_now()})
        write_json(ROOT/'reports/driver_state.json',{'stage':'PARALLEL_CAMPAIGN','pid':os.getpid(),'time':utc_now()})
        state=run(runtime_path,'SD',42,shared=(cached,cache_receipt,val,p0,p0_receipt))
        stage='PARALLEL_ASSIGNED_RUNS_COMPLETE_WAITING_FOR_GLOBAL' if state=='COMPLETE' else state
        write_json(ROOT/'reports/campaign.json',{'stage':stage,'current_run':'SD_seed42','queue':[['SD',42]],'time':utc_now()})
        write_json(ROOT/'reports/driver_state.json',{'stage':stage,'pid':os.getpid(),'time':utc_now()})
    except BaseException as e:
        write_json(ROOT/'reports/driver_state.json',{'stage':'TECHNICAL_PAUSE','reason':str(e),'automatic_restart':False,'pid':os.getpid(),'time':utc_now()});traceback.print_exc();raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
