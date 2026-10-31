"""Diagnosed loader omission: reuse passed C/system/parent; complete S/D/SD exactly."""
import sys,os,gc,time,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,cv2
torch.set_num_threads(1);cv2.setNumThreads(1)
from qlib.common import read_json,write_json,sha,utc_now,assert_source
from tests.gpu_acceptance import (DDR,TJDR,cache_sources,val_dataset,cache_p0,initialize,optimizer_for,Schedule,
    EMA,loader_for,rows_for,update,evaluate,save,load_kernel)

def main():
    runtime_path=ROOT/'configs/runtime.server.json';runtime=read_json(runtime_path);source=assert_source(ROOT)
    assert not list((ROOT/'runs').glob('*/identity.json')) and not (ROOT/'reports/campaign.json').exists()
    state=read_json(ROOT/'reports/driver_state.json')
    assert state['stage']=='TECHNICAL_PAUSE' and state['reason']=="No module named 'MultiScaleDeformableAttention'"
    assert not (ROOT/'reports/driver.lock').exists()
    gate=read_json(ROOT/'reports/gpu_readiness.json')
    assert gate['status']=='RUNNING' and set(gate['benchmarks'])=={'C'} and gate['benchmarks']['C']['complete_updates']==100
    assert gate['source_sha256']==source['sha256'] and gate['runtime_sha256']==sha(runtime_path)
    early=read_json(ROOT/'reports/early_gpu_gates.json');assert early['status']=='PASS' and early['source_sha256']==source['sha256']
    system=read_json(ROOT/'reports/system_acceptance.json')
    assert system['status']=='PASS' and all(system['R_val'][str(s)]['all76_per_image_counts_equal'] for s in (42,2024))
    assert read_json(ROOT/'reports/parent_validation_seed42.json')['single']['counts']==read_json(Path(runtime['parent_f12_execution'])/'runs/F12_seed42/val_u12000/metrics.json')['single']['counts']
    archive=ROOT/'reports/preparation_repairs/acceptance_kernel_loader_20261005';archive.mkdir(parents=True,exist_ok=True)
    for rel in ('driver_state.json','gpu_readiness.json','server_driver.log'):
        shutil.copy2(ROOT/'reports'/rel,archive/('original_'+rel))
    revision={'stage':'EXPLICIT_REMAINING_GPU_ACCEPTANCE','time':utc_now(),'pid':os.getpid(),'source_sha256':source['sha256'],
        'operational_helper_sha256':sha(Path(__file__)),'cause':'Early gates ran in another process; full benchmark reused their receipt without importing the compiled extension into this process.',
        'repair':'Call the already verified kernel loader before native S/SD benchmarking. Main train_run already does this.',
        'already_passed_not_repeated':['both76-image historical R controls','native parent76-image counts','all early kernel/model/state/routing gates','C100updates/fullval/checkpoint timing'],
        'remaining_arms':['S','D','SD'],'formal_updates_lost':0,'frozen_source_and_scientific_settings_changed':False}
    write_json(ROOT/'reports/acceptance_kernel_loader_revision.json',revision)
    write_json(ROOT/'reports/driver_state.json',{'stage':'GPU_ACCEPTANCE_REMAINING','pid':os.getpid(),'time':utc_now(),'revision':'reports/acceptance_kernel_loader_revision.json'})
    load_kernel()
    datasets={'DDR':DDR(runtime['ddr_root'],'train',read_json(ROOT/'reports/data_identity.json')['splits']['train']),
              'TJDR':TJDR(runtime['TJDR_root'],read_json(ROOT/'reports/TJDR_manifest.json'))}
    cached,cache_receipt=cache_sources(datasets,runtime['resources']['shared_cache_limit_bytes'],lambda p:print('cache',p,flush=True))
    assert cache_receipt['bytes']==gate['caches']['training']['bytes'] and cache_receipt['images']==893
    val=val_dataset(runtime);p0,p0_receipt=cache_p0(val,runtime)
    assert p0_receipt['bytes']==gate['caches']['P0']['bytes']
    for arm in ('S','D','SD'):
        model,_=initialize(runtime,arm,42);model.train();opt,_=optimizer_for(model);sched=Schedule(opt);ema=EMA(model)
        loaders={'DDR':loader_for(cached,rows_for('DDR',42),'DDR',0,100,runtime['resources'],42)}
        if model.has_source:loaders['TJDR']=loader_for(cached,rows_for('TJDR',42),'TJDR',0,100,runtime['resources'],42)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();wait=0
        for u in range(1,101):
            stat=update(model,opt,sched,ema,loaders,42,u-1);wait+=stat['data_wait_seconds']
            if u%20==0:print('acceptance',arm,u,stat,flush=True)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        timing=evaluate(ema.model,val,runtime,ROOT/f'reports/timing_{arm}_val',100,p0_ram=p0)
        assert timing['n_images']==76
        identity={'arm':arm,'acceptance_only':True};path=ROOT/f'reports/timing_{arm}_checkpoint.pth';before=time.perf_counter()
        save(path,model,opt,sched,ema,100,identity,{});save_seconds=time.perf_counter()-before;checkpoint_bytes=path.stat().st_size;path.unlink()
        gate['benchmarks'][arm]={'complete_updates':100,'seconds_per_update':elapsed/100,'data_wait_seconds_per_update':wait/100,
            'full_val_seconds':timing['seconds'],'checkpoint_seconds':save_seconds,'checkpoint_bytes':checkpoint_bytes,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        write_json(ROOT/'reports/gpu_readiness.json',gate);print(arm,gate['benchmarks'][arm],flush=True)
        del model,opt,sched,ema,loaders;gc.collect();torch.cuda.empty_cache()
    assert set(gate['benchmarks'])=={'C','S','D','SD'} and assert_source(ROOT)['sha256']==source['sha256']
    gate.update(status='READY',completed_at=utc_now(),benchmarks_used_for_scientific_initialization=False,formal_CUDA_settings_restored=True,
                diagnosed_acceptance_dispatch_recovery=revision)
    write_json(ROOT/'reports/gpu_readiness.json',gate)
    del cached,datasets,val,p0;gc.collect();torch.cuda.empty_cache()
    write_json(ROOT/'reports/acceptance_kernel_loader_recovery.json',{'stage':'ALL_REQUIRED_GPU_ACCEPTANCE_COMPLETE','time':utc_now(),'pid':os.getpid()})
    from server_driver import main as driver
    driver(resume_paused=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(ROOT/'reports/driver_state.json',{'stage':'TECHNICAL_PAUSE','reason':str(e),'automatic_restart':False,'updated_at':utc_now()})
        traceback.print_exc();raise
