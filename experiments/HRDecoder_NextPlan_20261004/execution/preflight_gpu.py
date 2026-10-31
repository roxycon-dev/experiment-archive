"""Only experiment-required initialization, real update timing and full val acceptance."""
import argparse
import gc
import json
import time
import numpy as np
import torch
import adaptation
from qlib.common import assert_source,read_json,sha,utc_now,write_json
from qlib.evaluate import evaluate
from train import ROOT,initialize,fingerprint,datasets_for,rows_for,loader_for,update


def _measure(runtime,arm,resources):
    model,loading=initialize(runtime,arm,42)
    model=model.to('cuda').train()
    optimizer,names=adaptation.optimizer_for(model)
    scheduler=adaptation.Schedule(optimizer)
    datasets,val=datasets_for(runtime,arm)
    rows,_=rows_for(arm,42)
    loader=loader_for(datasets,rows,0,110,resources,42)
    torch.cuda.reset_peak_memory_stats()
    timing=[];valid_audit={}
    for u in range(110):
        statistics=update(model,optimizer,scheduler,loader,resources,u,valid_audit,rows)
        if u>=10:
            timing.append(statistics)
        if u==10:
            missing=[]
            for name,p in model.named_parameters():
                if p.requires_grad and p.grad is None:
                    missing.append(name)
            if missing:
                raise ValueError('Trainable parameters absent from forward/backward: '+repr(missing))
        if u%20==19:
            print('preflight',arm,u+1,flush=True)
    probe_runtime=dict(runtime,resources=resources)
    full=evaluate(model,val,probe_runtime,ROOT/f'reports/preflight/{arm}/val_after_unregistered110',110)
    reference=read_json(ROOT/'historical/Q_DDR_seed42/metrics.json')
    assert full['p0']['counts']==reference['p0']['counts'], 'P0 or validation label contract changed'
    old_rows={r['sid']:r['p0'] for r in reference['per_image']}
    assert all(r['p0']==old_rows[r['sid']] for r in full['per_image'])
    result={'status':'PASS','arm':arm,'resources':resources,'steady_real_effective_updates':100,
            'warmup_measurement_updates':10,'seconds_per_update_mean':float(np.mean([r['seconds_per_update'] for r in timing])),
            'seconds_per_update_p90':float(np.quantile([r['seconds_per_update'] for r in timing],.9)),
            'data_wait_seconds_mean':float(np.mean([r['data_wait_seconds'] for r in timing])),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'full76val_seconds':full['seconds'],
            'timing_rows':timing,'trainable_groups':names,'formal_training_started':False,
            'not_a_candidate_or_selection_node':True,'initial_weights':loading}
    write_json(ROOT/f'reports/preflight/{arm}.json',result)
    del loader,model,optimizer,datasets,val
    gc.collect();torch.cuda.empty_cache()
    return result


def measure(runtime,arm,resources):
    try:
        return _measure(runtime,arm,resources)
    except torch.cuda.OutOfMemoryError as error:
        # Return a plain record, releasing exception frames/tensors before another scheme.
        return {'status':'OOM','arm':arm,'reason':str(error)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',default='configs/runtime.server.json')
    args=ap.parse_args();runtime=read_json(args.runtime);source=assert_source(ROOT)
    cpu=read_json(ROOT/'reports/cpu_acceptance.json');assert cpu['status']=='PASS'
    old=read_json(ROOT/'reports/historical_implementation.json')
    for rel in ('qlib/model.py','qlib/data.py','qlib/geometry.py','qlib/loss.py','qlib/optim.py','qlib/common.py'):
        assert sha(ROOT/rel)==old[rel]
    for rel,h in old.items():
        if rel.startswith('vendor/dinov3/'):
            assert sha(__import__('pathlib').Path(runtime['old_root'])/rel)==h, 'Official backbone source changed'
    original_resources=read_json(ROOT/'historical/Q_MIX_seed42/identity.json')['resources']
    assert runtime['resources']==original_resources
    manifest=read_json(runtime['external_manifest'])
    changed=read_json(ROOT/'reports/negative_diff.server.json')['changed_entries']
    inputs={(r['source'],r['sid']):r for r in manifest['samples']}
    for item in changed:
        row=inputs['IDRiD',item['sid']]
        assert sha(row['image'])==item['image_sha256']
        assert sha(row['classes'][item['class']]['mask'])==item['mask_sha256']
    fingerprints={};rng_receipts={};platform_comparison={}
    for seed in (42,2024):
        a,la=initialize(runtime,'E_NEG',seed);fa=fingerprint(a);del a
        b,lb=initialize(runtime,'E_NEG',seed);fb=fingerprint(b);del b
        reconstructed_rng=torch.get_rng_state().clone()
        assert fa==fb
        assert la['sha256']==read_json(ROOT/f'historical/Q_MIX_seed{seed}/initialization.json')['sha256']
        local=read_json(ROOT/'reports/local_initialization_receipt.json')
        platform_comparison[str(seed)]={'local_tensor_sha256':local['initialization_sha256_by_seed'][str(seed)],
             'server_tensor_sha256':fa,'byte_equal_across_torch_versions':fa==local['initialization_sha256_by_seed'][str(seed)],
             'cross_platform_equality_is_not_the_historical_server_pairing_gate':True}
        # Main-process CPU RNG remains at post-Q-initialization in the historical
        # run: augmentation is precomputed, DataLoader uses its own generator,
        # and model stochastic operations occur on CUDA. Verify against the
        # actual archived24k state, rather than a different local torch version.
        old_checkpoint=__import__('pathlib').Path(runtime['old_root'])/f'runs/Q_MIX_seed{seed}/checkpoint_u24000.pth'
        archived=torch.load(old_checkpoint,map_location='cpu',weights_only=False)
        assert archived['completed']==24000
        assert archived['identity']==read_json(ROOT/f'historical/Q_MIX_seed{seed}/identity.json')
        assert torch.equal(reconstructed_rng,archived['rng']['torch'].cpu()), 'Historical initializer RNG consumption differs'
        rng_receipts[str(seed)]={'post_initialization_CPU_RNG_matches_archived_checkpoint':True,
                                'historical_checkpoint_sha256':sha(old_checkpoint)}
        del archived;gc.collect()
        fingerprints[str(seed)]=fa
    paired={'status':'PASS','initialization_sha256_by_seed':fingerprints,
            'original_pretrained_and_head_initialization_reconstructed_from_verified_source':True,
            'historical_archived_u0_tensor_hash_available':False,
            'archived_rng_identity_by_seed':rng_receipts,'platform_comparison':platform_comparison,
            'server_torch_version':torch.__version__,'historical_threads':{'OMP_NUM_THREADS':4,'MKL_NUM_THREADS':4},
            'stream_and_augmentation_byte_identity':True,'micro_batch_accumulation_unchanged':True,
            'model_geometry_loss_optimizer_schedule_implementation_byte_identity':True,
            'same_original_seed_order':'Historical train_run seed_all precedes deterministic dataset construction and Q factory; those constructors consume no RNG',
            'proof_limit':'Initialization is deterministically reconstructed from archived code/seed/pretraining, not compared to an archived u0 tensor file'}
    measurements={}
    resources=dict(runtime['f_resources'])
    while True:
        measurements={}
        for arm in ('F4','F12'):
            measurements[arm]=measure(runtime,arm,resources)
            if measurements[arm]['status']=='OOM':break
        if all(m['status']=='PASS' for m in measurements.values()) and len(measurements)==2:
            break
        write_json(ROOT/'reports/preflight/micro2_oom.json',dict(measurements[arm],
                   formal_training_started=False,time=utc_now()))
        gc.collect();torch.cuda.empty_cache()
        if resources['micro_batch']!=2:
            raise RuntimeError('F micro1x8 also OOM; technical pause, no further unregistered fallback')
        resources.update(micro_batch=1,accumulation=8)
        runtime['f_resources']=resources
        runtime['f_micro_dice_semantics_changed']=True
        write_json(args.runtime,runtime)
    # E_NEG always retains historical micro2x4. F arms share the one accepted scheme.
    historical_update=.460820545963943
    f_hours=sum(m['seconds_per_update_mean']*12000+m['full76val_seconds']*4 for m in measurements.values())/3600
    estimate={'discovery_hours_measured_F_plus_historical_E_estimate':1.2*(f_hours+(24000*historical_update+7*207.498169776)/3600)+.5,
              'all_promoted_hours_estimate':1.2*(2*f_hours+2*(24000*historical_update+7*207.498169776)/3600)+1.,
              'E_NEG_seconds_is_historical_estimate_not_new_measurement':True,
              'F12_seconds_is_new_real_measurement':True,'hourly_rate_cny':runtime['budget']['hourly_rate_cny']}
    if estimate['hourly_rate_cny'] is not None:
        estimate['discovery_cny']=estimate['discovery_hours_measured_F_plus_historical_E_estimate']*estimate['hourly_rate_cny']
        estimate['all_promoted_cny']=estimate['all_promoted_hours_estimate']*estimate['hourly_rate_cny']
    gate={'status':'READY','runtime_sha256':sha(args.runtime),'source_sha256':source['sha256'],
          'E_NEG_paired_acceptance':paired,'F_resources':resources,'measurements':measurements,
          'estimate':estimate,'created_at':utc_now(),'test_read':False}
    write_json(ROOT/'reports/gpu_readiness.json',gate)
    print(json.dumps({'status':'READY','F_resources':resources,'estimate':estimate}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        write_json(ROOT/'reports/preflight_failure.json',{'stage':'TECHNICAL_PAUSE','reason':str(error),
                   'automatic_restart':False,'time':utc_now()})
        raise
