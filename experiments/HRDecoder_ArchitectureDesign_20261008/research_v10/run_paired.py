"""Two bounded private-only arms on the same real CUDA batches; no fresh/test run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import torch
from torch.utils.data import DataLoader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'residual_execution'))
sys.path.insert(0,str(ROOT/'normalization_execution'))
import train as core
from fast_objectives import residual_gt,trusted_kl
from objectives import fixed_teacher
from intervention import balanced_kl
from diagnostic_invariants import (FrozenProviderStudent,named_private_optimizer_state,
    restore_immutable_parent,assert_frozen_provider,provider_snapshot,assert_provider_unchanged)
from loader import worker_setup
from deterministic_resize import install as install_deterministic_resize

WORK=ROOT/'training_runs/diagnostic_v10'
REPORTS=ROOT/'reports/diagnostic_v10'
ARMS=('control','gt_stratified')


def write(path,value):core.write_json(path,value)


def run():
    WORK.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    assert not (WORK/'PROBE_STARTED.json').exists(), 'No implicit repeated diagnostic'
    assert torch.__version__=='2.0.1+cu117'
    cpu=json.loads((HERE/'DIAGNOSTIC_INVARIANTS_CPU_PASS.json').read_text())
    assert cpu['status']=='CPU_INVARIANT_PASS_GPU_PENDING' and cpu['torch']==torch.__version__
    for name,digest in cpu['source_SHA256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    plan=json.loads((HERE/'probe_protocol.json').read_text())
    assert plan['warm_updates_per_arm']==500 and plan['canvas']==2048 and plan['batch']==2
    with open(ROOT/'reports/gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        import subprocess
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader'],universal_newlines=True).strip(), 'Foreign GPU process; no interference'
        parent_path=ROOT/'training_runs/residual_normalization/parent_08000.pt'
        parent=torch.load(parent_path,map_location='cpu')
        material,parent_digest=core.identity()
        assert parent_digest==parent['identity_SHA']==plan['parent_identity_SHA']
        assert parent['completed']==8000 and parent['optimizer']['state'] and 'rng' in parent
        core.install_legacy_bf16_interpolation();torch.set_num_threads(4)
        resize=json.loads((HERE/'RESIZE_CUDA_PASS.json').read_text())
        assert resize['status']=='PASS' and resize['torch']==torch.__version__
        install_deterministic_resize()
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        ready=core.install_ram(ROOT)
        from data import TrainCacheDataset
        dataset=TrainCacheDataset(ROOT,core.DATA)
        sampler=core.ResumableBatches(len(dataset),8000,8003)
        loader=core.CUDAPrefetch(dataset,sampler,workers=12)
        # Spawn avoids fork-after-CUDA/native-library mutex inheritance.
        loader.loader=DataLoader(dataset,batch_sampler=sampler,num_workers=12,prefetch_factor=1,
            persistent_workers=True,pin_memory=True,worker_init_fn=worker_setup,
            generator=torch.Generator().manual_seed(42),multiprocessing_context='spawn')
        models={};optimizers={};filtered={};snapshots={}
        for arm in ARMS:
            baseline,_=core.build_baseline(core.REPO,core.W48,verified_weight_sha=material['ImageNet_W48_SHA'])
            model=FrozenProviderStudent(baseline).cuda().train()
            optimizer=torch.optim.SGD(model.diagnostic_parameters(),lr=.005,momentum=.9,
                weight_decay=.0005,foreach=True)
            names=[name for name,_ in model.named_parameters()]
            state=named_private_optimizer_state(parent['optimizer'],names,model)
            models[arm],optimizers[arm],filtered[arm]=model,optimizer,state
            restore_immutable_parent(model,optimizer,parent['model'],state)
            snapshots[arm]=provider_snapshot(model)
        identity={'kind':'frozen-provider warm8k+500, not fresh40k','parent_identity_SHA':parent_digest,
            'protocol':plan,'torch':torch.__version__,
            'code':{name:core.file_sha(HERE/name) for name in
                    ('run_paired.py','diagnostic_invariants.py','deterministic_resize.py','probe_protocol.json')},
            'benchmark':False,'cudnn_deterministic':True,'deterministic_algorithms':True,
            'TF32':False,'precision':'BF16 network, FP32 residual and losses'}
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        write(WORK/'protocol.json',{'identity_SHA':digest,'identity':identity,
             'same_model_private_optimizer_RNG_parent':True,'RAM_pid':ready['pid']})
        deadline=time.time()+3600

        def reload_parent():
            for arm in ARMS:
                restore_immutable_parent(models[arm],optimizers[arm],parent['model'],filtered[arm])
                models[arm].train();assert_frozen_provider(models[arm],optimizers[arm])
            core.restore_rng(parent['rng'])

        def paired_step(batch,index):
            assert tuple(batch['image'].shape)==(2,3,2048,2048)
            originals={name:batch[name].clone() for name in ('image','label','valid','A','U','S')}
            shared_rng=core.rng_state();base_reference=None;metrics={}
            target=fixed_teacher(batch['A'],batch['U'],batch['S'],batch['label'])
            for arm in ARMS:
                model,optimizer=models[arm],optimizers[arm]
                assert_frozen_provider(model,optimizer);core.restore_rng(shared_rng)
                for group in optimizer.param_groups:group['lr']=core.learning_rate(index)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model(batch['image'],batch['label'],batch['valid'])
                if base_reference is None:base_reference=output['base_logits'].detach().clone()
                else:assert torch.equal(base_reference,output['base_logits']), 'Actual provider outputs diverged'
                gt=residual_gt(output['logits'],batch['label'],batch['valid'])
                kl=(trusted_kl if arm=='control' else balanced_kl)(output['logits'],target,batch['label'],batch['valid'])
                loss=gt+core.lambda_k(index+1)*kl
                assert bool(torch.isfinite(loss));loss.backward()
                gradients=[p.grad for p in model.private.parameters() if p.grad is not None]
                assert gradients and all(g.dtype==torch.float32 for g in gradients)
                found=torch.zeros(1,device='cuda');inv=torch.ones(1,device='cuda')
                torch._amp_foreach_non_finite_check_and_unscale_(gradients,found,inv)
                assert found.item()==0 and all(p.grad is None for p in model.base.parameters())
                nonzero={name:any(p.grad is not None and bool(p.grad.ne(0).any()) for p in part.parameters())
                    for name,part in [('local',model.private.local),('global',model.private.global_path),('head',model.private.head)]}
                assert all(nonzero.values());optimizer.step()
                metrics[arm]={'gt_final':float(gt.detach()),'KL':float(kl.detach()),'loss':float(loss.detach()),'private_nonzero':nonzero}
                del output,gt,kl,loss
            assert all(torch.equal(batch[name],value) for name,value in originals.items()), 'Arm mutated shared real input'
            return metrics

        reload_parent();smoke_times=[]
        for index,batch in enumerate(loader,8000):
            t=time.monotonic();metrics=paired_step(batch,index);smoke_times.append(time.monotonic()-t)
            print(json.dumps({'discarded_pair_smoke':index-7999,'metrics':metrics,'seconds':smoke_times[-1]}),flush=True)
        for arm in ARMS:assert_provider_unchanged(models[arm],snapshots[arm])
        write(WORK/'GPU_SMOKE_PASS.json',{'status':'PASS','identity_SHA':digest,'canvas':2048,'batch':2,
            'discarded_updates_per_arm':3,'formal_updates':0,'actual_inputs_shared_unchanged':True,
            'provider_outputs_bitwise_equal':True,'provider_parameters_BN_unchanged':True,
            'private_finite_nonzero_gradients':True,'seconds_per_pair':smoke_times})
        reload_parent()
        write(WORK/'PROBE_STARTED.json',{'identity_SHA':digest,'controller_pid':os.getpid(),'time_unix':time.time(),
            'own_stop_unix':deadline,'automatic_fresh_training':False,'test_accessed':False})
        provider_counts=None

        def validate(completed):
            nonlocal provider_counts
            results={}
            for arm in ARMS:
                core.WORK=WORK/arm
                result=core.evaluate(models[arm]);assert_frozen_provider(models[arm],optimizers[arm])
                assert_provider_unchanged(models[arm],snapshots[arm])
                counts=result['base_head_metrics']['counts']
                if provider_counts is None:provider_counts=counts
                else:assert counts==provider_counts,'Frozen full-val provider counts changed'
                result.update(parent_completed=8000,probe_updates=completed,arm=arm,identity_SHA=digest,
                    provider_parameters_BN_bitwise_unchanged=True)
                write(core.WORK/('val_probe_%05d.json'%completed),result);results[arm]=result
            if completed==0:assert results['control']['counts']==results['gt_stratified']['counts'],'Initial private outputs differ'
            return results

        values={'0':validate(0)};sampler.start,sampler.stop=8000,8500
        started=time.monotonic()
        with (WORK/'batch_alignment.jsonl').open('x',buffering=1) as trace:
            expected=iter(core.ResumableBatches(len(dataset),8000,8500))
            for index,batch in enumerate(loader,8000):
                assert time.time()<deadline,'Bounded diagnostic deadline reached'
                exposure=next(expected);assert batch['sid']==[dataset.rows[i]['sid'] for i,_ in exposure]
                t=time.monotonic();metrics=paired_step(batch,index);completed=index-7999
                trace.write(json.dumps({'update':index+1,'sid':batch['sid'],'augmentation_seeds':[s for _,s in exposure],
                    'same_real_input_objects_unchanged':True,'provider_logits_bitwise_equal':True})+'\n')
                if completed%25==0:
                    result={'status':'DIAGNOSTIC_TRAINING','probe_updates_per_arm':completed,'identity_SHA':digest,
                        'metrics':metrics,'pair_step_seconds':time.monotonic()-t,'time_unix':time.time()}
                    write(WORK/'progress.json',result);print(json.dumps(result),flush=True)
                if completed in (250,500):values[str(completed)]=validate(completed)
        assert completed==500
        for arm in ARMS:
            path=WORK/arm/'probe_last.pt'
            core.save_checkpoint(path,{'model':models[arm].state_dict(),'optimizer':optimizers[arm].state_dict(),
                'rng':core.rng_state(),'parent_completed':8000,'probe_updates':500,'identity_SHA':digest,'arm':arm})
        outcome={'status':'COMPLETE','identity_SHA':digest,'values':values,'updates_per_arm':500,
            'seconds':time.monotonic()-started,'provider_state_and_full_val_counts_identical':True,
            'all_real_training_inputs_shared_unchanged':True,'fresh_screen_PASS':False,'test_accessed':False,
            'goal_achieved':False,'server_kept_running':True,'external_diagnostics_controlled':False,'time_unix':time.time()}
        write(WORK/'PROBE_COMPLETE.json',outcome);write(REPORTS/'OUTCOME.json',outcome)


if __name__=='__main__':
    try:run()
    except BaseException as error:
        REPORTS.mkdir(parents=True,exist_ok=True)
        write(REPORTS/'ERROR.json',{'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),
            'time_unix':time.time(),'server_kept_running':True,'external_diagnostics_controlled':False})
        raise
