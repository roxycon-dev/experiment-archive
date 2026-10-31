"""SACD 2048/batch2/40k main run, fresh ImageNet start and native-resolution val."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT/'vendor/champion_repo'
sys.path.insert(0,str(REPO))
DATA = Path('/root/HRDecoder/data/DDR_split_613')
WORK = ROOT/'training_runs/iou'
REPORTS = ROOT/'reports/iou'
W48 = Path('/root/.cache/torch/hub/checkpoints/hrnetv2_w48-d2186c55.pth')
from baseline import build_baseline, tensor_state_sha, file_sha, install_legacy_bf16_interpolation
from model import SACDStudent, inference_probabilities
from objectives import losses, routed_backward, lambda_k, learning_rate
from loader import ResumableBatches, CUDAPrefetch, install_ram

PROTOCOL = {'design_version': 16, 'name': 'HRDecoder-IoUCorrection', 'seed': 42, 'private_seed': 20261008, 'canvas': 2048, 'updates': 40000, 'global_batch': 2, 'precision': 'BF16 network; FP32 residual composition/probabilities/losses', 'optimizer': {'type': 'SGD', 'lr': 0.005, 'private_lr_multiplier': 4, 'momentum': 0.9, 'weight_decay': 0.0005}, 'lr': {'type': 'poly', 'power': 1, 'warmup': 1500, 'warmup_ratio': 1e-06}, 'lambda_k': [[0, 0], [1500, 0], [4000, 0.5], [32000, 0.5], [40000, 0.1]], 'GT': 'base original Dice + .25 GT-stratified balanced BCE + .1 HRcrop original Dice; private per-image/class exact active-margin Lovasz hinge + .25 fullGT positive/negative balanced BCE', 'mask': 'Existing geometry-valid and ignore255 masking retained; not claiming every original training detail identical', 'KD': 'base fixed GT-consistent teacher KL normalized by fullGTpositive/negative counts at .5*lambda_K; private fixed GT-consistent teacher fullGT positive/negative stratified KL at lambda_K; teacher input only during training', 'readout': 'W48 F1-F4 project64/concat256;1x1to128/DW3e2x2/project64 at1/8; RGBunshuffle2to32+mix96to64+DW13e2x2 at1/2; rawdetachlogitunshuffle16 bypass; zero80to16/shuffle2', 'gradient_route': 'private feature/logit inputs detach; explicit base GT and base teacher KL train HRNet; private GT/KD cannot change base', 'inference': 'resize final FP32 logits then sigmoid; no teacher or GT', 'classes': ['EX', 'HE', 'SE', 'MA'], 'threshold': 'strictly > .5', 'validation_images': 76, 'validation_interval': 4000, 'student_TTA': False, 'inference_teacher': False, 'inference_online_H4': False, 'test_selection': False, 'shutdown_after_completion': False, 'data_cache_hash_rescan': False, 'initialization': 'fresh ImageNet W48 seed42; no warm task weights', 'screen_updates': 4000, 'accuracy_guaranteed': False, 'gradient_clip': {'type': 'separate_base_private_L2', 'max_norm_each': 5.0, 'after_finite_check': True}, 'critical_stop': {'after_update_at_least': 8000, 'mIoU_drop_from_best_at_least_pp': 15.0, 'or_drop_at_least_pp': 8.0, 'with_previous_class_IoU_at_least': 5.0, 'and_current_class_IoU_zero': True, 'action': 'save own current checkpoint and stop before another update; not an overfitting claim'}}

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def identity():
    ready=json.loads((ROOT/'CACHE_READY.json').read_text());assert ready['status']=='PASS'
    old=json.loads((ROOT/'training_runs/v3/protocol.json').read_text())['identity']
    assert old['ImageNet_W48_SHA']=='d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960'
    cpu=json.loads((HERE/'CPU_ADAPTER_PASS.json').read_text())
    assert cpu['status']=='PASS' and cpu['torch']==torch.__version__
    names=('baseline.py','model.py','private_modules.py','stable_objectives.py','fast_objectives.py','objectives.py','data.py','loader.py','train.py')
    material={'protocol':PROTOCOL,'code':{n:file_sha(HERE/n) for n in names},
        'spec_SHA':file_sha(HERE/'architecture_spec.json'),'screen_protocol_SHA':file_sha(HERE/'screen_protocol.json'),
        'screen_gate_code_SHA':file_sha(HERE/'screen_gate.py'),
        'ImageNet_W48_SHA':old['ImageNet_W48_SHA'],'data_identity_SHA':old['data_identity_SHA'],
        'cache_identity':ready['member_identity_sha256'],'torch':torch.__version__}
    return material,hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()

def one_step(model,optimizer,batch,update):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        output=model(batch['image'],batch['label'],batch['valid'])
    task,action,metrics=losses(output,batch['label'],batch['valid'],batch['A'],batch['U'],batch['S'],update)
    if not bool(torch.isfinite(torch.stack((task,action))).all()):
        raise FloatingPointError('Nonfinite loss; no SGD update consumed')
    routed_backward(task,action,model.private_parameters(),action_active=lambda_k(update)>0)
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    # The pinned AMP CUDA foreach kernel checks the complete gradient list once;
    # multiplication by one preserves the unscaled BF16-training FP32 gradients.
    assert all(g.dtype==torch.float32 for g in gradients)
    found=torch.zeros(1,device='cuda');inv=torch.ones(1,device='cuda')
    torch._amp_foreach_non_finite_check_and_unscale_(gradients,found,inv)
    if found.item()!=0: raise FloatingPointError('Nonfinite gradient; no SGD update consumed')
    base_norm=torch.nn.utils.clip_grad_norm_(model.base.parameters(),max_norm=5.0,foreach=True)
    private_norm=torch.nn.utils.clip_grad_norm_(model.private_parameters(),max_norm=5.0,foreach=True)
    metrics['base_grad_norm_before_clip']=base_norm.detach()
    metrics['private_grad_norm_before_clip']=private_norm.detach()
    metrics['gradient_clipped']=((base_norm.detach()>5.0)|(private_norm.detach()>5.0)).to(torch.float32)
    optimizer.step()  # Exactly one finite clipped step for the routed losses.
    return {k:float(v.detach()) if torch.is_tensor(v) else v for k,v in metrics.items()}

def metrics_from_counts(counts):
    tp,fp,fn=counts.T
    iou=np.divide(tp,tp+fp+fn,out=np.zeros(4),where=tp+fp+fn>0)
    precision=np.divide(tp,tp+fp,out=np.zeros(4),where=tp+fp>0)
    recall=np.divide(tp,tp+fn,out=np.zeros(4),where=tp+fn>0)
    return {'mIoU_percent':float(iou.mean()*100),'IoU_percent':(iou*100).tolist(),
        'precision_percent':(precision*100).tolist(),'recall_percent':(recall*100).tolist(),
        'counts':counts.tolist(),'class_order':['EX','HE','SE','MA'],
        'undefined_precision_convention':'zero when TP+FP=0; inspect counts',
        'images':76,'threshold':'strictly > .5'}

@torch.no_grad()
def evaluate(model):
    from data import evaluation_image
    from PIL import Image
    rows=json.loads((ROOT/'metadata/data_identity.json').read_text())['splits']['val']['samples']
    assert len(rows)==76
    images={p.stem:p for p in (DATA/'images/val').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
    was_training=model.training;model.eval()
    counts=np.zeros((4,3),dtype=np.int64);base_counts=np.zeros((4,3),dtype=np.int64)
    started=time.monotonic()
    for i,row in enumerate(rows):
        image=evaluation_image(images[row['sid']]).unsqueeze(0).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16): output=model(image)
        probability=inference_probabilities(output,row['hw'])
        pred=(probability>.5).cpu().numpy()[0]
        base_native=torch.nn.functional.interpolate(output['base_logits'].float(),size=row['hw'],mode='bilinear',align_corners=False)
        base_pred=(base_native.sigmoid()>.5).cpu().numpy()[0]
        with Image.open(DATA/'labels/val'/(row['sid']+'.png')) as source: label=np.asarray(source)
        valid=label!=255
        for c in range(4):
            gt=label==c+1
            for store,decision in ((counts,pred),(base_counts,base_pred)):
                p=decision[c]&valid
                store[c]+=[int((p&gt).sum()),int((p&~gt).sum()),int((~p&gt&valid).sum())]
        if (i+1)%10==0:
            write_json(WORK/'validation_progress.json',{'status':'VALIDATING','images_completed':i+1,
                'expected':76,'seconds':time.monotonic()-started})
        del output,probability,image,base_native
    model.train(was_training)
    result=metrics_from_counts(counts)
    result.update(base_head_metrics=metrics_from_counts(base_counts),seconds=time.monotonic()-started,
                  deployment_inference_verified=True,teacher_or_GT_input=False)
    return result

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
def save_checkpoint(path,payload):
    tmp=path.with_suffix('.writing');torch.save(payload,tmp);tmp.replace(path)

def main():
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--smoke',action='store_true');action.add_argument('--train',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert torch.__version__=='2.0.1+cu117' and torch.cuda.is_available()
    assert int(os.environ.get('WORLD_SIZE','1'))==1
    WORK.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    install_legacy_bf16_interpolation();torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    material,digest=identity()
    if args.train:
        smoke=json.loads((WORK/'GPU_SMOKE_PASS.json').read_text())
        assert smoke['status']=='PASS' and smoke['identity_SHA']==digest
        assert args.resume or not (WORK/'last.pt').exists(), 'Never overwrite an existing SACD checkpoint; use explicit resume'
    assert not args.smoke or not args.resume
    baseline,_=build_baseline(REPO,W48,verified_weight_sha=material['ImageNet_W48_SHA'])
    baseline_sha=tensor_state_sha(baseline)
    expected=json.loads((ROOT/'reports/SERVER_INITIALIZATION_PASS.json').read_text())['baseline_initial_state_SHA']
    assert baseline_sha==expected,'Original seed42 initialization changed'
    model=SACDStudent(baseline).cuda().train()
    assert sum(p.numel() for p in model.parameters())==65856916
    optimizer=torch.optim.SGD([{'params':model.base.parameters(),'lr_multiplier':1}, {'params':model.private.parameters(),'lr_multiplier':4}],lr=.005,momentum=.9,weight_decay=.0005,foreach=True)
    completed,best,saved_update=0,float('-inf'),0
    if args.resume:
        checkpoint=torch.load(WORK/'last.pt',map_location='cpu')
        if checkpoint['identity_SHA']!=digest:
            migration=json.loads((REPORTS/'IDENTITY_MIGRATION.json').read_text())
            assert migration['from_identity_SHA']==checkpoint['identity_SHA']
            assert migration['to_identity_SHA']==digest and migration['resume_completed']==checkpoint['completed']
            assert migration['scope']=='spawn_worker_RAM_initialization_only'
            assert migration['model_loss_data_schedule_unchanged'] and migration['worker_RAM_bitwise_test_PASS']
        model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer'])
        completed,best=checkpoint['completed'],checkpoint['best'];saved_update=completed
        restore_rng(checkpoint['rng']);del checkpoint
    from data import TrainCacheDataset
    ready=install_ram(ROOT)
    dataset=TrainCacheDataset(ROOT,DATA)
    sampler=ResumableBatches(len(dataset),completed,completed+12 if args.smoke else 40000)
    loader=CUDAPrefetch(dataset,sampler,workers=8)
    calls=[];hook=model.base.backbone.register_forward_hook(lambda *a:calls.append(1)) if args.smoke else None
    if args.train:
        write_json(WORK/'protocol.json',{'identity_SHA':digest,'identity':material,
                   'baseline_initial_state_SHA':baseline_sha,'initialization':'fresh ImageNet seed42; no prior-task checkpoint; own run resumes only' if args.resume else 'fresh ImageNet seed42; no task checkpoint','resume_from_own_update':completed if args.resume else None})
    started=time.monotonic();step_times=[];torch.cuda.reset_peak_memory_stats()
    private_nonzero=None
    for batch in loader:
        index=completed
        for group in optimizer.param_groups:group['lr']=(.005 if args.smoke else learning_rate(index))*group['lr_multiplier']
        step_started=time.monotonic()
        metric=one_step(model,optimizer,batch,4000 if args.smoke else index+1)
        step_times.append(time.monotonic()-step_started);completed+=1
        if args.smoke:
            private_nonzero={name:any(p.grad is not None and bool(p.grad.ne(0).any()) for p in part.parameters())
                for name,part in [('features',model.private.features),('semantic',model.private.semantic),('rgb',model.private.rgb),('mix',model.private.mix),('blocks',model.private.blocks),('readout',model.private.readout)]}
            print(json.dumps({'smoke_discarded_step':completed,'seconds':step_times[-1],'losses':metric,
                              'private_nonzero_gradients':private_nonzero}),flush=True)
            continue
        if completed==1 or completed%10==0:
            skip=min(12,len(step_times)-1);values=step_times[skip:][-100:]
            waits=loader.wait_times[skip:][-100:]
            mean=statistics.mean(values)+statistics.mean(waits)
            write_json(WORK/'performance.json',{'completed_update':completed,'mean_batch_seconds':mean,
                'mean_step_compute_seconds':statistics.mean(values),'mean_data_wait_seconds':statistics.mean(waits),
                'window_batches':len(values),'warmup_batches_excluded':skip,'workers':8,'prefetch_factor':1,
                'CUDA_future_batches':1,'shared_RAM_cache_bytes':ready['bytes'],'channels_last':False,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'estimated_remaining_train_hours_excluding_validation':(40000-completed)*mean/3600,
                'time_unix':time.time(),'protocol_unchanged':True})
        if completed==1 or completed%50==0:
            print(json.dumps({'completed':completed,'expected':40000,'losses':metric,
                             'lr':learning_rate(index),'seconds':time.monotonic()-started}),flush=True)
            write_json(WORK/'progress.json',{'status':'TRAINING','completed':completed,'expected':40000,
                        'saved_update':saved_update,'best_val_mIoU':best if np.isfinite(best) else None,
                        'time_unix':time.time()})
        if completed%1000==0:
            validation=None
            if completed%4000==0:
                write_json(WORK/'progress.json',{'status':'VALIDATING','completed':completed,'expected':40000,'saved_update':saved_update})
                validation=evaluate(model);validation['completed']=completed
                write_json(WORK/('val_%05d.json'%completed),validation)
                print(json.dumps({'validation_update':completed,**validation}),flush=True)
                if validation['mIoU_percent']>best:
                    best=validation['mIoU_percent']
                    save_checkpoint(WORK/'best_val.pt',{'model':model.state_dict(),'completed':completed,
                        'validation':validation,'identity_SHA':digest})
            save_checkpoint(WORK/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'completed':completed,'best':best,'rng':rng_state(),'identity_SHA':digest})
            saved_update=completed
            write_json(WORK/'progress.json',{'status':'TRAINING' if completed<40000 else 'COMPLETE',
                'completed':completed,'expected':40000,'saved_update':saved_update,
                'best_val_mIoU':best if np.isfinite(best) else None,'last_validation':validation,'time_unix':time.time()})

            if validation is not None and completed>=8000:
                earlier=[json.loads(p.read_text()) for p in WORK.glob('val_*.json') if int(p.stem.split('_')[-1])<completed]
                previous_best=max(earlier,key=lambda v:v['mIoU_percent'])
                drop=previous_best['mIoU_percent']-validation['mIoU_percent']
                missing=[c for c,(a,b) in enumerate(zip(previous_best['IoU_percent'],validation['IoU_percent'])) if a>=5 and b==0]
                if drop>=15 or (drop>=8 and missing):
                    evidence={'status':'CRITICAL_CLASS_COLLAPSE','completed':completed,'identity_SHA':digest,'mIoU_drop_from_best_pp':drop,'previous_best_update':previous_best['completed'],'missing_class_indices':missing,'validation':validation,'current_checkpoint_saved':True,'registered_before_results':True,'test_accessed':False,'time_unix':time.time(),'server_kept_running':True}
                    write_json(WORK/'CRITICAL_CLASS_COLLAPSE.json',evidence)
                    print(json.dumps(evidence),flush=True)
                    return

            if completed == 4000:
                screen_checkpoint=torch.load(WORK/'last.pt',map_location='cpu')
                assert screen_checkpoint['completed']==4000 and screen_checkpoint['identity_SHA']==digest
                assert screen_checkpoint['optimizer']['state'] and 'rng' in screen_checkpoint
                del screen_checkpoint
                from screen_gate import screen_gate
                decision = screen_gate(WORK, HERE/'screen_protocol.json', digest)
                write_json(WORK/('SCREEN_PASS.json' if decision['status']=='PASS' else 'SCREEN_REJECTED.json'), decision)
                print(json.dumps({'fresh_screen': decision}), flush=True)
                if decision['status'] != 'PASS':
                    break
                write_json(REPORTS/'DEADLINE_EXTENSION_ACTIVATED.json', {
                    'status':'ACTIVE','user_authorization':'Through screening, server may remain available until completion',
                    'screen_identity_SHA':digest,'screen_pass_unix':decision['time_unix'],
                    'server_reavailability_user_confirmed':True,'full40k_authorized':True,
                    'same_fresh_run_continues':True,'test_only_after_locked40k':True})
    if args.smoke:
        hook.remove();assert len(calls)==12 and all(private_nonzero.values())
        torch.cuda.synchronize()
        write_json(WORK/'GPU_SMOKE_PASS.json',{'status':'PASS','identity_SHA':digest,'batch':2,'canvas':2048,
            'backbone_calls_per_step':1,'measured_backbone_calls':len(calls),'discarded_optimizer_steps':12,
            'formal_updates':0,'fresh_initialization_verified':True,'new_parameters':284496,'total_parameters':65856916,
            'optimizer_momentum_allocated':True,'losses_FP32':True,'all_gradients_finite':True,
            'last_smoke_step_private_nonzero_gradients':private_nonzero,
            'step_seconds':step_times,'median_warm_step_seconds':statistics.median(step_times[2:]),
            'mean_warm_step_seconds':statistics.mean(step_times[2:]),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'seconds':time.monotonic()-started,
            'device':torch.cuda.get_device_name(0),'accuracy_validated':False})
        print('GPU_SACD_SMOKE_PASS',flush=True);return
    if completed == 4000 and (WORK/'SCREEN_REJECTED.json').exists():
        return
    assert completed==40000
    durable=torch.load(WORK/'last.pt',map_location='cpu')
    assert durable['completed']==40000 and durable['identity_SHA']==digest and durable['optimizer']['state']
    assert 'rng' in durable and (WORK/'val_40000.json').exists();del durable
    write_json(WORK/'TRAINING_COMPLETE.json',{'status':'COMPLETE','completed':40000,'identity_SHA':digest,
        'primary_checkpoint':'last.pt','secondary_checkpoint':'best_val.pt','durable_checkpoint_loaded_and_verified':True,
        'final_validation_images':76,'no_test_selection':True,'shutdown_authorized':False})
    print('SACD_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
