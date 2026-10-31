"""Create a separate registered candidate; never edit prior experimental sources."""
import hashlib
import json
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / 'sacd_execution'
TEMPLATE = ROOT / 'sacd_isolated_execution'
NEW = ROOT / 'residual_execution'
NEW.mkdir(exist_ok=True)
assert not (NEW / 'SEALED_LOCAL_SOURCE.json').exists()
for name in ('baseline.py','private_modules.py','stable_objectives.py','data.py','loader.py'):
    (NEW / name).write_bytes((OLD / name).read_bytes())

source=(OLD/'model.py').read_text(encoding='utf-8')
source=source.replace('from objectives import transport_logits','')
source=source.replace('nn.Conv2d(32, 48, 1, bias=True)','nn.Conv2d(32, 16, 1, bias=True)')
start=source.index('            with torch.no_grad():')
end=source.index('            self.readouts.append',start)
source=source[:start]+'            nn.init.zeros_(output.bias)\n'+source[end:]
source=source.replace('== 40768','== 36544').replace('== 6419456','== 6415232')
source=source.replace('return F.pixel_shuffle(raw,4).reshape(raw.shape[0],4,3,p0.shape[2],p0.shape[3])','return F.pixel_shuffle(raw,4)')
source=source.replace('        f1, f2, f3, f4 = features','        f1, f2, f3, f4 = (feature.detach() for feature in features)')
start=source.index('        actions = self.private(image, features, z0)')
end=source.index('\ndef inference_probabilities',start)
source=source[:start]+'''        residual = self.private(image, features, z0)
        # Both direct and feature-mediated private-to-provider routes are cut.
        # Numerically this is a continuous signed logit residual, in FP32.
        logits = z0.float().detach() + residual.float()
        return {'logits': logits, 'base_logits': z0, 'residual_logits': residual,
                'hr_loss': hr_loss}

'''+source[end:]
source=source[:source.index('\ndef inference_probabilities')]+'''
def inference_probabilities(output, original_hw):
    # Addition is linear; sigmoid follows native-resolution logit interpolation.
    logits = F.interpolate(output['logits'].float(), size=original_hw,
                           mode='bilinear', align_corners=False)
    return logits.sigmoid()
'''
(NEW/'model.py').write_text(source,encoding='utf-8')
(NEW/'fast_objectives.py').write_text('''"""Original squared-denominator Dice, with previously registered valid masking."""
import torch
import torch.nn.functional as F

def masks(logits,label,valid):
    labels=label[:,0] if label.ndim==4 else label
    y=torch.stack([labels==c for c in range(1,5)],1)
    if valid.ndim==3:valid=valid[:,None]
    return y,valid.bool().expand_as(y)&(labels[:,None]!=255)

def macro(value,mask):
    active=mask.sum((-2,-1))>0
    return (value*active).sum()/active.sum().clamp_min(1)

def gt_loss(logits,label,valid):
    z=logits.double() if logits.dtype==torch.float64 else logits.float()
    y,m=masks(z,label,valid); p=z.sigmoid()
    dice=1-(2*(p*y*m).sum((-2,-1))+1e-5)/((p.square()*m).sum((-2,-1))+(y*m).sum((-2,-1))+1e-5)
    # Empty but annotated classes remain included, as in original BinaryDice.
    return macro(dice,m)

def residual_gt(logits,label,valid):
    z=logits.double() if logits.dtype==torch.float64 else logits.float()
    y,m=masks(z,label,valid)
    bce=F.binary_cross_entropy_with_logits(z,y.to(z.dtype),reduction='none')
    return gt_loss(z,label,valid)+.25*macro((bce*m).sum((-2,-1))/m.sum((-2,-1)).clamp_min(1),m)

def trusted_kl(logits,teacher,label,valid):
    z=logits.double() if logits.dtype==torch.float64 else logits.float()
    q=teacher.detach().to(z.dtype);y,m=masks(z,label,valid)
    trusted=(q>.5)==y
    kl=(F.binary_cross_entropy_with_logits(z,q,reduction='none')+torch.xlogy(q,q)+torch.xlogy(1-q,1-q)).clamp_min(0)
    return macro((kl*m*trusted).sum((-2,-1))/m.sum((-2,-1)).clamp_min(1),m)
''',encoding='utf-8')
(NEW/'objectives.py').write_text('''"""Independent base and residual supervision; no moving student action targets."""
import torch
from fast_objectives import gt_loss,residual_gt,trusted_kl

def lambda_k(update):
    assert 0<=update<=40000
    if update<=1500:return 0.
    if update<=4000:return .5*(update-1500)/2500
    if update<=32000:return .5
    return .5-.4*(update-32000)/8000

def learning_rate(index):
    assert 0<=index<40000
    value=.005*(1-index/40000)
    if index<1500:value*=1-(1-index/1500)*(1-1e-6)
    return value

@torch.no_grad()
def fixed_teacher(a,u,s,label):
    a,u,s=(x.detach().float() for x in (a,u,s))
    q=.5625*a+.1875*u+.25*s
    high=torch.maximum(torch.maximum(a,u),s);low=torch.minimum(torch.minimum(a,u),s)
    labels=label[:,0] if label.ndim==4 else label
    y=torch.stack([labels==c for c in range(1,5)],1)
    # Correction depends only on cached train teachers and train GT, never on z0.
    return torch.where(y&(q<=.5)&(high>.5),high,
                       torch.where(~y&(q>.5)&(low<=.5),low,q))

def losses(output,label,valid,a,u,s,update):
    final_gt=residual_gt(output['logits'],label,valid)
    base_gt=gt_loss(output['base_logits'],label,valid)
    kd=trusted_kl(output['logits'],fixed_teacher(a,u,s,label),label,valid)
    lk=lambda_k(update)
    task=base_gt+.1*output['hr_loss']+final_gt+lk*kd
    zero=task.detach()*0
    return task,zero,{'gt_final':final_gt,'gt_base':base_gt,'gt_hr_crop':output['hr_loss'],
                      'kl_final':kd,'lambda_k':lk,'task_loss':task,'action_kl_removed':True}

def routed_backward(task,auxiliary,parameters,*,action_active=True):
    # All desired paths are already encoded by detached provider boundaries.
    task.backward()
''',encoding='utf-8')

protocol={'design_version':8,'name':'HRDecoder-DecoupledResidual','seed':42,'private_seed':20261008,
 'canvas':2048,'updates':40000,'global_batch':2,'precision':'BF16 network; FP32 residual composition/probabilities/losses',
 'optimizer':{'type':'SGD','lr':.005,'momentum':.9,'weight_decay':.0005},
 'lr':{'type':'poly','power':1,'warmup':1500,'warmup_ratio':1e-6},
 'lambda_k':[[0,0],[1500,0],[4000,.5],[32000,.5],[40000,.1]],
 'GT':'base original-formula Dice + .1 HRcrop Dice; private Dice + .25 ordinary all-valid BCE',
 'mask':'Existing geometry-valid and ignore255 masking retained; not claiming every original training detail identical',
 'KD':'private only fixed GT-consistent train teacher T*, all-valid normalization; no moving student targets',
 'readout':'four independent continuous residual readouts conv32->16+pixelshuffle4; exactly zero initialized',
 'gradient_route':'all four HRNet feature inputs and p0 detached; final=z0.float().detach()+r.float(); no private gradient into base',
 'inference':'resize final FP32 logits then sigmoid; no teacher or GT',
 'classes':['EX','HE','SE','MA'],'threshold':'strictly > .5','validation_images':76,
 'validation_interval':4000,'student_TTA':False,'inference_teacher':False,'inference_online_H4':False,
 'test_selection':False,'shutdown_after_completion':False,'data_cache_hash_rescan':False,
 'initialization':'fresh ImageNet W48 seed42; no warm task weights','screen_updates':8000,
 'accuracy_guaranteed':False}
(NEW/'architecture_spec.json').write_text(json.dumps(protocol,indent=2)+'\n')
screen={'registered_before_any_new_candidate_results':True,'screen_updates':8000,
 'original_deadline_unix':1791568800,'criteria':{
 '4k_mIoU_at_least':28.825649019798055,'8k_mIoU_at_least':34.469921210377,
 'gain_4k_to8k_pp_at_least':3.,'final_minus_base_8k_pp_at_least':.5,
 'max_class_IoU_loss_final_vs_base_pp':2.,
 'HE_MA_reject_if_FN_growth_over':1.1,'HE_MA_requires_FP_ratio_when_FN_grows_at_most':.9},
 'interpretation':'Practical early-trend and calibration screen, not proof of40k or fusion target',
 'test_access_for_screening':False,'reject_next_action':'autonomous evidence-based revision within remaining authorized time'}
(NEW/'screen_protocol.json').write_text(json.dumps(screen,indent=2)+'\n')
source=(TEMPLATE/'train.py').read_text(encoding='utf-8').replace('sacd_isolated','residual')
start=source.index('PROTOCOL = ');end=source.index('\ndef write_json',start)
source=source[:start]+'PROTOCOL = '+repr(protocol)+'\n'+source[end:]
source=source.replace("'CPU_SACD_PASS.json'","'CPU_RESIDUAL_PASS.json'")
start=source.index("        warm=json.loads");end=source.index("        smoke=json.loads",start)
source=source[:start]+source[end:]
source=source.replace('71991876','71987652').replace('6419456','6415232')
(NEW/'train.py').write_text(source,encoding='utf-8')
source=(TEMPLATE/'test_eval.py').read_text(encoding='utf-8').replace('sacd_isolated','residual')
(NEW/'test_eval.py').write_text(source,encoding='utf-8')
for path in NEW.glob('*.py'):py_compile.compile(str(path),doraise=True)
print(json.dumps({'prepared':str(NEW),'formal_updates':0,'architecture':protocol['name']}))
