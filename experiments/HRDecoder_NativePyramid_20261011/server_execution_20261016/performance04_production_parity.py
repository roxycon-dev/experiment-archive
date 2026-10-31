"""Extra zero-SGD actual-production (no-remat/micro1) parity on source04."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,sys,traceback
from pathlib import Path
BASE=Path('/root/autodl-tmp/nativepyramid_sp_v3_20261016')
sys.path.insert(0,str(BASE/'code_performance04_candidate'))
import torch
from train import configure_runtime,seed_all,execution_model,source_identity,write_json
from losses import SparseLesionLoss

if __name__=='__main__':
    configure_runtime();torch.set_num_threads(4);seed_all(42)
    result={'source_sha256':source_identity()[0],'SGD_updates':0,'test_accessed':False,'records':[]}
    try:
        assert result['source_sha256']=='14e47c3749db5dd28c2938ddbb227aad66b07c40b4fb9d049fdb0564b91c95d0'
        model=execution_model(BASE/'assets/hrnetv2_w48-d2186c55.pth').cuda().train()
        assert not model.backbone.checkpointing and not model.decoder.checkpointing
        assert all(p.dtype==torch.float32 for p in model.parameters())
        assert all(p.is_contiguous(memory_format=torch.channels_last) for p in model.parameters() if p.ndim==4)
        buffers={n:b.detach().cpu().clone() for n,b in model.named_buffers()}
        image=torch.randn(2,3,128,128,device='cuda')
        label=torch.zeros(2,128,128,dtype=torch.long,device='cuda')
        label[0,:4,:4]=1;label[0,4:8,:4]=2;label[1,:32,:32]=3;label[1,32:48,:32]=4;label[:,-8:]=255
        criterion=SparseLesionLoss();denominators=criterion.batch_denominators(label)
        for fixture in ('zero_residual','nonzero_residual'):
            if fixture=='nonzero_residual':
                with torch.no_grad():
                    for p in (model.decoder.auxiliary.weight,model.decoder.readout.weight):
                        p.copy_(torch.linspace(-.01,.01,p.numel(),device=p.device).reshape_as(p))
            for amp in (False,True):
                tolerance={'rtol':.03,'atol':.0005} if amp else {'rtol':.0003,'atol':.000003}
                reference=None
                for micro in (2,1):
                    model.zero_grad(set_to_none=True);pieces=[];semantic=[];total=0.
                    for j in range(0,2,micro):
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):output=model(image[j:j+micro])
                        loss,_=criterion(output,label[j:j+micro],denominators=denominators)
                        loss.backward();pieces.append(output['logits'].detach().cpu());semantic.append(output['auxiliary_logits'].detach().cpu());total+=float(loss.detach())
                        del output,loss
                    current={'Z':torch.cat(pieces),'S':torch.cat(semantic),'loss':torch.tensor(total)}
                    gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
                    assert all(bool(torch.isfinite(g).all()) for g in gradients.values())
                    if reference is None:reference=(current,gradients)
                    else:
                        for key,value in current.items():torch.testing.assert_close(value,reference[0][key],**tolerance)
                        for name,gradient in gradients.items():torch.testing.assert_close(gradient,reference[1][name],msg=name,**tolerance)
                    result['records'].append({'fixture':fixture,'BF16':amp,'checkpointing':False,
                        'physical_forward_batch':1,'loss_batch':micro,'backward_calls':2//micro,
                        'all_parameter_gradients_compared':True,'tolerance':tolerance,'loss':total})
                del reference,gradients
        assert all(torch.equal(b.detach().cpu(),buffers[n]) for n,b in model.named_buffers())
        result.update(status='PASS',all_BN_buffers_unchanged=True,actual_checkpointing=False,actual_memory_format='channels_last')
    except BaseException as error:
        result.update(status='FAILED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    write_json(BASE/'receipts/performance04_production_parity.json',result)
    print(json.dumps(result),flush=True)
