import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trainer.loss import masked_bce_dice_loss
from trainer.models.weight_loading import load_pretrained_subset
from trainer.eval import sliding_window_logits
from trainer.screening import decide_stop
from trainer.config import gate_config,load_yaml,metrics_config

def main():
    results=[]
    x=torch.zeros((1,4,256,256),dtype=torch.float16,requires_grad=True)
    y=torch.zeros_like(x);y[:,:,100:104,100:104]=1
    loss,_=masked_bce_dice_loss(x,y,torch.ones_like(x))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    results.append('fp16 reductions exceeding 65504 remain finite, with usable gradients')
    class FullModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=torch.nn.Sequential(*[torch.nn.Linear(2,2) for _ in range(12)])
            self.decode_head=torch.nn.Linear(2,4)
    source=FullModel();target=FullModel()
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'external.pth';torch.save({'model_state':source.state_dict()},path)
        r=load_pretrained_subset(target,ckpt_path=str(path),label='transfer')
        assert not r['missing'] and not r['unexpected']
        assert all(torch.equal(v,target.state_dict()[k]) for k,v in source.state_dict().items())
    results.append('full external transfer preserves backbone prefixes and restores decoder exactly')
    class PixelModel:
        def eval_forward(self,x):return x[:,:1].expand(-1,4,-1,-1)
    x=torch.randn(1,3,32,128)
    got=sliding_window_logits(PixelModel(),x,64,32)
    assert got.shape==(1,4,32,128) and torch.allclose(got,x[:,:1].expand(-1,4,-1,-1))
    results.append('sliding windows cover short edges and preserve original coordinates')
    policy=dict(enabled=True,reference_miou=.426694,minimum_gain=.003,gap=.015,min_progress=.5,
                reference_class_iou={'HE':.437583})
    rs=[dict(update=3000,mIoU_lesion4=.3,per_class={'HE':{'iou':.4}}),
        dict(update=6000,mIoU_lesion4=.301,per_class={'HE':{'iou':.4}})]
    assert decide_stop(rs,12000,policy)['stop']
    rs[-1]['mIoU_lesion4']=.32
    assert not decide_stop(rs,12000,policy)['stop']
    rs[-1]['mIoU_lesion4']=.301;rs[-1]['per_class']['HE']['iou']=.45
    assert not decide_stop(rs,12000,policy)['stop']
    results.append('screening stops a plateau, preserves learning and class complementarity')
    cfg=load_yaml(str(Path(__file__).resolve().parents[1]/'configs/resolved/U1_convnext_unet_ext_pretrain.yaml'))
    assert metrics_config(cfg)['resize_to']==[2048,2048]
    gate_config(cfg,mode='formal',stage='external_pretrain')
    try:gate_config(cfg,mode='formal',stage='ddr_finetune')
    except Exception:pass
    else:raise AssertionError('missing transfer checkpoint must block DDR adaptation')
    results.append('external pretrain has no circular output gate; DDR still requires its checkpoint')
    out=Path(__file__).resolve().parents[1]/'reports/regressions.json'
    out.write_text(json.dumps({'passed':len(results),'checks':results},indent=2),encoding='utf-8')
    print(json.dumps({'passed':len(results),'checks':results},indent=2))

if __name__=='__main__':main()
