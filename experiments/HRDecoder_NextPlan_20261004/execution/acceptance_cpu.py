"""Small necessary checks for validity gradients, trainability, and decision boundaries."""
import copy
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import adaptation
from decision import promoted,catastrophic,select,FORMULAS
from qlib.common import read_json,sha,utc_now,write_json
from qlib.data import External
from qlib.geometry import Tile,supervision
from qlib.loss import masked_loss

ROOT=Path(__file__).resolve().parent


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Module()
        self.backbone.blocks=nn.ModuleList(nn.Linear(2,2) for _ in range(12))
        self.backbone.norm=nn.LayerNorm(2)
        self.backbone.patch_embed=nn.Linear(3,2)
        self.backbone.cls_token=nn.Parameter(torch.zeros(1,1,2))
        self.backbone.mask_token=nn.Parameter(torch.zeros(1,2))
        self.head=nn.Linear(2,4)


def main():
    receipt={}
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);Image.fromarray(np.zeros((7,9,3),np.uint8)).save(root/'image.png')
        Image.fromarray(np.zeros((7,9),np.uint8)).save(root/'zero.png')
        row={'source':'IDRiD','sid':'tiny','split':'train','excluded_duplicate':False,'image':str(root/'image.png'),
             'classes':{c:{'review_status':'unknown','mask':str(root/'zero.png')} for c in ('EX','HE','SE','MA')}}
        row['classes']['SE']['review_status']='confirmed_negative'
        manifest={'status':'READY','dedup_status':'PASS','visual_review_count':20,'samples':[row]}
        _,g,v=External('IDRiD',manifest).load('tiny')
        assert g.sum()==0 and v[:,:,2].sum()==63 and v[:,:,1].sum()==0
        target,valid=supervision(g,v,Tile(0,0,7,9))
        assert valid[2].sum()==63 and not valid[:,7:,:].any() and not valid[:,:,9:].any()
        z=torch.full((1,4,768,768),2.,requires_grad=True)
        loss=masked_loss(z,torch.from_numpy(target)[None],torch.from_numpy(valid)[None]);loss.backward()
        assert torch.all(z.grad[0,2,:7,:9]>0), 'Known negative must suppress false positives'
        assert z.grad[0,1].abs().sum()==0, 'Unknown must stay ignored'
        assert z.grad[0,:,7:,:].abs().sum()==0 and z.grad[0,:,:,9:].abs().sum()==0
        receipt['negative_gradient_unknown_padding']='PASS'
    f4,f12=Toy(),Toy();adaptation.configure(f4,'F4');adaptation.configure(f12,'F12')
    assert not any(p.requires_grad for b in f4.backbone.blocks[:8] for p in b.parameters())
    assert all(p.requires_grad for b in f12.backbone.blocks for p in b.parameters())
    assert f12.backbone.patch_embed.weight.requires_grad and f12.backbone.cls_token.requires_grad
    assert not f12.backbone.mask_token.requires_grad
    o4,_=adaptation.optimizer_for(f4);o12,_=adaptation.optimizer_for(f12)
    peaks4={g['family']:g['peak_lr'] for g in o4.param_groups}
    peaks12={g['family']:g['peak_lr'] for g in o12.param_groups}
    assert all(peaks4[name]==peaks12[name] for name in peaks4)
    assert peaks12['block0']==1e-5*.8**11 and peaks12['embedding']==1e-5*.8**12
    assert adaptation.factor(1)==.005 and adaptation.factor(200)==1 and adaptation.factor(12000)==.01
    receipt['matched_layer_lr_trainability']='PASS'
    refs=read_json(ROOT/'configs/references.json');r=refs['r_val_by_seed']['42'];q=refs['q_single_val_by_seed']['42']
    def node(single,fusion,half):
        return {f:{'miou_pct':x} for f,x in zip(FORMULAS,(single,fusion,half))}
    assert catastrophic('F4',{2000:node(q-2,r-1,r),4000:node(q-2,r-1,r)},42,refs)
    assert not catastrophic('F4',{2000:node(q-2,r-1,r),4000:node(q-1.99,r-1,r)},42,refs)
    assert not catastrophic('E_NEG',{2000:node(0,0,0),4000:node(0,0,0)},42,refs)
    assert promoted('F12',{8000:node(r,r+.5,r),12000:node(r,r+.5,r)},42,refs)==['fixed_fusion']
    assert not promoted('F12',{8000:node(r,r+.5,r),12000:node(r,r,r+.5)},42,refs)
    def run(seed,score):
        return {'stage':'COMPLETE','completed':12000,'endpoint_metrics':node(score,score,score)}
    tied={(a,s):run(s,46.) for a in ('F4','F12') for s in (42,2024)}
    selected=select(tied,refs)
    assert selected['selected']['arm']=='F4' and selected['selected']['formula']=='single' and selected['test_authorized']
    assert select({('F12',42):run(42,50.)},refs)['selected'] is None
    receipt['catastrophic_promotion_selection_and_test_gates']='PASS'
    # Byte-identical source files provide the non-treatment implementation evidence.
    historical=read_json(ROOT/'reports/historical_implementation.json')
    for rel in ('qlib/model.py','qlib/data.py','qlib/geometry.py','qlib/loss.py','qlib/optim.py','qlib/common.py'):
        assert sha(ROOT/rel)==historical[rel]
    receipt.update(status='PASS',test_pixels_read=False,gpu_training=False,created_at=utc_now())
    write_json(ROOT/'reports/cpu_acceptance.json',receipt);print(receipt)


if __name__=='__main__':main()
