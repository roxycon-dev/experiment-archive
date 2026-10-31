"""Compare adapter inference with the actual source HRDecoder methods."""
import ast
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import trainer.models.hrdecoder_adapter as adapter

def main():
    torch.set_num_threads(2)
    class Backbone(nn.Module):
        def __init__(self,**kw):super().__init__()
        def forward(self,img):
            x=F.interpolate(img[:,:1],size=(16,16),mode='bilinear',align_corners=False)
            return [x.expand(-1,c,-1,-1) for c in (48,96,192,384)]
    class Head(nn.Module):
        def __init__(self,**kw):super().__init__()
        def forward(self,features):return features[0][:,:4]
    adapter.build_reference_modules=lambda root:(Backbone,Head)
    student=adapter.HRDecoderStudent.build('unused',norm_eval=True)
    path=Path(__file__).resolve().parents[1]/'vendor/HRDecoder/mmseg/models/segmentors/HRDecoder.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HRDecoder')
    names={'encode_decode','hr_slide_inference','_resizecrop_feats','_fuse_logits','_transform_inputs'}
    funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=funcs,type_ignores=[])
    def resize(x,size,mode='bilinear',**kwargs):return F.interpolate(x,size=size,mode=mode,align_corners=False)
    def crop(x,bbox):return x[...,bbox[0]:bbox[1],bbox[2]:bbox[3]]
    ns=dict(torch=torch,resize=resize,crop=crop);exec(compile(module,str(path),'exec'),ns)
    Ref=type('SourceReference',(),{name:ns[name] for name in names})
    ref=Ref();ref.hr_scale=(1024,1024);ref.num_classes=4;ref.fuse_mode='simple';ref.detail_branch=None
    ref.extract_feat=student.backbone;ref.decode_head=student.decode_head
    ref._decode_head_forward_test=lambda features,meta:student.decode_head(features)
    image=torch.randn(1,3,2048,2048)
    with torch.no_grad():
        got=student.eval_forward(image);expected=ref.encode_decode(image,[])
    assert got.shape==(1,4,2048,2048) and torch.equal(got,expected)
    print('HR inference matches source global/local feature geometry exactly (CPU mock operators).')

if __name__=='__main__':main()
