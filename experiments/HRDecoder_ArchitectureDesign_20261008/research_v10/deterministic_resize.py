"""Bilinear forward unchanged; deterministic separable transpose for backward."""
from functools import lru_cache
import torch
import torch.nn.functional as F

_original=None


@lru_cache(maxsize=32)
def weights(n,m,dtype):
    # Each output samples two input coordinates. Construct on CPU, where
    # repeated endpoint weights are deterministic, then copy the small matrix.
    t=(torch.arange(m,dtype=dtype)+.5)*(n/m)-.5
    t=t.clamp_min(0);lo=t.floor().long();hi=(lo+1).clamp_max(n-1)
    frac=t-lo.to(dtype)
    a=F.one_hot(lo,num_classes=n).to(dtype)*(1-frac[:,None])
    b=F.one_hot(hi,num_classes=n).to(dtype)*frac[:,None]
    return a+b


class Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,height,width):
        ctx.input_hw=x.shape[-2:];ctx.output_hw=(height,width);ctx.dtype=x.dtype
        v=x.float() if x.dtype==torch.bfloat16 else x
        return _original(v,size=(height,width),mode='bilinear',align_corners=False).to(x.dtype)

    @staticmethod
    def backward(ctx,gradient):
        ih,iw=ctx.input_hw;oh,ow=ctx.output_hw
        dtype=torch.float64 if ctx.dtype==torch.float64 else torch.float32
        h=weights(ih,oh,dtype).to(gradient.device);w=weights(iw,ow,dtype).to(gradient.device)
        # Matrix multiplication reduces each source location in a fixed order;
        # CUDA bilinear atomic scatter-add is not used.
        with torch.autocast(gradient.device.type,enabled=False):
            result=torch.matmul(torch.matmul(h.T,gradient.to(dtype)),w)
        return result.to(ctx.dtype),None,None


def install():
    global _original
    if getattr(F.interpolate,'_hrdecoder_deterministic_resize',False):return
    _original=F.interpolate
    def interpolate(input,size=None,scale_factor=None,mode='nearest',align_corners=None,
                    recompute_scale_factor=None,antialias=False):
        if input.requires_grad and mode=='bilinear' and size is not None:
            assert input.ndim==4 and align_corners is False and not antialias
            assert scale_factor is None and recompute_scale_factor is None
            hw=(size,size) if isinstance(size,int) else tuple(size)
            return Bilinear.apply(input,*hw)
        return _original(input,size=size,scale_factor=scale_factor,mode=mode,
                         align_corners=align_corners,recompute_scale_factor=recompute_scale_factor,antialias=antialias)
    interpolate._hrdecoder_deterministic_resize=True
    F.interpolate=interpolate
