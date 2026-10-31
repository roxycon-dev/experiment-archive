"""Numerical transpose-gradient and CUDA repeatability checks; no real data."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import deterministic_resize as op

torch.set_num_threads(4)
original=F.interpolate;op.install();tests=[]
device='cuda' if '--cuda' in sys.argv else 'cpu'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
for dtype in (torch.float64,torch.float32):
 for ih,iw,oh,ow in [(3,5,7,9),(8,6,4,3),(4,4,8,8),(1,2,3,5),(7,9,7,9)]:
  generator=torch.Generator().manual_seed(61008)
  x=torch.randn(2,3,ih,iw,generator=generator,dtype=dtype).requires_grad_()
  dy=torch.randn(2,3,oh,ow,generator=generator,dtype=dtype)
  reference=original(x,size=(oh,ow),mode='bilinear',align_corners=False)
  reference.backward(dy);expected=x.grad.clone()
  gradients=[]
  for repeat in range(2):
   actual=x.detach().to(device).requires_grad_()
   result=F.interpolate(actual,size=(oh,ow),mode='bilinear',align_corners=False)
   result.backward(dy.to(device));gradients.append(actual.grad.cpu())
   tolerance=1e-12 if dtype==torch.float64 else 3e-6
   assert torch.allclose(result.cpu(),reference,atol=tolerance,rtol=tolerance)
   assert torch.allclose(actual.grad.cpu(),expected,atol=tolerance,rtol=tolerance)
  assert torch.equal(*gradients)
  tests.append({'dtype':str(dtype),'in_hw':[ih,iw],'out_hw':[oh,ow],
                'maximum_gradient_error':float((gradients[0]-expected).abs().max())})
if device=='cuda':
 x=torch.randn(2,3,8,8,device='cuda',dtype=torch.bfloat16).requires_grad_()
 z=F.interpolate(x,size=(16,16),mode='bilinear',align_corners=False);z.float().square().mean().backward()
 assert torch.isfinite(x.grad).all()
report={'status':'PASS','torch':torch.__version__,'device':device,'tests':tests,
        'deterministic_algorithms':True,'TF32':False,'no_data_or_checkpoint_read':True}
(Path(__file__).parent/('RESIZE_'+device.upper()+'_PASS.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
