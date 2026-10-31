"""CPU serialization/resume check for the real SACD readout and sole SGD state."""
import io
import json
from pathlib import Path
import random
import numpy as np
import torch
from model import CorrectionHead
from objectives import transport_logits
from loader import ResumableBatches
from train import rng_state, restore_rng

assert not torch.cuda.is_available(), 'This check must stay on CPU'
torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);np.random.seed(42)
head=CorrectionHead();optimizer=torch.optim.SGD(head.parameters(),lr=.005,momentum=.9,weight_decay=.0005)
def step(module,opt):
    opt.zero_grad(set_to_none=True)
    f=torch.randn(2,32,4,4);z=torch.randn(2,4,16,16)
    actions=module(f,f,z.sigmoid())
    logits,_=transport_logits(z,actions)
    target=torch.rand_like(logits)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    loss.backward();opt.step()
    return float(loss)
step(head,optimizer)
buffer=io.BytesIO()
torch.save({'model':head.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state()},buffer)
expected=step(head,optimizer)
expected_state={k:v.clone() for k,v in head.state_dict().items()}
buffer.seek(0);checkpoint=torch.load(buffer,map_location='cpu',weights_only=False)
restored=CorrectionHead();own=torch.optim.SGD(restored.parameters(),lr=.005,momentum=.9,weight_decay=.0005)
restored.load_state_dict(checkpoint['model']);own.load_state_dict(checkpoint['optimizer']);restore_rng(checkpoint['rng'])
actual=step(restored,own)
assert actual==expected
assert all(torch.equal(v,expected_state[k]) for k,v in restored.state_dict().items())
whole=list(ResumableBatches(454,0,230))
assert list(ResumableBatches(454,226,230))==whole[226:230]
report={'status':'PASS','torch':torch.__version__,'device':'CPU','SGD_momentum_resume_bitwise_equal':True,
        'RNG_restored':True,'exposure_resume_crosses_epoch_boundary':True,'formal_updates':0}
(Path(__file__).resolve().parent/'CHECKPOINT_RESUME_PASS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
