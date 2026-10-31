"""Exact DDR mean + .25 source mean; one optimizer step then full-model EMA."""
import time
import torch
from qlib.geometry import center_logits
from qlib.loss import masked_loss
from random_streams import forward_rng

def update(model,optimizer,scheduler,ema,loaders,seed,completed,audits=None):
    optimizer.zero_grad(set_to_none=True)
    started=time.perf_counter();wait=0.;loss_values=[]
    sources=[('DDR',4,1.)]
    if model.has_source and completed<20000:sources.append(('TJDR',2,.25))
    for source,micros,weight in sources:
        for micro in range(micros):
            before=time.perf_counter();batch=next(loaders[source]);wait+=time.perf_counter()-before
            x,g,v,legal,positive=batch
            if audits is not None:
                audit=audits.setdefault(source,{'windows':0,'valid_pixels':[0]*4,'positive_pixels':[0]*4})
                audit['windows']+=len(x)
                for key,tensor in (('valid_pixels',legal),('positive_pixels',positive)):
                    audit[key]=[a+int(b) for a,b in zip(audit[key],tensor.sum(0).tolist())]
            x,g,v=(tensor.to('cuda',non_blocking=True) for tensor in (x,g,v))
            with forward_rng(seed,source,completed+1,micro):
                with torch.autocast('cuda',dtype=torch.bfloat16):z=center_logits(model(x,source))
                loss=masked_loss(z,g,v)*(weight/micros)
                loss.backward()
            loss_values.append(loss.detach())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    # One necessary host check prevents committing a nonfinite update. Loss
    # transfers are combined, rather than synchronizing every microbatch.
    report=torch.stack((torch.stack(loss_values).sum(),norm)).float().cpu().tolist()
    if not all(__import__('math').isfinite(x) for x in report):raise FloatingPointError('Nonfinite loss/gradient; update not committed')
    optimizer.step();scheduler.step();ema.update(model)
    return {'loss':report[0],'grad_norm':report[1],'seconds':time.perf_counter()-started,'data_wait_seconds':wait}
