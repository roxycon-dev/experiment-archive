"""Separate true/pseudo loss normalization with bounded pseudo influence.

True DDR exposures, update schedule and initialization match the control.
Semi uses additional computation for auxiliary crops, disclosed explicitly.
Backpropagate the true and auxiliary micro-batches sequentially so memory
remains bounded by the already-calibrated main micro-batch.
"""
import json
from pathlib import Path
import numpy as np
import torch
from .engine import StageRunner
from .data.common import DatasetRuntime,read_image_rgb,normalize_image,sample_rng,file_sha256
from .seed import worker_init_fn
from .gate import GateError

class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self,manifest,mean,std,augment):
        self.records=[json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
        self.mean=mean;self.std=std;self.augment=augment;self.runtime=DatasetRuntime(2*1024**3)
        for r in self.records:
            if r['split']!='train' or file_sha256(r['image'])!=r['image_sha256'] or file_sha256(r['mask'])!=r['mask_sha256']:
                raise GateError('Pseudo image/mask identity or split violation')
        if len(self.records)<200:raise GateError('Pseudo pool has insufficient useful images')
    def __len__(self):return len(self.records)
    def set_epoch(self,e):self.runtime.epoch_signal.value=int(e)
    def __getitem__(self,i):
        def load():
            r=self.records[i];img=read_image_rgb(r['image'])
            with np.load(r['mask'],allow_pickle=False) as z:t=z['target'];v=z['valid']
            if t.shape!=(4,*img.shape[:2]) or v.shape!=t.shape or np.any(t>v):raise GateError('Invalid pseudo geometry/ignore semantics')
            return img,t,v
        img,t,v=self.runtime.get(i,load)
        img,t,v=self.augment(img,t,v,sample_rng(torch.initial_seed(),i,self.runtime.epoch_signal.value))
        return dict(image=torch.from_numpy(normalize_image(img,self.mean,self.std).transpose(2,0,1).copy()),
                    target=torch.from_numpy(t.copy()),valid=torch.from_numpy(v.copy()))

class SemiRunner(StageRunner):
    def __init__(self,*args,pseudo_dataset,pseudo_micro_batch=2,pseudo_weight=.25,**kwargs):
        super().__init__(*args,**kwargs)
        self.pseudo=pseudo_dataset;self.pseudo_weight=float(pseudo_weight)
        if not 0<self.pseudo_weight<=.25:raise GateError('Pseudo influence must be bounded')
        self.pseudo_loader=torch.utils.data.DataLoader(pseudo_dataset,batch_size=pseudo_micro_batch,shuffle=True,drop_last=True,
            num_workers=2,pin_memory=self._cuda(),persistent_workers=True,worker_init_fn=worker_init_fn,
            generator=torch.Generator().manual_seed(int(self.cfg['seed'])+10000))
        self.pseudo_stream=self._pseudo_batches()
        self.pseudo_crops=0
    def _pseudo_batches(self):
        epoch=self.state.epoch
        while True:
            self.pseudo.set_epoch(epoch)
            for b in self.pseudo_loader:yield b
            epoch+=1
    def train_one_update_window(self,batch_iter):
        summary=None
        for _ in range(self.grad_accum):
            true=next(batch_iter);self.state.micro_step+=1
            loss,parts=self._forward_loss(true)
            self._backward(loss/self.grad_accum);summary=parts
            del loss,true
            pseudo=next(self.pseudo_stream)
            aux,aux_parts=self._forward_loss(pseudo)
            self._backward(aux*(self.pseudo_weight/self.grad_accum))
            self.pseudo_crops+=len(pseudo['image']);del aux,pseudo
        step=self._optimizer_step()
        if step.get('skipped'):return self.train_one_update_window(batch_iter)
        if self._loss_window is not None:self._loss_window.append(float(summary['loss_total']))
        return True,dict(summary,pseudo_loss=aux_parts['loss_total'])
    def _backward(self,loss):
        if self.scaler is None:loss.backward()
        else:self.scaler.scale(loss).backward()
    def run(self,**kwargs):
        result=super().run(**kwargs)
        result.update(pseudo_crops=self.pseudo_crops,pseudo_weight=self.pseudo_weight,
            comparison='Matched initialization, true-label exposures, updates and LR. Semi uses extra auxiliary compute.')
        from .metrics import save_metrics_json
        save_metrics_json(result,str(Path(self.cfg['out_dir'])/'completion_summary.json'))
        return result
