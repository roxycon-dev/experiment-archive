"""Separate local synthetic graph checks from server exact-D/data/memory acceptance."""
import argparse
import copy
import importlib.util
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from nextlib.common import ROOT, read_json, write_json, checked_asset, source_identity, strict_load, sha, digest
from nextlib.data import DDR
from nextlib.geometry import global_view, tiles, patch_tensor, align_global, supervision, center_logits
from nextlib.model import SemanticDetailUNet, load_frozen_d
from nextlib.train import gradient_evidence
from trainer.models.convnext_unet import ConvNeXtUNet
from trainer.loss import masked_bce_dice_loss


class SyntheticGlobal(nn.Module):
    """ONLY for local tensor/gradient wiring tests; never used by the training builder."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,64,4,stride=4)
        for p in self.parameters(): p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad(): return self.conv(x)


def local_check(settings):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local = ConvNeXtUNet()
    loaded = strict_load(local,checked_asset(settings,'U0_highres'))
    branch = SyntheticGlobal()
    torch.manual_seed(43)
    m0 = SemanticDetailUNet(copy.deepcopy(local),copy.deepcopy(branch),False).to(device)
    torch.manual_seed(43)
    m1 = SemanticDetailUNet(local,branch,True).to(device)
    m0.eval(); m1.eval()
    x = torch.randn(1,3,64,64,device=device)
    g = torch.randn(1,64,16,16,device=device)
    with torch.no_grad():
        original = m1.local(x)
        p0,p1 = m0(x,g),m1(x,g)
    zero_error = max(float((original-p0).abs().max()),float((p0-p1).abs().max()))
    if zero_error != 0:
        raise ValueError(f'Zero-step output mismatch {zero_error}')
    del m0
    optimizer=torch.optim.AdamW(m1.parameter_groups(),weight_decay=.01)
    frozen={n:p.detach().clone() for n,p in m1.named_parameters() if not p.requires_grad}
    label=torch.zeros(1,4,64,64,device=device)
    for c in range(4): label[:,c,8*c:8*c+12,5*c:5*c+10]=1
    evidence=[]
    m1.train()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss,_=masked_bce_dice_loss(m1(x,g),label,torch.ones_like(label))
        loss.backward()
        evidence.append(gradient_evidence(m1,step>0))
        if step==1 and not any(p.grad is not None and p.grad.abs().sum()>0
                              for p in m1.global_adapter[:-1].parameters()):
            raise ValueError('Global adapter upstream not connected after zero projection update')
        optimizer.step()
    for n,p in m1.named_parameters():
        if n in frozen and not torch.equal(frozen[n],p):
            raise ValueError(f'Frozen parameter changed {n}')
    from trainer.models.hrdecoder_adapter import probe_reference
    dependencies=probe_reference(str(ROOT/'vendor/champion_repo'))
    report=dict(status='LOCAL_GRAPH_PASS',device=device,checkpoint=loaded,
        initialization_max_abs_error=zero_error,two_synthetic_updates=evidence,
        frozen_parameters_unchanged=True,full_D_status='SERVER_REQUIRED' if not dependencies['importable'] else 'NOT_RUN',
        full_D_dependencies=dependencies,scope='real pretrained U0, synthetic frozen global tensor wiring; no DDR training')
    write_json(ROOT/'reports/local_graph_preflight.json',report)
    print({k:v for k,v in report.items() if k not in ('checkpoint','full_D_dependencies')})


def server_check(settings):
    if not torch.cuda.is_available():
        raise RuntimeError('Server campaign requires CUDA')
    # G-only U1/H1 uploads may overlap the real M acceptance; each matrix entry
    # strictly checks its immutable checkpoint before any prediction.
    checked_assets = []
    for name in settings['assets']:
        if name in ('U1','H1') and not Path(settings['assets'][name]['path']).exists():
            continue
        checked_asset(settings,name)
        checked_assets.append(name)
    protocol=read_json(settings['protocol'])
    vc=read_json(settings['val_contract'])
    train=DDR(settings['ddr_root'],'train')
    val=DDR(settings['ddr_root'],'val',vc['stems'])
    if set(train.stems)&set(val.stems): raise ValueError('Train/val overlap')
    # Split-identity audit only: no test pixels, labels or scores are inspected.
    test_ids={p.stem for p in (Path(settings['ddr_root'])/'images/test').glob('*.jpg')}
    if test_ids and (len(test_ids)!=227 or test_ids&(set(train.stems)|set(val.stems))):
        raise ValueError('Fixed split identity separation failed')
    for sid,expected in vc['label_files_sha256'].items():
        if sha(val.root/'labels/val'/(sid+'.png')) != expected:
            raise ValueError(f'Historical validation GT hash differs {sid}')
    fp=dict(train=train.fingerprint(),val=val.fingerprint())
    write_json(ROOT/'reports/data_fingerprint.json',fp)
    native_pixels=sum(np.prod(row['hw']) for row in fp['val']['samples'])
    raw_GiB=float(native_pixels*4*4/1024**3)
    write_json(ROOT/'reports/storage_estimate.json',dict(free_GiB=shutil.disk_usage(ROOT).free/1024**3,
        one_full_val_fp32_raw_GiB=raw_GiB,
        G_up_to_14_distinct_original_coordinate_caches_raw_GiB=raw_GiB*14,
        M_best_probabilities_per_seed_raw_GiB=raw_GiB*2,
        note='Lossless deflate may reduce these values; do not assume a compression ratio. '
             'Mirror G caches if the server disk cannot retain them. No historical files are deleted.'))
    model=load_frozen_d(settings,ROOT/'reports/D_loading.json').cuda()
    # Verify feature export against the actual exact D FCN forward just before conv_seg.
    rgb,label=train.load(train.stems[0])
    view,ledger=global_view(rgb,protocol['global_long_side'],'cuda')
    captured=[]
    handle=model.head.conv_seg.register_forward_pre_hook(lambda mod,args: captured.append(args[0].detach()))
    with torch.no_grad():
        branches=model.backbone(view)
        from torch.nn import functional as F
        merged=torch.cat([b if b.shape[-2:]==branches[0].shape[-2:] else F.interpolate(
            b,branches[0].shape[-2:],mode='bilinear',align_corners=False) for b in branches],1)
        model.head([merged])
        feature=model(view)
    handle.remove()
    if not captured or not torch.equal(captured[0],feature):
        raise ValueError('Frozen D feature export differs from its trained pre-classifier tensor')
    from nextlib.global_cache import exact_global, cache_stats
    cached, cache_ledger = exact_global(model, rgb, protocol['global_long_side'], 'acceptance', .1)
    hit, _ = exact_global(model, rgb, protocol['global_long_side'], 'acceptance', .1)
    if cache_ledger != ledger or not torch.equal(feature, cached) or not torch.equal(cached, hit):
        raise ValueError('Exact FP32 frozen-global cache changes the actual full-image forward')
    from nextlib.geometry import augment_full
    rotated = augment_full(rgb, dict(rotation=1, flip_h=True, flip_v=False))
    augmented_view, augmented_ledger = global_view(rotated, protocol['global_long_side'], 'cuda')
    augmented_live = model(augmented_view)
    augmented_cached, _ = exact_global(model, rotated, protocol['global_long_side'], 'acceptance', .1)
    if not torch.equal(augmented_live, augmented_cached):
        raise ValueError('Augmented full-image cache disagrees with actual network forward')
    cache_acceptance = cache_stats()
    del model, cached, hit, feature, captured, augmented_live, augmented_cached, augmented_view
    torch.cuda.empty_cache()
    # Test the true 1024 training graph, loss and normalization before committing microbatch.
    from nextlib.model import build_fusion
    model,loaded=build_fusion(settings,'U0_highres','M1',42)
    model.cuda().train()
    tile=next(tiles(*label.shape,protocol['center'],protocol['halo']))
    x=patch_tensor(rgb,tile,'cuda')
    with torch.no_grad(): feature=model.global_branch(view)
    g=align_global(feature,ledger,tile)
    target,valid=supervision(label,tile,'cuda')
    optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler()
    torch.cuda.reset_peak_memory_stats()
    started=time.time()
    model.eval()
    with torch.no_grad():
        zero_original=model.local(x)
        zero_m1=model(x,g)
        torch.manual_seed(43)
        m0=SemanticDetailUNet(model.local,model.global_branch,False).cuda().eval()
        zero_m0=m0(x,g)
    parity_error=max(float((zero_original-zero_m1).abs().max()),float((zero_m0-zero_m1).abs().max()))
    if parity_error != 0:raise ValueError('Real D/U0 zero-step M0/M1 parity failed')
    del m0,zero_original,zero_m1,zero_m0
    bn_before={n:b.detach().clone() for n,b in model.global_branch.named_buffers()}
    model.train()
    # Two tiny acceptance updates are on the first TRAIN image only and discarded.
    # They test the fixed window/microbatch, never select a model or a val setting.
    checks=[]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(protocol['grad_accum']):
            with torch.cuda.amp.autocast():
                logits=model(x.repeat(protocol['micro_batch'],1,1,1),g.repeat(protocol['micro_batch'],1,1,1))
                logits=center_logits(logits,tile)
            loss,_=masked_bce_dice_loss(logits,target.repeat(protocol['micro_batch'],1,1,1),
                                       valid.repeat(protocol['micro_batch'],1,1,1))
            scaler.scale(loss/protocol['grad_accum']).backward()
        scaler.unscale_(optimizer)
        checks.append(gradient_evidence(model,step>0))
        torch.nn.utils.clip_grad_norm_(list(p for p in model.parameters() if p.requires_grad),1.)
        scaler.step(optimizer);scaler.update()
    if any(not torch.equal(old,dict(model.global_branch.named_buffers())[n]) for n,old in bn_before.items()):
        raise ValueError('Frozen D normalization buffers changed')
    report=dict(status='PASS',source_sha256=source_identity()['sha256'],data_sha256=digest(fp),
        settings_sha256=sha(ROOT/'configs/settings.json'),protocol_sha256=digest(protocol),
        D_exact_load=True,D_preclassifier_export_equal=True,zero_step_parity_max_error=parity_error,
        checked_assets=checked_assets,G_weights_strictly_checked_on_use=True,
        frozen_global_cache_exact=True,frozen_global_cache_acceptance=cache_acceptance,
        D_buffers_unchanged=True,
        true_window=[protocol['window'],protocol['window']], micro_batch=protocol['micro_batch'],
        grad_accum=protocol['grad_accum'],gradient_checks=checks,
        gpu=torch.cuda.get_device_name(0),peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
        acceptance_seconds=time.time()-started,environment=dict(python=sys.version,torch=torch.__version__),
        note='Acceptance updates discarded; formal M always reloads immutable historical tensors')
    write_json(ROOT/'reports/server_preflight.json',report)
    print(report)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--server',action='store_true');a=ap.parse_args()
    settings=read_json(ROOT/'configs/settings.json')
    if a.server: server_check(settings)
    else: local_check(settings)
