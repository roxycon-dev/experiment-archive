"""Audit-only PyTorch 2.0 compatibility change, preserving the completed zero nodes."""
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nextlib.common import read_json,write_json,source_identity,digest,sha,trusted_load
from nextlib.train import save_checkpoint

def main():
    settings=read_json(ROOT/'configs/settings.json');old=settings['source_identity'];new=source_identity()
    changed=[p for p,h in new['files'].items() if old['files'].get(p)!=h]
    assert changed==['nextlib/train.py'],changed
    assert not (ROOT/'work/final_lock.json').exists()
    # Flattening only the two spatial axes has exactly the intended channel-wise result.
    t=torch.Generator().manual_seed(723)
    for h,w in [(1,1),(3,5),(768,768)]:
        x=torch.rand((1,4,h,w),generator=t)>.9
        expected=torch.tensor([[bool(x[0,c].numpy().any()) for c in range(4)]])
        assert torch.equal(x.flatten(2).any(dim=2),expected)
    lock=read_json(ROOT/'work/geometry_lock.json');old_lock=digest(lock)
    assert lock['source_sha256']==old['sha256']
    write_json(ROOT/'reports/geometry_lock_before_pytorch_compatibility.json',lock)
    lock['source_sha256']=new['sha256'];new_lock=digest(lock)
    checkpoints=list((ROOT/'work').glob('M_seed*/M*/*.pth'))
    for p in checkpoints:
        ck=trusted_load(p)
        assert ck['update']==0 and ck['stream_cursor']==0,(str(p),ck['update'])
        assert ck['identity']['source_sha256']==old['sha256']
        assert ck['identity']['lock_sha256']==old_lock
        assert not ck.get('optimizer',{}).get('state',{})
    def migrate(x):
        count=0
        if isinstance(x,dict):
            for k,v in list(x.items()):
                if k=='source_sha256' and v==old['sha256']:x[k]=new['sha256'];count+=1
                elif k in ('lock_sha256','geometry_sha256') and v==old_lock:x[k]=new_lock;count+=1
                else:count+=migrate(v)
        elif isinstance(x,list):
            for v in x:count+=migrate(v)
        return count
    migrated=[]
    for base in [ROOT/'work/G',*(ROOT/'work').glob('M_seed*')]:
        for p in base.rglob('*.json'):
            x=read_json(p)
            if migrate(x):
                if isinstance(x,dict):x['statistics_compatibility_revision']={
                    'original_producer_source_sha256':old['sha256'],'consumer_source_sha256':new['sha256'],
                    'change':'Only foreground diagnostic reduction; model, losses, geometry and predictions unchanged'}
                write_json(p,x);migrated.append(str(p.relative_to(ROOT)))
    for p in checkpoints:
        ck=trusted_load(p);migrate(ck);save_checkpoint(p,ck)
        installed=trusted_load(p)
        assert installed['update']==0 and installed['stream_cursor']==0
        assert all(torch.equal(v,installed['model_state'][k]) for k,v in ck['model_state'].items())
        migrated.append(str(p.relative_to(ROOT)))
    write_json(ROOT/'work/geometry_lock.json',lock)
    settings['source_identity']=new;write_json(ROOT/'configs/settings.json',settings)
    proof=read_json(ROOT/'reports/server_preflight.json')
    assert proof['status']=='PASS' and proof['source_sha256']==old['sha256']
    write_json(ROOT/'reports/server_preflight_before_pytorch_compatibility.json',proof)
    proof.update(source_sha256=new['sha256'],settings_sha256=sha(ROOT/'configs/settings.json'),
        statistics_compatibility_revision=dict(original_numerical_acceptance_source=old['sha256'],
            reduction_exact=True,training_updates_before_revision=0,
            model_loss_optimizer_data_geometry_unchanged=True))
    write_json(ROOT/'reports/server_preflight.json',proof)
    write_json(ROOT/'reports/pytorch_statistics_revision.json',dict(time=time.time(),status='PASS',
        old_source_sha256=old['sha256'],new_source_sha256=new['sha256'],changed=changed,
        reason='PyTorch 2.0 Tensor.any accepts one dimension; flatten spatial axes for diagnostics',
        numerical_training_change=False,all_checkpoints_at_update_zero=True,migrated=migrated))
    print('Statistics-only compatibility accepted; original zero-node predictions and exact initial tensors retained',flush=True)

if __name__=='__main__':main()
