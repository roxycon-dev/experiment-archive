"""Strict initializer, immutable run identities, and fixed paired streams."""
import os, json
from pathlib import Path
import torch
from qlib.common import read_json, sha, digest, seed_all, assert_source
from qlib.model import build_model
from qlib.data import DDR
from final_model import FinalModel, inherited_parameter_map
ROOT=Path(__file__).resolve().parent

def parent_path(runtime,seed):
    if seed==42:
        path=Path(runtime['parent_f12_execution'])/('runs' if os.name!='nt' else 'server_results/runs')/'F12_seed42/checkpoint_u12000.pth'
        expected=read_json(ROOT/'configs/protocol.json')['initialization']['seed42']['sha256']
    else:
        path=ROOT/'runs/PARENT_F12_seed2024/checkpoint_u12000.pth'
        receipt=read_json(ROOT/'reports/parent_seed2024.json')
        assert receipt['status']=='READY';expected=receipt['sha256']
    return path,expected

def load_parent(runtime,seed,q_parent=False):
    seed_all(seed)
    pretraining=read_json(ROOT/'reports/pretraining_identity.json')
    model,loading=build_model(runtime['dinov3_repo'],runtime['pretrained_checkpoint'],pretraining['sha256'])
    if q_parent:
        row=read_json(ROOT/'configs/parent_Q16k.json')[str(seed)]
        path=Path(row['local'] if os.name=='nt' else row['server']);expected=row['sha256'];completed=16000
    else:path,expected=parent_path(runtime,seed);completed=12000
    assert sha(path)==expected,'Parent checkpoint SHA mismatch'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['completed']==completed and saved['identity_sha256']==digest(saved['identity'])
    model.load_state_dict(saved['model'],strict=True)
    loading.update(parent_sha256=expected,parent_completed=completed,parent_optimizer_loaded=False)
    return model,loading

def initialize(runtime,arm,seed,device='cuda',backend='cuda'):
    parent,loading=load_parent(runtime,seed)
    mapping=inherited_parameter_map(parent)
    model=FinalModel(parent,arm,seed,backend=backend,checkpoint_blocks=runtime.get('checkpoint_blocks',False))
    loading['inherited_key_map']=mapping
    loading['explicit_new_state_keys']=[n for n in model.state_dict() if n not in set(mapping.values())]
    return model.to(device),loading

def rows_for(source,seed):
    name=f'stream_{source}_seed{seed}.jsonl';path=ROOT/'reports'/name
    identity=read_json(ROOT/'reports/stream_identity.json')[name]
    assert sha(path)==identity['sha256']
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

def next_rows(streams,completed,arm):
    return {'DDR':streams['DDR'][completed*8:completed*8+8],
            'TJDR':streams['TJDR'][completed*4:completed*4+4] if arm in ('D','SD') and completed<20000 else []}

def val_dataset(runtime,split='val',test_authorized=False):
    return DDR(runtime['ddr_root'],split,read_json(ROOT/'reports/data_identity.json')['splits'][split],test_authorized)

def identity_for(runtime_path,arm,seed,loading):
    return {'campaign':'20261005','arm':arm,'seed':seed,'endpoint':32000,
        'source_sha256':assert_source(ROOT)['sha256'],'runtime_sha256':sha(runtime_path),
        'protocol_sha256':sha(ROOT/'configs/protocol.json'),'interface_sha256':sha(ROOT/'configs/spatial_interface.md'),
        'data_identity_sha256':sha(ROOT/'reports/data_identity.json'),'TJDR_manifest_sha256':sha(ROOT/'reports/TJDR_manifest.json'),
        'streams_identity_sha256':sha(ROOT/'reports/stream_identity.json'),'p0_identity_sha256':sha(ROOT/'configs/p0_identity.json'),
        'initialization':loading}
