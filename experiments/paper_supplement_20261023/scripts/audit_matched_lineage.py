"""Verify the compact S EMA and the matched C/S checkpoint lineage, without GPU."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    run = EXP/'HRDecoder_FinalDecision_20261005/execution/server_results/runs'
    identities = {a: json.loads((run/(a+'_seed42')/'identity.json').read_text(encoding='utf-8')) for a in ['C','S']}
    keys = ['seed','protocol_sha256','data_identity_sha256','streams_identity_sha256','p0_identity_sha256']
    checks = {k: identities['C'][k] == identities['S'][k] for k in keys}
    checks['parent_sha256'] = identities['C']['initialization']['parent_sha256'] == identities['S']['initialization']['parent_sha256']
    checks['parent_completed'] = identities['C']['initialization']['parent_completed'] == identities['S']['initialization']['parent_completed'] == 12000
    assert all(checks.values()), checks
    parent = EXP/'HRDecoder_NextPlan_20261004/execution/server_results/runs/F12_seed42/checkpoint_u12000.pth'
    assert sha(parent) == identities['S']['initialization']['parent_sha256']
    compact_path = EXP/'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/S20_ema_inference.pth'
    full_path = run/'S_seed42/checkpoint_u20000.pth'
    compact = torch.load(compact_path, map_location='cpu', weights_only=False, mmap=True)
    assert sha(full_path) == compact['source_sha256']
    full = torch.load(full_path, map_location='cpu', weights_only=False, mmap=True)
    assert full['completed'] == compact['source_completed'] == 20000
    assert set(full['ema']) == set(compact['ema'])
    assert all(torch.equal(full['ema'][k], compact['ema'][k]) for k in full['ema'])
    c_path = run/'C_seed42/checkpoint_u20000.pth'
    c = torch.load(c_path,map_location='cpu',weights_only=False,mmap=True)
    assert c['completed'] == 20000
    proof = {'status':'PASS','seed':42,'comparison_update':20000,'parent_update':12000,'matched_checks':checks,
             'parent_path':str(parent),'parent_sha256':sha(parent),'S_full_path':str(full_path),'S_full_sha256':sha(full_path),
             'S_compact_path':str(compact_path),'S_compact_sha256':sha(compact_path),'S_compact_EMA_tensor_count':len(full['ema']),
             'S_compact_all_EMA_tensors_bitwise_equal':True,'C_path':str(c_path),'C_sha256':sha(c_path),
             'registered_schedule_endpoint':32000,'evaluation_node_is_not_a_shortened_schedule':True,
             'lineage':'DDR-only Q_DDR seed42 u16000 -> F12 seed42 u12000 -> C/S seed42 u20000 EMA',
             'limits':'Single parent and seed. Historical training identities, not a new rerun of optimizer/data streams.'}
    (ROOT/'evidence/matched_lineage_audit.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
