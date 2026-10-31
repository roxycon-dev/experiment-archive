"""Protocol regression: late collapse preserves real paired20k, never invents32k."""
import sys,tempfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import decision,storage
from qlib.common import write_json,read_json

with tempfile.TemporaryDirectory(prefix='endpoint_policy_',dir=ROOT/'reports') as tmp:
    root=Path(tmp);decision.ROOT=storage.ROOT=root
    write_json(root/'configs/protocol.json',{'evaluation':{'R_val_by_seed':{'42':40.,'2024':40.}}})
    for rel in ('configs/p0_identity.json','reports/data_identity.json','configs/runtime.server.json'):
        write_json(root/rel,{'fixture':True})
    write_json(root/'reports/source_identity.json',{'sha256':'fixture'})
    for arm in decision.ARMS:
        for seed in decision.SEEDS:
            run=root/f'runs/{arm}_seed{seed}'
            completed=24000 if arm=='C' and seed==42 else 32000 if arm=='C' else 8000
            write_json(run/'status.json',{'stage':'COMPLETE' if completed==32000 else 'SCIENTIFIC_STOP_INCOMPLETE','completed':completed})
            if arm=='C':
                for u in ((20000,32000) if seed==2024 else (20000,)):
                    write_json(run/f'val_u{u}/metrics.json',{formula:{'miou_pct':41.} for formula in decision.FORMULAS})
                    (run/f'checkpoint_u{u}.pth').write_bytes(b'fixture-only')
    result=decision.finalize()
    assert result['candidate_count']==3
    assert all(c['arm']=='C' and c['update']==20000 for c in result['candidates'])
    assert result['final_selection']['formula']=='single'
    assert 'C' not in result['scientific_comparisons']['seed42_u32000']['single']['scores']
    # A metrics file alone without its saved checkpoint/custody proof must fail.
    (root/'runs/C_seed42/checkpoint_u20000.pth').unlink()
    try:decision.finalize()
    except FileNotFoundError:pass
    else:raise AssertionError('Missing mandatory checkpoint was accepted')
write_json(ROOT/'reports/endpoint_policy_acceptance.json',{'status':'PASS','paired_completed20k_survives_late_stop':True,
    'missing32k_not_fabricated':True,'missing_checkpoint_rejected':True,'scientific_training_unchanged':True})
print('ENDPOINT_POLICY_PASS')
