"""Serial server queue. No connection, credentials, deletion or shutdown logic."""
import argparse
import sys
from pathlib import Path
import torch

from nextlib.common import ROOT, read_json, write_json, source_identity, digest
from nextlib.data import DDR, make_stream
from nextlib.g_stage import run_g
from nextlib.train import run_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['G','M','all'], default='G')
    a = ap.parse_args()
    settings = read_json(ROOT/'configs/settings.json')
    protocol = read_json(settings['protocol'])
    preflight = read_json(ROOT/'reports/server_preflight.json')
    if (preflight['status'] != 'PASS' or preflight['source_sha256'] != source_identity()['sha256'] or
            preflight['protocol_sha256'] != digest(protocol)):
        raise ValueError('Current code requires a passing server preflight')
    train = DDR(settings['ddr_root'],'train')
    val = DDR(settings['ddr_root'],'val',read_json(settings['val_contract'])['stems'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    # G reference is FP32 deterministic; new M performance mode is separately declared.
    if a.stage in ('G','all'):
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        lock = run_g(settings,val,preflight['data_sha256'])
    else:
        lock = read_json(ROOT/'work/geometry_lock.json')
    if lock['data_sha256'] != preflight['data_sha256'] or lock['source_sha256'] != source_identity()['sha256']:
        raise ValueError('Data or code changed after geometry lock')
    if a.stage == 'G':
        print('Finite G complete. Geometry and shared start locked.')
        return
    # A calibration-only gain can be examined before M using analyze_cache.py;
    # no oracle or full-val fitted threshold is silently substituted for 0.5.
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    results=[]
    for seed in protocol['seeds']:
        stream=make_stream(train,ROOT/f'work/stream_seed{seed}.json',seed,lock['canvas'],
                           protocol['updates'],8,protocol['center'],protocol['halo'])
        gate,winner=run_pair(settings,protocol,stream,train,val,lock,seed,
                            ROOT/f'work/M_seed{seed}',lock['baseline_cache'])
        results.append(dict(seed=seed,winner=winner,second_seed_gate=gate))
        write_json(ROOT/'work/M_selections.json',results)
        if not gate:
            break
    write_json(ROOT/'work/campaign_complete.json',dict(status='G_M_complete',results=results,
        E='conditional_assets_branch_not_launched',test_evaluation='not_run',
        next='Inspect cross-seed per-image evidence, freeze output recipe, then final test once'))
    print('G/M queue complete. Selections saved; final test remains locked behind final recipe.')


if __name__ == '__main__':
    main()
