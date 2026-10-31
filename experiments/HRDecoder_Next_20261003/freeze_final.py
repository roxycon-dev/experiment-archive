"""Freeze the selected complete pipeline from validation only, before accessing test."""
import argparse
from pathlib import Path
from nextlib.common import ROOT,read_json,write_json,sha,digest,source_identity


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--arm',choices=['M0','M1'],required=True);a=ap.parse_args()
    settings=read_json(ROOT/'configs/settings.json');geometry=read_json(ROOT/'work/geometry_lock.json')
    run=ROOT/f'work/M_seed{a.seed}'/a.arm
    selection=read_json(run/'history.json')
    from nextlib.decision import best_candidate
    best=best_candidate(selection)
    if best['score']<=geometry['baseline_val_miou']:
        raise ValueError('Selected new pipeline did not beat the retained complete old pipeline on val')
    checkpoint=run/'best.pth'
    lock=dict(version='final-20261003-v1',basis='validation_only',seed=a.seed,arm=a.arm,
        output=best['output'],update=best['update'],val_miou=best['score'],
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        geometry=geometry,geometry_sha256=digest(geometry),
        protocol_sha256=sha(ROOT/'configs/protocol.json'),
        champion_settings=str(ROOT/'configs/champion_resolved.json'),
        champion_settings_sha256=sha(ROOT/'configs/champion_resolved.json'),
        assets={k:dict(path=v['path'],sha256=v['sha256']) for k,v in settings['assets'].items()},
        ddr_root=settings['ddr_root'],source_sha256=source_identity()['sha256'],
        threshold=.5,strict_greater=True,old_pipeline='0.75*champion+0.25*historical_U0',
        new_pipeline='M solo' if best['output']=='solo' else '0.5*old_complete_pipeline+0.5*M',
        historical_test_note='227-image test has been inspected in past rounds; exploratory continuation')
    path=ROOT/'work/final_lock.json'
    if (ROOT/'work/test_progress.json').exists():raise ValueError('Test already started; final lock cannot change')
    if path.exists() and read_json(path)!=lock:raise ValueError('Existing final lock differs; no silent replacement')
    write_json(path,lock)
    print(path)


if __name__=='__main__':main()
