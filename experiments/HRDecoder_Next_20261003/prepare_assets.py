"""Local-only preparation: isolate code, preserve originals, strip only unused optimizer state."""
import argparse
import shutil
from pathlib import Path
import torch
from nextlib.common import ROOT, sha, trusted_load, state_from, write_json, read_json


def main():
    old = ROOT.parent/'HRDecoder_Ready_20261002'
    manifest = read_json(old/'best_pipeline/local_weight_manifest.json')
    assets = {}
    names = dict(D='D_student_2000.pth', B='B_weights.pth', E='E_1500.pth',
                 E4='E4_weights.pth', H4='H4_best.pth', U0='U0_historical_best.pth')
    for name, file in names.items():
        spec = manifest['weights'][file]
        source = Path(spec['path'])
        actual = sha(source)
        if actual != spec['sha256']:
            raise ValueError(f'Historical asset hash mismatch {name}')
        dest = ROOT/'assets'/file
        if name == 'U0':
            obj = trusted_load(source)
            state = state_from(obj)
            if not dest.exists():
                torch.save(dict(model_state=state, update=obj.get('update'), source_sha256=actual), dest)
            restored = state_from(trusted_load(dest))
            if set(restored) != set(state) or not all(torch.equal(state[k], restored[k]) for k in state):
                raise ValueError('Optimizer stripping changed model tensors')
        elif not dest.exists():
            shutil.copy2(source, dest)
        elif sha(dest) != actual:
            raise ValueError('Existing prepared asset has wrong SHA')
        assets[name] = dict(file=dest.name, sha256=sha(dest), bytes=dest.stat().st_size,
            source=str(source), source_sha256=actual, model_tensors_unchanged=True)
    extra = {'U1': old/'server_results/work_dirs/U1_ddr_finetune/ckpt_best.pth',
             'H1': old/'server_results/work_dirs/H1_ddr_finetune/ckpt_best.pth',
             'U0_highres': old/'round2_highres/server_results/work_dirs/U0_highres/ckpt_best.pth'}
    expected_update = dict(U1=12000, H1=9000, U0_highres=1000)
    for name, source in extra.items():
        actual = sha(source)
        ck = trusted_load(source)
        if ck['update'] != expected_update[name] or ck.get('is_calibration', False):
            raise ValueError('Wrong formal historical node')
        state = state_from(ck)
        dest = ROOT/'assets'/(name+'.pth')
        if not dest.exists():
            torch.save(dict(model_state=state, update=ck['update'], source_sha256=actual,
                            source_config_sha256=ck['config_sha256']), dest)
        new = state_from(trusted_load(dest))
        if set(new) != set(state) or not all(torch.equal(v, new[k]) for k,v in state.items()):
            raise ValueError('Prepared model tensors changed')
        assets[name] = dict(file=dest.name, sha256=sha(dest), bytes=dest.stat().st_size,
            source=str(source), source_sha256=actual, update=ck['update'],
            source_config_sha256=ck['config_sha256'], model_tensors_unchanged=True)
    write_json(ROOT/'configs/assets.json', assets)
    print({name: round(spec['bytes']/1024**2, 1) for name,spec in assets.items()})


if __name__ == '__main__':
    main()
