"""Relocate a NEW protocol; never edit the historical deployment lock."""
import argparse
import platform
from pathlib import Path
from nextlib.common import ROOT, read_json, write_json, digest, sha, source_identity


def configure(ddr_root, micro=4, accum=2, workers=4, existing=None):
    if (micro, accum) not in ((4,2),(2,4)):
        raise ValueError('Only predeclared paired micro4x2 / micro2x4 allowed')
    assets = read_json(ROOT/'configs/assets.json')
    resolved = {name: dict(**spec, path=str(ROOT/'assets'/spec['file'])) for name,spec in assets.items()}
    reused = {}
    if existing:
        old = Path(existing).resolve()
        candidates = {'U0': old/'assets/U0_historical_best.pth',
                      'U0_highres': old/'round2_highres/work_dirs/U0_highres/ckpt_best.pth'}
        for name, path in candidates.items():
            if path.is_file() and sha(path) == assets[name]['source_sha256']:
                reused[name] = dict(path=str(path), original_sha256=sha(path),
                    prepared_sha256=assets[name]['sha256'],
                    proof='Original source checkpoint SHA equals source of locally tensor-verified compact variant')
                resolved[name].update(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    champion = read_json(ROOT/'configs/champion_server.json')
    old_repo = champion['repo']
    champion['repo'] = str(ROOT/'vendor/champion_repo')
    champion['data_root'] = str(Path(ddr_root).resolve())
    mapping = {old_repo: champion['repo']}
    names = {'D_student_2000.pth':'D','B_weights.pth':'B','E_1500.pth':'E',
             'E4_weights.pth':'E4','H4_best.pth':'H4'}
    def relocation(value):
        if not isinstance(value, str): return value
        filename = Path(value).name
        if filename in names:
            mapping[value] = resolved[names[filename]]['path']
            return mapping[value]
        return value
    champion['student_checkpoint'] = resolved['D']['path']
    champion['reference_checkpoint'] = resolved['B']['path']
    champion['cache_dir'] = str(ROOT/'work/cache/legacy_reference')
    champion['champion']['head'] = relocation(champion['champion']['head'])
    for member in champion['champion']['members']:
        member['checkpoint'] = relocation(member['checkpoint'])
    champion['expected_sha256'] = {relocation(k):v for k,v in champion['expected_sha256'].items()}
    write_json(ROOT/'configs/champion_resolved.json', champion)
    protocol = dict(version='strict-next-20261003-v1', nodes=[0,1000,2000,4000,8000], updates=8000,
        center=768, halo=128, window=1024, global_long_side=1024, micro_batch=micro, grad_accum=accum,
        workers=workers if platform.system() != 'Windows' else 0, gradient_checkpointing=False,
        seeds=[42,2024], second_seed_gain_gate=.005, pretrained_lr=1e-5, new_lr=1e-4,
        optimizer='AdamW', weight_decay=.01, grad_clip=1., warmup_updates=400,
        schedule='cosine_min_0.01', loss='existing_fp32_BCE_plus_Dice_microbatch',
        augmented_full_global_forward=True, frozen_global_cache=True,
        frozen_global_cache_GiB=48,
        cache_policy='G_all_M_best_per_arm',
        fixed_output_fusion_weight=.5, threshold=.5, strict_greater=True,
        target_test_miou=.472115, current_test_miou=.447809,
        reference_val_miou=.434877, allow_test_for_selection=False,
        canvas_lock_rule='native unless long2048 improves BOTH U0 and U1 fixed-protocol fusion by >=0.002',
        extension='only as new matched phase with written protocol revision',
        mechanism_lowpass_control='only after M1 stable wins',
        external='official pixel GT and paired originals only; no automatic external training queue')
    settings = dict(root=str(ROOT), ddr_root=str(Path(ddr_root).resolve()), assets=resolved,
        champion_settings=str(ROOT/'configs/champion_resolved.json'),
        protocol=str(ROOT/'configs/protocol.json'), val_contract=str(ROOT/'configs/val_contract.json'),
        source_identity=source_identity(), device='cuda')
    write_json(ROOT/'configs/protocol.json', protocol)
    write_json(ROOT/'configs/settings.json', settings)
    write_json(ROOT/'reports/path_relocation.json', dict(mapping=mapping,
        reused_original_checkpoints=reused,
        original_lock_sha256=sha(ROOT/'configs/original_deployment_lock.json'),
        historical_lock_preserved=True))
    return settings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddr-root', required=True)
    parser.add_argument('--micro-batch', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefer-existing')
    a = parser.parse_args()
    configure(a.ddr_root, a.micro_batch, a.grad_accum, a.workers, a.prefer_existing)
    print('Resolved new settings and common paired protocol; no training started')
