"""Finite geometry matrix -> immutable geometry/start lock -> no further size search."""
import subprocess
import sys
from pathlib import Path
import torch

from trainer.models.convnext_unet import ConvNeXtUNet
from trainer.models.hrdecoder_adapter import HRDecoderStudent
from tools.probability_cache import load_probability
from .common import ROOT, checked_asset, strict_load, read_json, write_json, digest, source_identity
from .evaluate import evaluate, fusion_cache, legacy_logits, counts, metrics


def load_old(settings, name):
    if name.startswith('U'):
        model = ConvNeXtUNet()
    elif name == 'H1':
        model = HRDecoderStudent.build(str(ROOT/'vendor/HRDecoder'), norm_eval=True)
    else:
        raise ValueError(name)
    strict_load(model, checked_asset(settings, name))
    return model.cuda().eval()


def matrix_entry(settings, dataset, name, geometry, mode, directory, ordinary=False):
    identity = dict(asset_sha256=settings['assets'][name]['sha256'], geometry=geometry,
                    inference=mode, H_ordinary=ordinary,
                    normalization='local_imagenet' if name.startswith('U') else 'DDR_global',
                    source_sha256=source_identity()['sha256'], split='val',
                    data_sha256=settings['campaign_data_sha256'])
    out = Path(directory)/f'{name}_{geometry}_{mode}'
    if ordinary: out = out.with_name(out.name+'_ordinary')
    if (out/'unavailable.json').exists():
        unavailable=read_json(out/'unavailable.json')
        if unavailable['identity'] != identity:
            raise ValueError('OOM record identity conflict')
        return out,unavailable
    # Evaluate only once; an exact identity cache remains reusable across restarts.
    if (out/'index.json').exists():
        if read_json(out/'index.json')['identity'] != identity:
            raise ValueError('Geometry cache conflict')
        return out, read_json(out/'metrics.json')
    model = load_old(settings, name)
    try:
        predict = lambda rgb: legacy_logits(model, rgb, geometry, mode,
                         'U' if name.startswith('U') else 'H', ordinary=ordinary)
        report = evaluate(dataset, predict, out, identity)
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():
            raise
        report = dict(status='OOM_unavailable', identity=identity,
                      note='No silent resizing or smaller-window substitution')
        write_json(out/'unavailable.json', report)
    finally:
        del model
        torch.cuda.empty_cache()
    return out, report


def run_g(settings, dataset, data_sha256):
    settings=dict(settings,campaign_data_sha256=data_sha256)
    root = ROOT/'work/G'
    root.mkdir(parents=True, exist_ok=True)
    champion = root/'champion'
    if (champion/'data_identity.json').exists() and read_json(champion/'data_identity.json')['data_sha256'] != data_sha256:
        raise ValueError('Champion cache belongs to different data')
    write_json(champion/'data_identity.json',dict(data_sha256=data_sha256))
    if not (champion/'index.json').exists():
        subprocess.run([sys.executable, str(ROOT/'tools/export_champion.py'), '--settings',
                        settings['champion_settings'], '--out', str(champion)], check=True, cwd=ROOT)
    champion_report = read_json(champion/'index.json')['metrics']
    if abs(champion_report['mIoU_lesion4']-.426691) > .00005:
        raise ValueError('Historical champion was not reproduced; do not spend training budget')
    reports, caches = {}, {}
    for name in ('U0','U1'):
        for geometry in ('native','long2048'):
            for mode in ('whole','slide'):
                key = f'{name}_{geometry}_{mode}'
                cache, report = matrix_entry(settings,dataset,name,geometry,mode,root)
                reports[key], caches[key] = report, str(cache)
                if report.get('status') != 'OOM_unavailable':
                    reports[key]['fixed_25'] = fusion_cache(dataset,champion,cache,root/(key+'_fixed25'),.75)
                write_json(root/'matrix.json', reports)
    cache, report = matrix_entry(settings,dataset,'U1','square2048','slide',root)
    reports['U1_original_square2048_slide'] = report
    for ordinary in (True,False):
        cache, report = matrix_entry(settings,dataset,'H1','square2048','whole',root,ordinary)
        reports['H1_forward' if ordinary else 'H1_eval_forward'] = report
    write_json(root/'matrix.json',reports)
    native = reports['U0_native_whole']
    if native.get('status') == 'OOM_unavailable':
        raise RuntimeError('Native whole U0 reference OOM: select larger-memory server for baseline reproduction')
    if abs(native['mIoU_lesion4']-.384469) > .00005:
        raise ValueError('Historical U0 native whole not reproduced')
    baseline = root/'current_deployment'
    b = fusion_cache(dataset,champion,caches['U0_native_whole'],baseline,.75)
    if abs(b['mIoU_lesion4']-.434877) > .00005:
        raise ValueError('Complete current 44.7809-test pipeline val not reproduced')
    # Prefer original native pixels unless a nonnative canvas has consistent evidence across both weights.
    locked_canvas = 'native'
    canvas_evidence = {}
    for name in ('U0','U1'):
        scores = {}
        for geometry in ('native','long2048'):
            available = [r['fixed_25']['mIoU_lesion4'] for key,r in reports.items()
                         if key.startswith(f'{name}_{geometry}_') and 'fixed_25' in r]
            scores[geometry] = max(available) if available else None
        canvas_evidence[name] = scores
    if all(v['long2048'] is not None and v['native'] is not None and
           v['long2048'] >= v['native']+.002 for v in canvas_evidence.values()):
        locked_canvas = 'long2048'
    available = [(mode,reports[f'U0_{locked_canvas}_{mode}']) for mode in ('whole','slide')
                 if 'fixed_25' in reports[f'U0_{locked_canvas}_{mode}']]
    mode, historical = max(available, key=lambda item:item[1]['fixed_25']['mIoU_lesion4'])
    # The highres candidate gets ONE evaluation under the now locked geometry/mode.
    high_cache, high = matrix_entry(settings,dataset,'U0_highres',locked_canvas,mode,root)
    start = 'U0'
    if high.get('status') != 'OOM_unavailable' and high['mIoU_lesion4'] > historical['mIoU_lesion4']:
        start = 'U0_highres'
    write_json(root/'highres_locked_once.json',dict(history=historical, highres=high,
               criterion='single-model fixed geometry val; tie -> historical U0', start_asset=start))
    lock = dict(canvas=locked_canvas, G_inference=mode, start_asset=start,
        start_asset_sha256=settings['assets'][start]['sha256'], baseline_cache=str(baseline),
        champion_cache=str(champion), baseline_val_miou=b['mIoU_lesion4'],
        canvas_evidence=canvas_evidence, data_sha256=data_sha256,
        source_sha256=source_identity()['sha256'], M_inference='1024 halo128 center768 single-coverage',
        current_test_reference=.447809, target=.472115, test_access='none during G/M')
    path=ROOT/'work/geometry_lock.json'
    if path.exists() and read_json(path) != lock:
        raise ValueError('Geometry lock already exists and differs; write explicit protocol revision')
    write_json(path,lock)
    return lock
