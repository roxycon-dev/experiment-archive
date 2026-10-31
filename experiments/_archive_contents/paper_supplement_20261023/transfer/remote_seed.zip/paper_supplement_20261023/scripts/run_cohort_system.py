"""Frozen IDRiD holdout and full-only profiling; memory scheduling does not alter formulas."""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from legacy_canary import build_legacy

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
READY = EXP / 'HRDecoder_Ready_20261002'
BASEPY = Path('C:/Users/GuanXingRen/Miniconda3/python.exe')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def write(p, value):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(str(temp), str(p))


def counts(binary, gt):
    return np.stack([(binary & gt).sum((1, 2)), (binary & ~gt).sum((1, 2)),
                     (~binary & gt).sum((1, 2))], axis=1).astype(np.int64).tolist()


def move_champion(champion, device):
    for model, _, _ in champion.members: model.to(device)
    for head in champion.heads.values(): head.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['external', 'profile'], required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--max-estimated-seconds', type=float, default=3600)
    args = parser.parse_args()
    assert all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.run_name)
    formal = ROOT / 'results/ddr_test_local_v1'
    assert (formal / 'completion.json').exists(), 'Do not contend with the active DDR run'
    assert not (formal / 'failure.json').exists()
    out = ROOT / 'results' / args.run_name
    out.mkdir(exist_ok=False)
    for directory in ['per_image', 'predictions', 'exchange']: (out / directory).mkdir()
    exchange = out / 'exchange'
    if args.mode == 'external':
        source = ROOT / 'evidence/external_holdout_manifest.json'
        cohort = json.loads(source.read_text(encoding='utf-8'))
        assert cohort['status'] == 'FROZEN_BEFORE_EXTERNAL_PREDICTIONS'
        samples = cohort['samples']
        assert [r['sid'] for r in samples] == ['IDRiD_%02d' % i for i in range(55, 82)]
    else:
        source = ROOT / 'evidence/local_asset_audit.json'
        audit = json.loads(source.read_text(encoding='utf-8'))
        samples = []
        for row in audit['data']['val']['samples'][:3]:
            image = Path('D:/HRDecoder/data/DDR_split_613/images/val') / (row['sid'] + '.jpg')
            with Image.open(image) as im: w, h = im.size
            samples.append({**row, 'image_path': str(image), 'hw': [h, w]})
        cohort = {'samples': samples, 'scope': 'first three validation images; no selection by outcome'}
    cohort_path = out / 'cohort.json'; write(cohort_path, cohort)
    ddr = json.loads((formal / 'manifest.json').read_text(encoding='utf-8'))
    for asset in ddr['assets'].values(): assert sha(asset['path']) == asset['sha256']
    manifest = {'run': args.run_name, 'mode': args.mode, 'expected_images': len(samples),
                'sample_ids': [r['sid'] for r in samples], 'source_manifest_sha256': sha(source),
                'cohort_sha256': sha(cohort_path), 'assets': ddr['assets'], 'threshold': .5,
                'strict_greater': True, 'new_training': False, 'historical_exact_replay': False,
                'scope': 'independent_heldout_subset' if args.mode == 'external' else 'full_system_timing_only',
                'scripts': {p.name: sha(p) for p in [Path(__file__), ROOT/'scripts/cohort_worker.py', ROOT/'scripts/legacy_canary.py']},
                'runtime_amendment_sha256': sha(ROOT/'plan/runtime_amendment.md'),
                'external_protocol_sha256': sha(ROOT/'plan/external_protocol.md'),
                'memory_schedule': 'A then U then S/C; parameters on CPU while idle; unchanged precision/geometry',
                'full_formula': '.75*(.75*A+.25*U)+.25*S', 'hardware': torch.cuda.get_device_name()}
    write(out / 'manifest.json', manifest)
    runtime, champion, dataset = build_legacy('val')
    if args.mode == 'external':
        from mmseg.datasets import build_dataset
        cfg = runtime.load_config(str(READY/'vendor/champion_repo'), 'configs/lesion/exp/hrplus_patch_dice.py')
        d = copy.deepcopy(cfg.data['val']); d.data_root = 'D:/dataset/best_segmentation'
        d.img_dir = 'images/testing'; d.ann_dir = 'labels/testing'
        d.pipeline = copy.deepcopy(cfg.data['test']['pipeline']); d.test_mode = True
        if 'split' in d: d.split = None
        dataset = build_dataset(d)
    lookup = {Path(r['filename']).stem: i for i, r in enumerate(dataset.img_infos)}
    sys.path.insert(0, str(READY))
    from trainer.models.convnext_unet import ConvNeXtUNet
    from trainer.eval import pad_normalized_image
    from trainer.data.common import normalize_image
    u = ConvNeXtUNet()
    saved = torch.load(ddr['assets']['U0']['path'], map_location='cpu')
    u.load_state_dict(saved['model_state'], strict=True); del saved
    u.eval().requires_grad_(False)
    parameters = {spec['name']: sum(p.numel() for p in model.parameters()) for model, _, spec in champion.members}
    parameters['H4'] = sum(p.numel() for p in champion.heads['MA'].parameters())
    parameters['U0'] = sum(p.numel() for p in u.parameters())
    log = (out / 'modern_worker.log').open('w', encoding='utf-8')
    worker = subprocess.Popen([str(BASEPY), '-u', str(ROOT/'scripts/cohort_worker.py'), '--exchange', str(exchange),
                               '--manifest', str(cohort_path), '--arms', 'SC' if args.mode == 'external' else 'S'],
                              stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000 if os.name == 'nt' else 0)
    started = time.perf_counter()
    monitor = None; monitor_log = None
    try:
        while not (exchange/'worker_ready.json').exists():
            if worker.poll() is not None: raise RuntimeError('Worker startup failed: see log')
            if time.perf_counter()-started > 240: raise TimeoutError('Worker startup')
            time.sleep(.2)
        ready = json.loads((exchange/'worker_ready.json').read_text(encoding='utf-8'))
        parameters.update(ready['registered_parameters'])
        if args.mode == 'profile':
            monitor_log = (out/'GPU_memory_samples.csv').open('w',encoding='utf-8')
            monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used',
                                        '--format=csv,noheader,nounits','--loop-ms=200'],
                                       stdout=monitor_log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        write(out / 'parameters.json', {'registered_parameters_per_instantiated_module': parameters,
              'full_system_total_registered_parameters': sum(v for k, v in parameters.items() if k != 'C'),
              'note': 'Includes registered auxiliary modules and separately instantiated reference copies; not FLOPs or unique learned scalar count',
              'weight_artifact_bytes_full': sum(v['bytes'] if 'bytes' in v else Path(v['path']).stat().st_size for k, v in ddr['assets'].items() if k != 'C20')})
        # Profile warmup forwards are retained as explicitly marked records and excluded from reported latency.
        jobs = [samples[0]] + samples if args.mode == 'profile' else samples
        for i, row in enumerate(jobs):
            sid = row['sid']; warmup = args.mode == 'profile' and i == 0
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            image_start = time.perf_counter()
            image_path = Path(row['image_path'])
            assert sha(image_path) == row['image_sha256']
            rgb = np.asarray(Image.open(image_path).convert('RGB')); h, w = rgb.shape[:2]
            item = dataset[lookup[sid]]
            meta = runtime._unwrap_meta(item['img_metas'])
            image = runtime._unwrap_img(item['img']).unsqueeze(0).cuda()
            read_transform_seconds = time.perf_counter()-image_start
            a_start = time.perf_counter(); move_champion(champion, 'cuda')
            with torch.no_grad():
                pre, ps, pr = champion.pre_head_probs(image, meta)
                imgf = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)[0]
                a = pre.clone(); a[3] = champion.corrected_class_prob(3, pre, ps, pr, imgf, champion.heads['MA'])
            a, pre = a.cpu().numpy(), pre.cpu().numpy()
            del ps, pr, imgf, image
            move_champion(champion, 'cpu'); torch.cuda.empty_cache(); torch.cuda.synchronize()
            a_seconds = time.perf_counter()-a_start
            a_peak = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats(); u_start = time.perf_counter()
            chw = normalize_image(rgb, [123.675,116.28,103.53], [58.395,57.12,57.375]).transpose(2,0,1)
            padded, _ = pad_normalized_image(np.ascontiguousarray(chw))
            u.cuda()
            with torch.no_grad():
                x = torch.from_numpy(padded[None]).cuda().float()
                pu = u.eval_forward(x)[...,:h,:w].float().sigmoid()[0].cpu().numpy()
            del x, chw, padded
            u.cpu(); torch.cuda.synchronize()
            u_seconds = time.perf_counter()-u_start; u_peak = torch.cuda.max_memory_allocated()
            parent_reserved = torch.cuda.max_memory_reserved(); torch.cuda.empty_cache()
            write(exchange/'request.json', {'sid': sid})
            wait = time.monotonic()
            while not (exchange/'response.json').exists():
                if worker.poll() is not None: raise RuntimeError('Worker failed: see log')
                if time.monotonic()-wait > 600: raise TimeoutError('Modern inference')
                time.sleep(.1)
            response = json.loads((exchange/'response.json').read_text(encoding='utf-8'))
            assert response['sid'] == sid
            s = np.load(str(exchange/'S.npy'), allow_pickle=False)
            for prob in [a, pre, pu, s]:
                assert prob.dtype == np.float32 and prob.shape == (4,h,w) and np.isfinite(prob).all()
            del prob
            p0 = .75*a+.25*pu
            masks = {'full': (.75*p0+.25*s) > .5}
            if args.mode == 'external':
                masks['no_h4'] = (.75*(.75*pre+.25*pu)+.25*s) > .5
                masks['no_convnext'] = (np.float32(9/13)*a+np.float32(4/13)*s) > .5
            del s, pre, pu, a
            if args.mode == 'external':
                c = np.load(str(exchange/'C.npy'), allow_pickle=False)
                assert c.dtype == np.float32 and c.shape == (4,h,w) and np.isfinite(c).all()
                masks['matched_c20'] = (.75*p0+.25*c) > .5
                del c
                assert np.array_equal(masks['full'][:3], masks['no_h4'][:3])
            del p0
            torch.cuda.synchronize()
            full_pipeline_seconds = time.perf_counter()-image_start
            counts_start = time.perf_counter()
            if args.mode == 'external':
                gt = []
                for name in ['EX','HE','SE','MA']:
                    mr = row['masks'][name]; assert sha(mr['path']) == mr['sha256']
                    gt.append(np.asarray(Image.open(mr['path'])) > 0)
                gt = np.stack(gt)
            else:
                label = np.asarray(Image.open(Path('D:/HRDecoder/data/DDR_split_613/labels/val')/(sid+'.png')))
                gt = np.stack([label == c for c in range(1,5)])
                assert not (label == 255).any(), 'Profile count audit assumes all pixels valid'
                del label
            row_counts = {k: counts(b, gt) for k,b in masks.items()}
            name = ('warmup_' if warmup else '') + sid
            target = out/'predictions'/(name+'.npz')
            with target.with_suffix('.tmp').open('wb') as f: np.savez_compressed(f, **masks)
            os.replace(str(target.with_suffix('.tmp')), str(target))
            record = {'sid': sid, 'warmup': warmup, 'hw': [h,w], 'counts': row_counts, 'image_sha256': row['image_sha256'],
                      'mask_sha256': sha(target), 'prediction_path': str(target),
                      'timing': {'image_read_and_transform_seconds': read_transform_seconds, 'A_including_transfer_seconds': a_seconds,
                                 'U_including_transfer_seconds': u_seconds, 'modern': response['timing'],
                                 'pipeline_to_fused_mask_seconds': full_pipeline_seconds,
                                 'GT_counts_and_mask_persistence_seconds': time.perf_counter()-counts_start,
                                 'pipeline_through_persistence_seconds': time.perf_counter()-image_start},
                      'GPU_memory': {'A_parent_peak_allocated': a_peak, 'U_parent_peak_allocated': u_peak,
                                     'parent_peak_reserved': parent_reserved,
                                     'note': 'Per-process allocated/reserved. Parent weights CPU before worker CUDA; CUDA contexts/driver not included.'}}
            write(out/'per_image'/(name+'.json'), record)
            for fn in ['response.json', 'S.npy'] + (['C.npy'] if args.mode == 'external' else []):
                p = (exchange/fn).resolve(); assert p.parent == exchange.resolve(); p.unlink()
            del masks, gt, rgb
            progress = {'stage': 'RUNNING', 'mode': args.mode, 'completed': i+1, 'expected': len(jobs),
                        'last_sid': sid, 'warmup': warmup, 'elapsed_seconds': time.perf_counter()-started}
            write(out/'progress.json', progress); print(json.dumps(progress), flush=True)
            if args.mode == 'external' and i == 0:
                # Conservative prospective size-based estimate, without consulting segmentation scores.
                area_factor = sum(r['hw'][0]*r['hw'][1] for r in samples)/(h*w)
                estimate = record['timing']['pipeline_through_persistence_seconds'] * area_factor * 1.20
                write(out/'resource_estimate.json', {'first_sid': sid, 'estimated_batch_seconds_with_20pct_margin': estimate,
                                                   'rule': 'first-image total wall time multiplied by total area / first area, plus 20%; no scores consulted',
                                                   'local_limit_seconds': args.max_estimated_seconds})
                if estimate > args.max_estimated_seconds:
                    raise RuntimeError('Estimated external evaluation exceeds user local time budget; remaining images require server')
        write(out/'completion.json', {'status': 'COMPLETE', 'images': len(samples), 'warmup_forwards': int(args.mode == 'profile'),
                                      'seconds': time.perf_counter()-started, 'all_predeclared_rows_retained': True})
    except BaseException as error:
        write(out/'failure.json', {'type': type(error).__name__, 'message': str(error), 'formal_results_complete': False})
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            try: monitor.wait(timeout=5)
            except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()
        if monitor_log is not None: monitor_log.close()
        (exchange/'STOP').write_text('finish', encoding='ascii')
        try: worker.wait(timeout=15)
        except subprocess.TimeoutExpired: worker.terminate()
        log.close()


if __name__ == '__main__': main()
