"""Frozen four-row reinference in existing local environments; no optimization."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from legacy_canary import build_legacy

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
DATA = Path('D:/HRDecoder/data/DDR_split_613')
BASEPY = Path('C:/Users/GuanXingRen/Miniconda3/python.exe')
READY = EXP / 'HRDecoder_Ready_20261002'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write_json(p, value):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(str(tmp), str(p))


def pixel_counts(p, label):
    if p.dtype != np.float32 or not np.isfinite(p).all():
        raise ValueError('Probability contract failed')
    g = np.stack([label == c for c in range(1, 5)])
    v = label != 255
    b = p > .5
    return np.stack([(b & g & v).sum((1,2)), (b & ~g & v).sum((1,2)), (~b & g & v).sum((1,2))], axis=1).astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--limit', type=int, default=0, help='Compatibility smoke only when nonzero')
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.run_name):
        raise ValueError('Invalid run directory')
    out = ROOT / 'results' / args.run_name
    out.mkdir(exist_ok=True)
    (out / 'per_image').mkdir(exist_ok=True)
    (out / 'predictions').mkdir(exist_ok=True)
    exchange = out / 'exchange'
    exchange.mkdir(exist_ok=True)
    if any(exchange.iterdir()):
        raise RuntimeError('Nonempty exchange requires explicit diagnosed recovery')
    assets = {
        name: READY / 'best_pipeline/weights' / name for name in [
            'B_weights.pth', 'D_student_2000.pth', 'E_1500.pth', 'E4_weights.pth', 'H4_best.pth']}
    assets.update(U0=EXP / 'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/U0_inference.pth',
                  S20=EXP / 'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/S20_ema_inference.pth',
                  C20=EXP / 'HRDecoder_FinalDecision_20261005/execution/server_results/runs/C_seed42/checkpoint_u20000.pth')
    identity = json.loads((ROOT / 'evidence/local_asset_audit.json').read_text(encoding='utf-8'))
    samples = identity['data'][args.split]['samples']
    if args.limit:
        samples = samples[:args.limit]
    manifest = {'run': args.run_name, 'split': args.split, 'expected_images': len(samples),
                'scope': 'compatibility_smoke_only' if args.limit else 'new_runtime_fixed_four_row_reinference',
                'protocol_sha256': sha(ROOT / 'plan/protocol.md'),
                'runtime_amendment_sha256': sha(ROOT / 'plan/runtime_amendment.md'),
                'assets': {k:{'path':str(p), 'sha256':sha(p)} for k,p in assets.items()},
                'scripts': {p.name:sha(p) for p in [Path(__file__), ROOT/'scripts/legacy_canary.py', ROOT/'scripts/modern_worker.py']},
                'sample_ids': [r['sid'] for r in samples], 'legacy_torch': torch.__version__,
                'modern_python': str(BASEPY), 'threshold': .5, 'strict_greater': True,
                'coordinates': 'original_image', 'new_training': False, 'historical_exact_replay': False}
    write_json(out / 'manifest.json', manifest)
    runtime, champion, dataset = build_legacy(args.split)
    lookup = {Path(r['filename']).stem:i for i,r in enumerate(dataset.img_infos)}
    sys.path.insert(0, str(READY))
    from trainer.models.convnext_unet import ConvNeXtUNet
    from trainer.eval import pad_normalized_image
    from trainer.data.common import normalize_image
    u = ConvNeXtUNet()
    saved = torch.load(str(assets['U0']), map_location='cpu')
    u.load_state_dict(saved['model_state'], strict=True)
    del saved
    u = u.cuda().eval().requires_grad_(False)
    log = (out / 'modern_worker.log').open('w', encoding='utf-8')
    worker = subprocess.Popen([str(BASEPY), '-u', str(ROOT/'scripts/modern_worker.py'), '--exchange', str(exchange)],
                              stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000 if os.name=='nt' else 0)
    started = time.perf_counter()
    try:
        wait = time.monotonic()
        while not (exchange/'worker_ready.json').exists():
            if worker.poll() is not None:
                raise RuntimeError('Modern worker failed: read modern_worker.log')
            if time.monotonic()-wait > 180:
                raise TimeoutError('Modern worker startup')
            time.sleep(.2)
        for i, row in enumerate(samples):
            sid = row['sid']
            image_path = DATA/'images'/args.split/(sid+'.jpg')
            label_path = DATA/'labels'/args.split/(sid+'.png')
            assert sha(image_path)==row['image_sha256'] and sha(label_path)==row['label_sha256']
            item = dataset[lookup[sid]]
            image = runtime._unwrap_img(item['img']).unsqueeze(0).cuda()
            meta = runtime._unwrap_meta(item['img_metas'])
            t = time.perf_counter()
            with torch.no_grad():
                pre, ps, pr = champion.pre_head_probs(image, meta)
                imgf = F.interpolate(image, size=tuple(meta[0]['ori_shape'][:2]), mode='bilinear', align_corners=False)[0]
                a = pre.clone()
                a[3] = champion.corrected_class_prob(3, pre, ps, pr, imgf, champion.heads['MA'])
            a, pre = a.cpu().numpy(), pre.cpu().numpy()
            del image, ps, pr, imgf
            torch.cuda.synchronize()
            a_seconds = time.perf_counter()-t
            rgb = np.asarray(Image.open(image_path).convert('RGB'))
            h,w = rgb.shape[:2]
            t = time.perf_counter()
            chw = normalize_image(rgb,[123.675,116.28,103.53],[58.395,57.12,57.375]).transpose(2,0,1)
            padded,_ = pad_normalized_image(np.ascontiguousarray(chw))
            with torch.no_grad():
                x = torch.from_numpy(padded[None]).cuda().float()
                pu = u.eval_forward(x)[...,:h,:w].float().sigmoid()[0].cpu().numpy()
            del x, chw, padded
            torch.cuda.empty_cache()
            u_seconds = time.perf_counter()-t
            write_json(exchange/'request.json',{'sid':sid,'split':args.split})
            wait = time.monotonic()
            while not (exchange/'response.json').exists():
                if worker.poll() is not None:
                    raise RuntimeError('Modern worker failed: read modern_worker.log')
                if time.monotonic()-wait > 300:
                    raise TimeoutError('Modern image inference')
                time.sleep(.2)
            response = json.loads((exchange/'response.json').read_text(encoding='utf-8'))
            assert response['sid']==sid and response['split']==args.split
            s = np.load(str(exchange/'S.npy'),allow_pickle=False)
            c = np.load(str(exchange/'C.npy'),allow_pickle=False)
            p0 = .75*a+.25*pu
            predictions = {'full': .75*p0+.25*s, 'no_h4': .75*(.75*pre+.25*pu)+.25*s,
                           'no_convnext': np.float32(9/13)*a+np.float32(4/13)*s, 'matched_c20': .75*p0+.25*c}
            label = np.asarray(Image.open(label_path))
            counts = {k:pixel_counts(v,label).tolist() for k,v in predictions.items()}
            pred = out/'predictions'/(sid+'.npz')
            with pred.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,**{k:v>.5 for k,v in predictions.items()})
            os.replace(str(pred.with_suffix('.tmp')),str(pred))
            record = {'sid':sid,'hw':[h,w],'counts':counts,'mask_sha256':sha(pred),
                      'image_sha256':row['image_sha256'],'label_sha256':row['label_sha256'],
                      'timing':{'A_inference_seconds':a_seconds,'U_inference_seconds':u_seconds,**response['timing']}}
            write_json(out/'per_image'/(sid+'.json'),record)
            for name in ['response.json','S.npy','C.npy']:
                p = (exchange/name).resolve()
                assert p.parent==exchange.resolve()
                p.unlink()
            del s,c,a,pre,pu,p0,predictions,rgb,label
            progress = {'stage':'RUNNING','completed':i+1,'expected':len(samples),'last_sid':sid,'elapsed_seconds':time.perf_counter()-started}
            write_json(out/'progress.json',progress)
            print(json.dumps(progress),flush=True)
        write_json(out/'completion.json',{'status':'COMPLETE','images':len(samples),'seconds':time.perf_counter()-started,
                                         'scope':manifest['scope'],'all_predeclared_rows_retained':True})
    except BaseException as error:
        write_json(out/'failure.json',{'type':type(error).__name__,'message':str(error),'formal_results_complete':False})
        raise
    finally:
        (exchange/'STOP').write_text('finish',encoding='ascii')
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.terminate()
        log.close()


if __name__ == '__main__':
    main()
