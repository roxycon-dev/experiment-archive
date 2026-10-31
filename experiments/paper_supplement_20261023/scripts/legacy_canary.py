"""Diagnostic A/H4-bypass/U forward on one archived validation image."""
import copy
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
READY = EXP / 'HRDecoder_Ready_20261002'
DATA = Path('D:/HRDecoder/data/DDR_split_613')


def build_legacy(split='val'):
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(READY / 'legacy'))
    sys.path.append(str(EXP / 'HRDecoder_ArchitectureDesign_20261008/training_execution/python_deps'))
    import runtime
    settings = json.loads((READY / 'configs/champion_server.json').read_text(encoding='utf-8'))
    settings['repo'] = str(READY / 'vendor/champion_repo')
    settings['data_root'] = str(DATA)
    settings['reference_checkpoint'] = str(READY / 'best_pipeline/weights/B_weights.pth')
    settings['champion']['head'] = str(READY / 'best_pipeline/weights/H4_best.pth')
    for m in settings['champion']['members']:
        m['checkpoint'] = str(READY / 'best_pipeline/weights' / Path(m['checkpoint']).name)
    runtime.inject_repo(settings['repo'])
    cfg = runtime.load_config(settings['repo'], settings['config'])
    from mmseg.datasets import build_dataset
    d = copy.deepcopy(cfg.data['val'])
    d.data_root = str(DATA)
    d.img_dir = 'images/' + split
    d.ann_dir = 'labels/' + split
    d.pipeline = copy.deepcopy(cfg.data['test']['pipeline'])
    d.test_mode = True
    if 'split' in d:
        d.split = None
    dataset = build_dataset(d)
    champion = runtime.ChampionEnsemble(settings, settings['champion'], device='cuda')
    return runtime, champion, dataset


def main():
    runtime, champion, dataset = build_legacy()
    sid = '007-1774-100'
    index = next(i for i, r in enumerate(dataset.img_infos) if Path(r['filename']).stem == sid)
    item = dataset[index]
    image = runtime._unwrap_img(item['img']).unsqueeze(0).cuda()
    meta = runtime._unwrap_meta(item['img_metas'])
    started = time.perf_counter()
    with torch.no_grad():
        pf, ps, pr = champion.pre_head_probs(image, meta)
        imgf = F.interpolate(image, size=tuple(meta[0]['ori_shape'][:2]), mode='bilinear', align_corners=False)[0]
        corrected = pf.clone()
        corrected[3] = champion.corrected_class_prob(3, pf, ps, pr, imgf, champion.heads['MA'])
    target_path = ROOT / 'evidence/A_local_canary_probability.npz'
    np.savez_compressed(target_path, prob=corrected.cpu().numpy(), pre=pf.cpu().numpy())
    summary = {'sid': sid, 'torch': torch.__version__, 'seconds': time.perf_counter()-started,
               'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'shape': list(pf.shape),
               'unchanged_non_MA_channels': bool(torch.equal(corrected[:3], pf[:3])),
               'scope': 'single validation image, compatibility diagnostic only'}
    (ROOT / 'evidence/A_local_canary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
