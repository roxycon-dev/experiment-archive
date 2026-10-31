"""Paired S20/C20 BF16 replay worker. Requests are restricted to frozen DDR IDs."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
FINAL = EXP / 'HRDecoder_FinalDecision_20261005/execution'
sys.path.insert(0, str(FINAL))
sys.path.insert(0, str(EXP / 'HRDecoder_Q_20261003/vendor/dinov3'))
from dinov3.hub.backbones import dinov3_vitb16
from qlib.model import Q
from final_model import FinalModel
from qlib.geometry import tiles, rgb_window, center_logits


def atomic_json(p, value):
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exchange', type=Path, required=True)
    args = parser.parse_args()
    exchange = args.exchange.resolve()
    assert exchange.is_relative_to(ROOT.resolve())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Path('D:/HRDecoder/data/DDR_split_613')
    models = {}
    for arm in ['S', 'C']:
        torch.manual_seed(42)
        model = FinalModel(Q(dinov3_vitb16(pretrained=False)), arm, 42, backend='reference')
        p = (EXP / 'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/S20_ema_inference.pth'
             if arm == 'S' else FINAL / 'server_results/runs/C_seed42/checkpoint_u20000.pth')
        saved = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
        assert saved.get('completed', saved.get('source_completed')) == 20000
        model.load_state_dict(saved['ema'], strict=True)
        del saved
        models[arm] = model.cuda().eval().requires_grad_(False)
    atomic_json(exchange / 'worker_ready.json', {'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'backend': 'reference'})
    while not (exchange / 'STOP').exists():
        request = exchange / 'request.json'
        if not request.exists():
            time.sleep(.2)
            continue
        job = json.loads(request.read_text(encoding='utf-8'))
        sid, split = job['sid'], job['split']
        if split not in ['val', 'test'] or sid not in {p.stem for p in (data / 'images' / split).glob('*.jpg')}:
            raise ValueError('Request outside frozen DDR split')
        rgb = np.asarray(Image.open(data / 'images' / split / (sid + '.jpg')).convert('RGB'))
        h, w = rgb.shape[:2]
        rows = {}
        for arm, model in models.items():
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            q = np.empty((4, h, w), np.float32)
            with torch.inference_mode():
                for tile in tiles(h, w):
                    x = torch.from_numpy(rgb_window(rgb, tile))[None].cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        z = center_logits(model(x))
                    q[:, tile.y:tile.y+tile.height, tile.x:tile.x+tile.width] = z.float().sigmoid()[0, :, :tile.height, :tile.width].cpu().numpy()
            path = exchange / (arm + '.npy')
            with path.with_suffix('.tmp').open('wb') as f:
                np.save(f, q, allow_pickle=False)
            os.replace(path.with_suffix('.tmp'), path)
            rows[arm] = {'seconds': time.perf_counter()-start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            del q, x, z
            torch.cuda.empty_cache()
        request.unlink()
        atomic_json(exchange / 'response.json', {'sid': sid, 'split': split, 'timing': rows})


if __name__ == '__main__':
    main()
