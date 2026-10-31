"""Diagnostic only: check local BF16/reference-kernel replay on archived val image."""
import hashlib
import json
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
from qlib.metrics import counts


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sid = '007-1774-100'
    data = Path('D:/HRDecoder/data/DDR_split_613')
    rgb = np.asarray(Image.open(data / 'images/val' / (sid + '.jpg')).convert('RGB'))
    label = np.asarray(Image.open(data / 'labels/val' / (sid + '.png')))
    target = np.stack([label == c for c in range(1, 5)])
    valid = np.broadcast_to(label != 255, target.shape)
    results = {}
    for arm in ['S', 'C']:
        torch.manual_seed(42)
        model = FinalModel(Q(dinov3_vitb16(pretrained=False)), arm, 42, backend='reference')
        if arm == 'S':
            p = EXP / 'HRDecoder_ArchitectureDesign_20261008/cache_execution/inference_weights/S20_ema_inference.pth'
        else:
            p = FINAL / 'server_results/runs/C_seed42/checkpoint_u20000.pth'
        state = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
        assert state.get('completed', state.get('source_completed')) == 20000
        model.load_state_dict(state['ema'], strict=True)
        del state
        model = model.cuda().eval().requires_grad_(False)
        q = np.empty((4, *label.shape), np.float32)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            for tile in tiles(*label.shape):
                x = torch.from_numpy(rgb_window(rgb, tile))[None].cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = center_logits(model(x))
                q[:, tile.y:tile.y+tile.height, tile.x:tile.x+tile.width] = z.float().sigmoid()[0, :, :tile.height, :tile.width].cpu().numpy()
        ct = counts(q, target, valid)
        archived = json.loads((FINAL / f'server_results/runs/{arm}_seed42/val_u20000/metrics.json').read_text(encoding='utf-8'))
        row = next(r for r in archived['per_image'] if r['sid'] == sid)
        result = {'arm': arm, 'sid': sid, 'counts_now': ct.tolist(), 'counts_archived': row['single'],
                  'integer_counts_match': ct.tolist() == row['single'],
                  'seconds': time.perf_counter() - start, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                  'torch': torch.__version__, 'device': torch.cuda.get_device_name(), 'backend': 'reference',
                  'scope': 'single validation-image compatibility check; no formal test result'}
        out = ROOT / 'evidence' / (arm + '_local_canary.json')
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        np.savez_compressed(ROOT / 'evidence' / (arm + '_local_canary_probability.npz'), prob=q)
        results[arm] = result
        print(json.dumps(result), flush=True)
        del model, q, x, z
        torch.cuda.empty_cache()
    (ROOT / 'evidence/modern_local_compatibility.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
