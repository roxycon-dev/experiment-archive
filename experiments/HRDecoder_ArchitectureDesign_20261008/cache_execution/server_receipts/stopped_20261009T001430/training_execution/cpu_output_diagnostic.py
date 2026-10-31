"""One actual 2048 validation image, FP32 CPU only; no training update."""
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path('/root/autodl-tmp/HRDecoder_v3_20261008')
HERE = ROOT / 'training_execution'
OUT = ROOT / 'reports/diagnostics/cpu_output_20000_positive_EX_SE.json'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'vendor/champion_repo'))

def guard():
    maximum = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if maximum == 'max': return
    while True:
        remaining = int(maximum) - int(Path('/sys/fs/cgroup/memory.current').read_text())
        if remaining < 1024**3:
            error = {'status': 'ABORTED_FOR_MEMORY_HEADROOM', 'time_unix': time.time(),
                     'only_diagnostic_process_terminated': True, 'no_training_update': True}
            OUT.write_text(json.dumps(error) + '\n')
            os._exit(12)
        time.sleep(0.25)

threading.Thread(target=guard, daemon=True).start()
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from mmcv import Config
from mmseg.models import build_segmentor
from model import V3Student
from data import evaluation_image

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
started = time.monotonic()
settings = copy.deepcopy(Config.fromfile(str(ROOT / 'vendor/champion_repo/configs/lesion/exp/baseline_split613.py')).model)
settings.pretrained = None
baseline = build_segmentor(settings)
student = V3Student(baseline, canvas=2048)
checkpoint = torch.load(ROOT / 'reports/diagnostics/checkpoint_20000.pt', map_location='cpu')
assert checkpoint['completed'] == 20000
assert checkpoint['identity_SHA'] == json.loads((ROOT / 'training_runs/v3/protocol.json').read_text())['identity_SHA']
student.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
student.eval()
data = Path('/root/HRDecoder/data/DDR_split_613')
rows = json.loads((ROOT / 'metadata/data_identity.json').read_text())['splits']['val']['samples']
for candidate in rows:
    with Image.open(data / 'labels/val' / (candidate['sid'] + '.png')) as source:
        candidate_label = np.asarray(source)
    if (candidate_label == 1).any() and (candidate_label == 3).any():
        row = candidate
        break
else:
    raise RuntimeError('No validation image with both EX and SE labels found.')
image_path = next(p for p in (data / 'images/val').iterdir() if p.stem == row['sid'])
image = evaluation_image(image_path).unsqueeze(0)

def summary(value):
    return [{'class': name, 'finite': bool(torch.isfinite(channel).all()),
             'min': float(channel.min()), 'mean': float(channel.mean()), 'max': float(channel.max()),
             'positive_logit_pixels': int((channel > 0).sum())}
            for name, channel in zip(('EX', 'HE', 'SE', 'MA'), value[0].float())]

with torch.no_grad():
    output = student(image)
    logits = F.interpolate(output['logits'].float(), size=row['hw'], mode='bilinear', align_corners=False)
    probability = logits.sigmoid()
    probabilities = [{'class': name, 'finite': bool(torch.isfinite(channel).all()),
                      'min': float(channel.min()), 'max': float(channel.max()),
                      'pixels_above_0p5': int((channel > .5).sum())}
                     for name, channel in zip(('EX', 'HE', 'SE', 'MA'), probability[0])]
    components = {key: summary(output[key]) for key in ('base_logits', 'residual', 'logits')}
with Image.open(data / 'labels/val' / (row['sid'] + '.png')) as source:
    label = np.asarray(source)
report = {'status': 'COMPLETE', 'time_unix': time.time(), 'checkpoint_update': 20000,
          'device': 'CPU', 'dtype': 'FP32', 'canvas': 2048, 'validation_sid': row['sid'],
          'seconds': time.monotonic() - started,
          'GT_positive_pixels': [int((label == c + 1).sum()) for c in range(4)],
          'original_resolution_probabilities': probabilities, 'components': components,
          'formal_SGD_updates': 0, 'training_protocol_changed': False,
          'no_data_or_cache_hashes_recomputed': True,
          'limitation': 'One FP32 CPU image is a numerical diagnostic, not a replacement for the 76-image BF16 GPU validation.'}
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
