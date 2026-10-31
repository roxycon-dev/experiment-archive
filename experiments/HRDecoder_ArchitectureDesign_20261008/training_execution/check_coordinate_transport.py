"""Check native T/dU/dS transport agrees with transported A/U/S coordinates."""
import json
from pathlib import Path
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"python_deps"))
sys.path.insert(0, str(HERE.parents[1]/"HRDecoder_Ready_20261002/vendor/champion_repo"))
import mmcv
import data

root = HERE/"real_cache_probe"
row = json.loads((root/"probe_identity.json").read_text())
native = []
for member in ("A", "U", "S"):
    with np.load(root/"cache/train"/member/(row["sid"]+".npz")) as archive: native.append(archive["prob"])
a, u, s = native
p0 = .75*a+.25*u
coordinates = (.75*p0+.25*s, u-a, s-p0)
capture = {}
original_augment = data.augment
def captured_augment(*args, **kwargs):
    tensors, results = original_augment(*args, **kwargs)
    capture["geometry"] = results["geometry"]
    return tensors, results
data.augment = captured_augment
dataset = data.TrainCacheDataset(root, "D:/HRDecoder/data/DDR_split_613")
index = next(i for i, value in enumerate(dataset.rows) if value["sid"] == row["sid"])
batch = dataset[(index, 20261008)]
aa, uu, ss = (batch[m].numpy() for m in ("A", "U", "S"))
pp = .75*aa+.25*uu
transformed_coordinates = (.75*pp+.25*ss, uu-aa, ss-pp)
differences = []
for value, expected in zip(coordinates, transformed_coordinates):
    for step in capture["geometry"]:
        if "resize_hw" in step:
            h, w = step["resize_hw"]
            value = data.map_channels(value, lambda channel: mmcv.imresize(channel, (w, h), interpolation="bilinear"))
        elif "crop_y1_y2_x1_x2" in step:
            y1, y2, x1, x2 = step["crop_y1_y2_x1_x2"]
            value = value[:, y1:y2, x1:x2].copy()
        elif step.get("horizontal_flip"):
            value = value[:, :, ::-1].copy()
        elif step.get("rotate"):
            value = data.map_channels(value, lambda channel: mmcv.imrotate(channel,
                angle=step["degrees"], border_value=0, center=None, auto_bound=False, interpolation="bilinear"))
    h, w = value.shape[-2:]
    value = np.pad(value, ((0, 0), (0, 2048-h), (0, 2048-w)))
    differences.append(float(np.max(np.abs(value-expected))))
assert max(differences) < 1e-6
report = {"status": "PASS", "sid": row["sid"], "FP32_tolerance": 1e-6,
    "native_coordinates_then_geometry_vs_geometry_then_coordinates_max_abs": differences,
    "coordinate_order": ["T", "dU", "dS"], "GPU_used": False, "training_updates": 0}
(HERE/"COORDINATE_TRANSPORT_PASS.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report))
