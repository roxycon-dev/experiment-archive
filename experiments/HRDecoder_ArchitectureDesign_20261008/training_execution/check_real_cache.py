"""Real original images, FP32 caches, shared geometry and loss-target formation."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"python_deps"))
sys.path.insert(0, str(HERE.parents[1]/"HRDecoder_Ready_20261002/vendor/champion_repo"))
from data import TrainCacheDataset
from objectives import teacher_coordinates

torch.set_num_threads(4)
started = time.monotonic()
root = HERE/"real_cache_probe"
dataset = TrainCacheDataset(root, "D:/HRDecoder/data/DDR_split_613")
row = json.loads((root/"probe_identity.json").read_text())
index = next(i for i, r in enumerate(dataset.rows) if r["sid"] == row["sid"])
assert hashlib.sha256(dataset.images[row["sid"]].read_bytes()).hexdigest() == row["image_sha256"]
label_path = dataset.data_root/"labels/train"/(row["sid"]+".png")
assert hashlib.sha256(label_path.read_bytes()).hexdigest() == row["label_sha256"]
batch = dataset[(index, 20261008)]
assert batch["image"].shape == (3, 2048, 2048)
assert batch["image"].dtype == torch.float32
assert batch["label"].shape == batch["valid"].shape == (1, 2048, 2048)
assert batch["valid"].sum() > 0 and batch["valid"].sum() < 2048**2
for member in ("A", "U", "S"):
    assert batch[member].shape == (4, 2048, 2048) and batch[member].dtype == torch.float32
teacher, du, ds = teacher_coordinates(batch["A"], batch["U"], batch["S"])
ar = teacher-.25*du-.25*ds
assert torch.allclose(ar, batch["A"], atol=1e-6)
assert du.min() < 0 and du.max() > 0 and ds.min() < 0 and ds.max() > 0
report = {"status": "PASS", "sid": row["sid"], "original_hw": row["hw"],
    "canvas": [2048, 2048], "source_image_and_label_SHA_verified": True,
    "A_U_S_FP32_shared_geometry": True, "source_outside_interpolation_masked": True,
    "signed_dU_dS_both_directions_present": True,
    "valid_pixels": int(batch["valid"].sum()),
    "dU_range": [float(du.min()), float(du.max())], "dS_range": [float(ds.min()), float(ds.max())],
    "probability_coordinate_inverse_max_abs": float((ar-batch["A"]).abs().max()),
    "peak_working_set_bytes": None,
    "seconds": time.monotonic()-started, "GPU_used": False, "training_updates": 0}
(HERE/"REAL_CACHE_PIPELINE_PASS.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report), flush=True)
