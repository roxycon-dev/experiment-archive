"""Check streamed and full-array audit agree, including ignored pixels."""
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import cache_audit as full
import cache_audit_lowmem as stream
from cache_common import array_sha, sha

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    data = root / "data"
    (data / "labels/train").mkdir(parents=True)
    rng = np.random.RandomState(42)
    label = rng.randint(0, 5, (35, 29)).astype(np.uint8)
    label[0, :17] = 255
    Image.fromarray(label).save(data / "labels/train/probe.png")
    row = {"sid": "probe", "hw": [35, 29], "image_sha256": "image", "label_sha256": "label"}
    indexes = {}
    for member in ("A", "U", "S"):
        directory = root / "cache/train" / member
        directory.mkdir(parents=True)
        value = rng.uniform(0, 1, (4, 35, 29)).astype(np.float32)
        value[0, 0, :2] = [0., 1.]
        path = directory / "probe.npz"
        np.savez_compressed(path, prob=value)
        entry = {"file": path.name, "file_sha256": sha(path), "probability_sha256": array_sha(value),
            "image_sha256": "image", "label_sha256": "label", "bytes": path.stat().st_size}
        indexes[member] = {"entries": {"probe": entry}}
    full.ROOT = stream.ROOT = root
    full.DATA = stream.DATA = data
    stream.BLOCK = 17
    reference = full.inspect(row, indexes)
    actual = stream.inspect(row, indexes)
    assert {k: v for k, v in reference.items() if k != "class_statistics"} == {k: v for k, v in actual.items() if k != "class_statistics"}
    max_difference = 0.
    for r, a in zip(reference["class_statistics"], actual["class_statistics"]):
        for key, value in r.items():
            if isinstance(value, float):
                difference = abs(value-a[key]); max_difference = max(max_difference, difference)
                assert difference < 1e-10, (key, value, a[key])
            else: assert value == a[key], (key, value, a[key])
    print(json.dumps({"status": "PASS", "stream_vs_full_statistics_max_abs_difference": max_difference,
        "blocks": 17, "ignored_pixels": 17, "both_file_and_tensor_SHA_checked": True}))
