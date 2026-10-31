"""Original-coordinate float32 cache, resumable atomic lossless writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = Path("/root/HRDecoder/data/DDR_split_613")
CLASSES = ["EX", "HE", "SE", "MA"]

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""): digest.update(block)
    return digest.hexdigest()

def array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)

def samples(split="train"):
    assert split in ("train", "val")
    return read_json(ROOT / "metadata/data_identity.json")["splits"][split]["samples"]

def expected_weights(member):
    if member == "A":
        original = read_json(ROOT / "metadata/local_weight_manifest.json")["weights"]
        names = ["B_weights.pth", "D_student_2000.pth", "E_1500.pth", "E4_weights.pth", "H4_best.pth"]
        return {name: original[name]["sha256"] for name in names}
    name = "U0_inference.pth" if member == "U" else "S20_ema_inference.pth"
    receipt = read_json(ROOT / "metadata/inference_weight_receipt.json")[name]
    assert receipt["all_tensors_bitwise_equal"]
    if member == "S":
        lock = read_json(ROOT / "metadata/final_selection_lock.json")
        assert receipt["source_sha256"] == lock["checkpoints"]["42"]["sha256"]
        assert receipt["source_completed"] == receipt["source_ema_completed"] == 20000
    else:
        original = read_json(ROOT / "metadata/local_weight_manifest.json")["weights"]
        assert receipt["source_sha256"] == original["U0_historical_best.pth"]["sha256"]
    return {name: receipt["inference_file_sha256"]}

def verify_weights(member):
    expected = expected_weights(member)
    for name, digest in expected.items():
        actual = sha(ROOT / "weights" / name)
        if actual != digest: raise ValueError("Teacher weight SHA mismatch: " + name)
    return expected

def identity(member, recipe, runtime):
    data = {
        "schema": "hrdecoder-v3-original-fp32-train-cache/v1",
        "member": member, "split": "train", "channels": CLASSES,
        "dtype": "float32", "coordinates": "original_image",
        "data_identity_sha256": sha(ROOT / "metadata/data_identity.json"),
        "teacher_source_manifest_sha256": sha(ROOT / "metadata/source_manifest.json"),
        "export_script_sha256": sha(ROOT / ("export_" + member + ".py")),
        "cache_common_sha256": sha(ROOT / "cache_common.py"),
        "checkpoint_files_sha256": verify_weights(member),
        "recipe": recipe, "runtime": runtime,
        "in_sample_teacher_predictions": True, "OOF": False,
    }
    material = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return data, hashlib.sha256(material).hexdigest()

def arguments(member):
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--sid", help="Export one manifest-listed image as an early resource check")
    return parser.parse_args()

def save_probability(path, probability):
    p = np.ascontiguousarray(probability)
    assert p.dtype == np.float32 and p.ndim == 3 and p.shape[0] == 4
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
    if shutil.disk_usage(ROOT).free < p.nbytes * 2 + 512 * 1024**2:
        raise RuntimeError("Insufficient space for atomic FP32 cache write")
    digest = array_sha(p)
    temporary = path.with_suffix(".writing")
    with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        with archive.open("prob.npy", "w", force_zip64=True) as target:
            np.lib.format.write_array(target, p, allow_pickle=False)
    # Verify every exported tensor survives the chosen storage bit for bit.
    with np.load(temporary, allow_pickle=False) as stored:
        restored = stored["prob"]
    assert restored.dtype == np.float32 and restored.shape == p.shape
    assert array_sha(restored) == digest, "Lossless roundtrip mismatch"
    os.replace(temporary, path)
    return {"file": path.name, "file_sha256": sha(path), "probability_sha256": digest,
            "bytes": path.stat().st_size, "uncompressed_bytes": p.nbytes,
            "shape": list(p.shape), "dtype": "float32", "lossless_roundtrip_verified": True}

class Writer:
    def __init__(self, member, recipe, runtime, args):
        self.member = member
        self.args = args
        self.directory = ROOT / ("cache/train" if args.split == "train" else "canary/val") / member
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "index.json"
        self.identity, self.identity_sha = identity(member, recipe, runtime)
        self.identity["split"] = args.split
        self.identity_sha = hashlib.sha256(json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.rows = samples(args.split)
        if args.sid is not None:
            self.rows = [row for row in self.rows if row["sid"] == args.sid]
            assert len(self.rows) == 1, "Requested image is absent from the locked split"
        if args.limit is not None: self.rows = self.rows[:args.limit]
        self.entries = {}
        if self.path.exists():
            old = read_json(self.path)
            assert old["identity_sha256"] == self.identity_sha, "Existing cache has a different identity"
            for sid, entry in old["entries"].items():
                path = self.directory / entry["file"]
                assert sha(path) == entry["file_sha256"], "Corrupt cache entry: " + sid
            self.entries = old["entries"]
        self.started = time.monotonic()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pending = deque()
        self.flush("EXPORTING")

    def needed(self, sid):
        return sid not in self.entries

    def save(self, row, probability):
        assert list(probability.shape) == [4] + row["hw"]
        self.pending.append((row, self.pool.submit(save_probability, self.directory / (row["sid"] + ".npz"), probability)))
        if len(self.pending) >= 3:
            self.commit()

    def commit(self):
        row, future = self.pending.popleft()
        entry = future.result()
        entry.update(sid=row["sid"], image_sha256=row["image_sha256"], label_sha256=row["label_sha256"])
        self.entries[row["sid"]] = entry
        self.flush("EXPORTING")
        print(json.dumps({"member": self.member, "completed": len(self.entries), "expected": len(self.rows),
                          "sid": row["sid"], "seconds": time.monotonic() - self.started,
                          "bytes": entry["bytes"], "free_disk_bytes": shutil.disk_usage(ROOT).free}), flush=True)

    def flush(self, status):
        write_json(self.path, {"status": status, "identity": self.identity,
                              "identity_sha256": self.identity_sha, "expected_images": len(self.rows),
                              "completed_images": len(self.entries), "entries": self.entries,
                              "storage": "npz_deflate1_lossless_float32", "updated_at_unix": time.time()})

    def finish(self):
        while self.pending: self.commit()
        self.pool.shutdown(wait=True)
        assert set(self.entries) == {row["sid"] for row in self.rows}
        self.flush("COMPLETE")
        print(json.dumps({"member": self.member, "status": "COMPLETE", "images": len(self.entries),
                          "compressed_bytes": sum(row["bytes"] for row in self.entries.values()),
                          "seconds": time.monotonic() - self.started}), flush=True)

def torch_runtime(torch):
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
