"""Same full cache audit using bounded ZIP streams, suitable for a 2-GiB host."""
from contextlib import ExitStack
import hashlib
import json
import resource
import zipfile
import numpy as np
from PIL import Image
import cache_audit as audit
from cache_common import ROOT, DATA, CLASSES, sha, write_json

BLOCK = 262144

class SerialPool:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def map(self, function, rows): return map(function, rows)

def open_array(stack, path, shape):
    archive = stack.enter_context(zipfile.ZipFile(path))
    assert archive.namelist() == ["prob.npy"]
    stream = stack.enter_context(archive.open("prob.npy"))
    version = np.lib.format.read_magic(stream)
    if version == (1, 0): header = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0): header = np.lib.format.read_array_header_2_0(stream)
    else: raise ValueError("Unexpected NPY version " + str(version))
    actual_shape, fortran_order, dtype = header
    assert list(actual_shape) == shape and not fortran_order and dtype == np.dtype("float32")
    return stream

def inspect(row, indexes):
    sid = row["sid"]
    count = row["hw"][0] * row["hw"][1]
    file_bytes = 0
    entries = []
    paths = []
    for member in ("A", "U", "S"):
        entry = indexes[member]["entries"][sid]
        path = ROOT / "cache/train" / member / entry["file"]
        assert sha(path) == entry["file_sha256"], sid + ": file SHA mismatch"
        assert entry["image_sha256"] == row["image_sha256"]
        assert entry["label_sha256"] == row["label_sha256"]
        file_bytes += entry["bytes"]
        entries.append(entry); paths.append(path)
    with Image.open(DATA / "labels/train" / (sid + ".png")) as opened:
        label = np.asarray(opened).reshape(-1)
    assert len(label) == count
    class_rows = []
    max_inverse = 0.
    valid_count = 0
    hashes = [hashlib.sha256() for _ in paths]
    with ExitStack() as stack:
        streams = [open_array(stack, path, [4] + row["hw"]) for path in paths]
        for c, name in enumerate(CLASSES):
            stats = {"class": name, "n": 0, "gt_positive": 0,
                "sum_du": 0., "sum_ds": 0., "sum_du2": 0., "sum_ds2": 0., "sum_du_ds": 0.,
                "sum_abs_du": 0., "sum_abs_ds": 0., "positive_du": 0, "negative_du": 0,
                "positive_ds": 0, "negative_ds": 0, "opposite_sign": 0,
                "u_sum": 0., "u_square_sum": 0., "s_sum": 0., "s_square_sum": 0.}
            for start in range(0, count, BLOCK):
                stop = min(start + BLOCK, count)
                values = []
                for stream, digest in zip(streams, hashes):
                    raw = stream.read((stop-start)*4)
                    assert len(raw) == (stop-start)*4, sid + ": truncated probability"
                    digest.update(raw)
                    value = np.frombuffer(raw, dtype=np.float32)
                    assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
                    values.append(value)
                valid = label[start:stop] != 255
                av, uv, sv = (v[valid] for v in values)
                du = uv-av; p0 = .75*av+.25*uv; ds = sv-p0
                target = .75*p0+.25*sv
                ar = target-.25*du-.25*ds; ur = ar+du; sr = ar+.25*du+ds
                if len(av):
                    max_inverse = max(max_inverse, float(np.max(np.abs(ar-av))),
                        float(np.max(np.abs(ur-uv))), float(np.max(np.abs(sr-sv))))
                x, y = du.astype(np.float64), ds.astype(np.float64)
                stats["n"] += len(x)
                stats["gt_positive"] += int(((label[start:stop] == c+1) & valid).sum())
                stats["sum_du"] += float(x.sum()); stats["sum_ds"] += float(y.sum())
                stats["sum_du2"] += float(np.dot(x, x)); stats["sum_ds2"] += float(np.dot(y, y))
                stats["sum_du_ds"] += float(np.dot(x, y))
                stats["sum_abs_du"] += float(np.abs(x).sum()); stats["sum_abs_ds"] += float(np.abs(y).sum())
                stats["positive_du"] += int((du > 1e-7).sum()); stats["negative_du"] += int((du < -1e-7).sum())
                stats["positive_ds"] += int((ds > 1e-7).sum()); stats["negative_ds"] += int((ds < -1e-7).sum())
                stats["opposite_sign"] += int((((du > 1e-7) & (ds < -1e-7)) | ((du < -1e-7) & (ds > 1e-7))).sum())
                stats["u_sum"] += float(uv.sum(dtype=np.float64))
                stats["u_square_sum"] += float(np.square(uv.astype(np.float64)).sum())
                stats["s_sum"] += float(sv.sum(dtype=np.float64))
                stats["s_square_sum"] += float(np.square(sv.astype(np.float64)).sum())
            if c == 0: valid_count = stats["n"]
            class_rows.append(stats)
        for stream, digest, entry in zip(streams, hashes, entries):
            assert stream.read(1) == b"", sid + ": excess probability data"
            assert digest.hexdigest() == entry["probability_sha256"], sid + ": tensor SHA mismatch"
    assert max_inverse < 1e-6, sid + ": teacher coordinate reconstruction mismatch"
    return {"sid": sid, "hw": row["hw"], "valid_pixels": valid_count, "class_statistics": class_rows,
        "max_algebra_inverse_abs_difference": max_inverse, "compressed_bytes": file_bytes}

if __name__ == "__main__":
    audit.inspect = inspect
    audit.ThreadPoolExecutor = SerialPool
    audit.main()
    write_json(ROOT / "reports/low_memory_audit_resources.json", {
        "stream_block_elements": BLOCK, "workers": 1,
        "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "full_probability_array_loaded": False, "GPU_used": False})
