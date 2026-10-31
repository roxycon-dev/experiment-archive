set -u
R=/root/autodl-tmp/HRDecoder_FullCompetition_20260920
cat > /root/_npzprobe.py <<'PY'
import os, sys, hashlib, json
import numpy as np

R = "/root/autodl-tmp/HRDecoder_FullCompetition_20260920/expert_outputs"
dirs = []
for fam in ("hrplus", "fullmodel"):
    base = os.path.join(R, fam)
    if not os.path.isdir(base):
        continue
    for m in sorted(os.listdir(base)):
        p = os.path.join(base, m)
        if os.path.isdir(p):
            dirs.append((fam + "/" + m, p))

report = {}
for name, p in dirs:
    files = sorted(f for f in os.listdir(p) if f.endswith(".npz"))
    rec = {"n_files": len(files), "keys": None, "dtypes": {}, "shapes": {}, "classes": None,
           "sizes": [], "file_bytes": 0}
    if files:
        f0 = os.path.join(p, files[0])
        with np.load(f0) as z:
            rec["keys"] = sorted(z.files)
            for k in z.files:
                a = z[k]
                rec["dtypes"][k] = str(a.dtype)
                rec["shapes"][k] = list(a.shape)
        rec["file_bytes"] = os.path.getsize(f0)
    # consistency across all files: keys, dtypes, classes
    key_sig = set()
    dtype_sig = set()
    class_sig = set()
    total = 0
    for f in files:
        fp = os.path.join(p, f)
        total += os.path.getsize(fp)
        with np.load(fp) as z:
            key_sig.add(tuple(sorted(z.files)))
            dtype_sig.add(tuple((k, str(z[k].dtype)) for k in sorted(z.files)))
            if "classes" in z.files:
                class_sig.add(tuple(np.asarray(z["classes"]).ravel().tolist()))
            else:
                class_sig.add(None)
    rec["key_signatures"] = [list(x) for x in key_sig]
    rec["n_key_signatures"] = len(key_sig)
    rec["dtype_signatures"] = [list(x) for x in dtype_sig]
    rec["n_dtype_signatures"] = len(dtype_sig)
    rec["class_sets"] = [list(x) if x else None for x in class_sig]
    rec["total_bytes"] = total
    report[name] = rec

report_gt = {}
gtd = os.path.join(R, "_shared", "gt")
if os.path.isdir(gtd):
    gfiles = sorted(f for f in os.listdir(gtd) if f.endswith(".npz"))
    with np.load(os.path.join(gtd, gfiles[0])) as z:
        report_gt = {"n_files": len(gfiles), "keys": sorted(z.files),
                     "dtypes": {k: str(z[k].dtype) for k in z.files},
                     "shapes": {k: list(z[k].shape) for k in z.files}}

print(json.dumps({"models": report, "gt": report_gt}, indent=1, ensure_ascii=False))
PY
/root/miniconda3/bin/python /root/_npzprobe.py
