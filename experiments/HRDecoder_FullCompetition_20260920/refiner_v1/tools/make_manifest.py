#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""make_manifest — deliverable manifest: file hashes + provenance for refiner_v1.

Run on the SERVER (or locally with --root pointing at the local copy).  Writes
manifests/REFINER_V1_MANIFEST.json containing a sha256 per source/config/test
file plus the sha256 of the audit artefacts and any built caches, so a later
reviewer can prove which code produced which numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

INCLUDE_DIRS = ("refiner", "tests", "configs", "audit", "tools")
INCLUDE_FILES = ("build_patch_cache.py", "train_refiner.py", "eval_refiner.py",
                 "README.md", "GO_NO_GO.md")
EXTS = (".py", ".json", ".md")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=_HERE)
    ap.add_argument("--out", default=os.path.join(_HERE, "manifests",
                                                  "REFINER_V1_MANIFEST.json"))
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra files/dirs to hash (e.g. patch_cache, runs)")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    files = {}

    def add(p):
        rel = os.path.relpath(p, root).replace("\\", "/")
        files[rel] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}

    for d in INCLUDE_DIRS:
        base = os.path.join(root, d)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for fn in sorted(filenames):
                if fn.endswith(EXTS):
                    add(os.path.join(dirpath, fn))
    for fn in INCLUDE_FILES:
        p = os.path.join(root, fn)
        if os.path.isfile(p):
            add(p)

    extras = {}
    for e in args.extra:
        if os.path.isabs(e):
            p = e
        else:
            # accept paths relative to either --root or the current directory
            cands = [os.path.join(root, e), os.path.abspath(e)]
            p = next((c for c in cands if os.path.exists(c)), cands[0])
        if os.path.isfile(p):
            extras[os.path.basename(p)] = {"sha256": sha256_file(p),
                                           "bytes": os.path.getsize(p)}
        elif os.path.isdir(p):
            tot, n = 0, 0
            for dirpath, _dn, fns in os.walk(p):
                for fn in fns:
                    tot += os.path.getsize(os.path.join(dirpath, fn))
                    n += 1
            extras[os.path.basename(p.rstrip("/\\"))] = {"path": p,
                                                         "n_files": n, "bytes": tot}
        else:
            extras[str(e)] = {"error": "not found", "tried": p}

    man = {"schema": "hrdecoder-refiner-manifest/v1",
           "root": root, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_files": len(files), "files": files, "artifacts": extras}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("[manifest] %d files -> %s" % (len(files), args.out))
    for k in sorted(files):
        print("  %s  %s" % (files[k]["sha256"][:16], k))
    for k, v in sorted(extras.items()):
        print("  ARTIFACT %s %r" % (k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
