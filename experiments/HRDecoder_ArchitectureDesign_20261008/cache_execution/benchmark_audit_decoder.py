"""Check accelerated ZIP reading preserves bytes and full statistics."""
import json
from pathlib import Path
import sys
import time
import zipfile
from cache_common import ROOT, read_json, samples
import cache_audit_lowmem as audit

sys.path.insert(0, str(ROOT/"audit_deps"))
from isal import isal_zlib
import isal

indexes = {m: read_json(ROOT/"cache/train"/m/"index.json") for m in ("A", "U", "S")}
row = min(samples("train"), key=lambda r: r["hw"][0]*r["hw"][1])
started = time.monotonic()
reference = audit.inspect(row, indexes)
stdlib_seconds = time.monotonic()-started
zipfile.zlib = isal_zlib
started = time.monotonic()
accelerated = audit.inspect(row, indexes)
isal_seconds = time.monotonic()-started
assert reference == accelerated
result = {"status": "PASS", "sid": row["sid"], "all_file_and_tensor_SHA_verified_both_backends": True,
    "statistics_bitwise_equal": True, "stdlib_seconds": stdlib_seconds,
    "isal_seconds": isal_seconds, "speedup": stdlib_seconds/isal_seconds,
    "isal_version": isal.__version__}
write = ROOT/"reports/audit_decoder_benchmark.json"
write.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result), flush=True)
