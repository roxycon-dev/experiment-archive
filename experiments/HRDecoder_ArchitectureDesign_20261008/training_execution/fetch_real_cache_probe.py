"""Fetch one smallest real cache triplet for local pipeline integration checks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"local_preparation"))
from remote import connect

receipts = HERE.parent/"cache_execution/server_receipts"
identity = json.loads((receipts/"metadata/data_identity.json").read_text(encoding="utf-8"))
row = min(identity["splits"]["train"]["samples"], key=lambda r: r["hw"][0]*r["hw"][1])
target = HERE/"real_cache_probe"
(target/"metadata").mkdir(parents=True, exist_ok=True)
shutil.copyfile(receipts/"metadata/data_identity.json", target/"metadata/data_identity.json")

def download(member):
    manifest_path = receipts/"cache/train"/member/"index.json"
    index = json.loads(manifest_path.read_text())
    entry = index["entries"][row["sid"]]
    directory = target/"cache/train"/member
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, directory/"index.json")
    destination = directory/entry["file"]
    if not destination.exists():
        client = connect()
        try:
            with client.open_sftp() as ftp:
                ftp.get("/root/autodl-tmp/HRDecoder_v3_20261008/cache/train/"+member+"/"+entry["file"], str(destination))
        finally: client.close()
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == entry["file_sha256"]
    return destination.stat().st_size

with ThreadPoolExecutor(max_workers=3) as pool: total = sum(pool.map(download, ("A", "U", "S")))
(target/"probe_identity.json").write_text(json.dumps(row, indent=2))
print(json.dumps({"sid": row["sid"], "hw": row["hw"], "cache_bytes": total, "three_file_SHA_verified": True}))
