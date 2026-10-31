"""Wait for the full CPU audit, then archive evidence without reading bulk maps."""
import json
from pathlib import Path
import sys
import time
import paramiko
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"local_preparation"))
from remote import connect

ROOT = "/root/autodl-tmp/HRDecoder_v3_20261008"
DEST = HERE.parent/"cache_execution/server_receipts"
DEST.mkdir(exist_ok=True)
last = None
def read(ftp, name):
    with ftp.open(ROOT+"/"+name, "rb") as stream: return json.loads(stream.read().decode())

while True:
    client = None
    try:
        client = connect()
        with client.open_sftp() as ftp:
            try: ready = read(ftp, "CACHE_READY.json")
            except FileNotFoundError: ready = None
            if ready is not None:
                assert ready["status"] == "PASS" and ready["images_per_member"] == 454
                assert ready["all_file_and_tensor_SHA_verified"]
                assert all(row["covariance_rank_relative_tolerance_1e_minus6"] == 2 and row["gt_positive_pixels"] > 0 for row in ready["class_statistics"])
                for name in ("TRAINER_PREPARATION_PASS.json", "SERVER_INITIALIZATION_PASS.json", "SERVER_OPTIMIZER_RESUME_PASS.json", "SERVER_REAL_HRNET_ROUTE_PASS.json"):
                    assert read(ftp, "reports/"+name)["status"] == "PASS"
                files = ["CACHE_READY.json", "reports/train_cache_audit.json", "reports/low_memory_audit_resources.json",
                    "reports/TRAINER_PREPARATION_PASS.json", "reports/SERVER_INITIALIZATION_PASS.json",
                    "reports/SERVER_OPTIMIZER_RESUME_PASS.json", "reports/cpu_audit_nogpu.log",
                    "reports/SERVER_REAL_HRNET_ROUTE_PASS.json",
                    "reports/audit_decoder_benchmark.json", "training_execution/execution_manifest.json"]
                for name in files:
                    destination = DEST/name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    ftp.get(ROOT+"/"+name, str(destination))
                (DEST/"no_gpu_completion.json").write_text(json.dumps({"full_cache_audit": "PASS",
                    "CPU_training_preparation": "PASS", "formal_training_updates": 0,
                    "CUDA_smoke": "PENDING_4090_RESTART", "receipts_archived": True,
                    "time_unix": time.time()}, indent=2)+"\n")
                print(json.dumps({"status": "FULL_CPU_AUDIT_AND_PREPARATION_ARCHIVED", "images": 454,
                    "class_statistics": ready["class_statistics"], "audit_seconds": ready["seconds"]}), flush=True)
                break
            progress = read(ftp, "reports/cache_audit_progress.json")
            if progress != last:
                print(json.dumps(progress), flush=True)
                last = progress
    except (OSError, paramiko.SSHException) as error:
        print(json.dumps({"monitor_retry": str(error)}), flush=True)
    finally:
        if client is not None: client.close()
    time.sleep(20)
