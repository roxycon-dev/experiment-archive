"""Archive cache completion receipts before the explicitly authorized shutdown."""
import json
from pathlib import Path
import socket
import sys
import time
import paramiko

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local_preparation"))
from remote import connect
REMOTE = "/root/autodl-tmp/HRDecoder_v3_20261008"
DEST = HERE / "server_receipts"

def fetch(sftp, relative):
    with sftp.open(REMOTE + "/" + relative, "rb") as handle:
        return json.loads(handle.read().decode())

def main():
    DEST.mkdir(exist_ok=True)
    last = None
    last_print = 0.
    while True:
        client = None
        try:
            client = connect()
            with client.open_sftp() as ftp:
                try:
                    ready = fetch(ftp, "CACHE_READY.json")
                except FileNotFoundError: ready = None
                if ready is not None:
                    assert ready["status"] == "PASS" and ready["images_per_member"] == 454
                    files = ["CACHE_READY.json", "reports/train_cache_audit.json", "reports/input_audit.json",
                             "reports/S20_canary_parity.json", "metadata/inference_weight_receipt.json",
                             "metadata/source_manifest.json"]
                    files += ["cache/train/" + member + "/index.json" for member in ("A", "U", "S")]
                    for relative in files:
                        destination = DEST / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        ftp.get(REMOTE + "/" + relative, str(destination))
                    try: ftp.get(REMOTE + "/reports_pipeline.log", str(DEST / "reports_pipeline.log"))
                    except FileNotFoundError: pass
                    print(json.dumps({"status": "ARCHIVED_CACHE_READY", "summary": ready,
                                      "local_receipts": str(DEST)}, ensure_ascii=False), flush=True)
                    # The remote pipeline independently performs the already-authorized
                    # shutdown 45 seconds after completion, even if this monitor drops.
                    try:
                        ftp.get(REMOTE + "/reports/shutdown_requested.txt", str(DEST / "shutdown_requested.txt"))
                    except FileNotFoundError: pass
                    (DEST / "completion.json").write_text(json.dumps({"cache_complete": True,
                        "shutdown_authorized": True, "automatic_remote_shutdown_after_seconds": 45,
                        "receipt_archive_complete": True, "time_unix": time.time()}, indent=2), encoding="utf-8")
                    client.close(); client = None
                    time.sleep(50)
                    try:
                        check = connect()
                        check.close()
                        print(json.dumps({"status": "CACHE_DONE_SERVER_STILL_REACHABLE"}), flush=True)
                        return 3
                    except (OSError, paramiko.SSHException):
                        (DEST / "shutdown_observation.json").write_text(json.dumps({"ssh_unreachable_after_authorized_shutdown": True,
                            "time_unix": time.time()}, indent=2), encoding="utf-8")
                        print(json.dumps({"status": "SERVER_SSH_UNREACHABLE_AFTER_SHUTDOWN", "cache_ready_archived": True}), flush=True)
                        return 0
                snapshot = {}
                for member in ("A", "U", "S"):
                    try:
                        index = fetch(ftp, "cache/train/" + member + "/index.json")
                        snapshot[member] = {"status": index["status"], "completed": index["completed_images"], "expected": index["expected_images"]}
                    except FileNotFoundError: snapshot[member] = {"status": "PENDING"}
                try: snapshot["audit"] = fetch(ftp, "reports/cache_audit_progress.json")
                except FileNotFoundError: pass
                try:
                    with ftp.open(REMOTE + "/reports/pipeline_failure.txt", "rb") as handle:
                        failure = handle.read().decode()
                    ftp.get(REMOTE + "/reports_pipeline.log", str(DEST / "failure_pipeline.log"))
                    print(json.dumps({"status": "PIPELINE_NEEDS_REPAIR", "failure": failure,
                                      "local_log": str(DEST / "failure_pipeline.log")}), flush=True)
                    return 2
                except FileNotFoundError: pass
                if snapshot != last and time.monotonic() - last_print > 50:
                    print(json.dumps({"status": "EXPORT_PROGRESS", "members": snapshot}), flush=True)
                    last = snapshot; last_print = time.monotonic()
        except (OSError, paramiko.SSHException) as error:
            print(json.dumps({"status": "MONITOR_CONNECTION_RETRY", "error": str(error)}), flush=True)
        finally:
            if client is not None: client.close()
        time.sleep(15)

if __name__ == "__main__": sys.exit(main())
