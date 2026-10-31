"""Archive the completed GPU outputs, defer CPU statistics, and shut down."""
import json
from pathlib import Path
import sys
import time
import paramiko

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local_preparation"))
from remote import connect

ROOT = "/root/autodl-tmp/HRDecoder_v3_20261008"
DEST = HERE / "server_receipts"
REMOTE_CODE = r'''
import json, os, shutil, signal, time
from pathlib import Path
from cache_common import ROOT, samples, read_json, write_json, sha
rows = samples("train")
assert len(rows) == 454
by_id = {row["sid"]: row for row in rows}
indexes = {}
for member in ("A", "U", "S"):
    index = read_json(ROOT / "cache/train" / member / "index.json")
    assert index["status"] == "COMPLETE"
    assert index["completed_images"] == index["expected_images"] == 454
    assert set(index["entries"]) == set(by_id)
    assert index["identity"]["dtype"] == "float32"
    assert index["identity"]["channels"] == ["EX", "HE", "SE", "MA"]
    assert index["identity"]["split"] == "train"
    for sid, entry in index["entries"].items():
        row = by_id[sid]
        assert entry["shape"] == [4] + row["hw"]
        assert entry["dtype"] == "float32"
        assert entry["image_sha256"] == row["image_sha256"]
        assert entry["label_sha256"] == row["label_sha256"]
        assert len(entry["file_sha256"]) == len(entry["probability_sha256"]) == 64
        assert (ROOT / "cache/train" / member / entry["file"]).stat().st_size == entry["bytes"]
    indexes[member] = index
parity = read_json(ROOT / "reports/S20_canary_parity.json")
assert parity["max_abs_difference"] == 0
progress_path = ROOT / "reports/cache_audit_progress.json"
progress = read_json(progress_path) if progress_path.exists() else None
# Stop only the observed controller and its CPU audit; the GPU exporters have exited.
for pid, required in ((2903, "/bin/bash cache_pipeline.sh"), (6731, "-u cache_audit.py")):
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    if cmdline_path.exists():
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode()
        assert required in cmdline, (pid, cmdline)
        os.kill(pid, signal.SIGKILL if pid == 2903 else signal.SIGTERM)
compressed = sum(entry["bytes"] for index in indexes.values() for entry in index["entries"].values())
result = {"status": "PASS", "stage": "GPU_TEACHER_CACHE_EXPORT_COMPLETE",
    "images_per_member": 454, "members": ["A", "U", "S"], "total_cache_files": 1362,
    "dtype": "float32", "channel_order": ["EX", "HE", "SE", "MA"],
    "coordinates": "original_image", "compression": "lossless npz deflate1",
    "all_lossless_roundtrips_verified_during_export": True,
    "file_and_tensor_SHA_recorded_at_export": True, "manifest_and_file_sizes_verified": True,
    "full_post_export_CPU_audit": "DEFERRED_UNTIL_NO_GPU_MODE",
    "partial_CPU_audit_progress": progress, "no_training_performed": True,
    "teacher_predictions_are_in_sample": True, "OOF": False,
    "member_compressed_bytes": {m: sum(e["bytes"] for e in i["entries"].values()) for m, i in indexes.items()},
    "index_sha256": {m: sha(ROOT / "cache/train" / m / "index.json") for m in indexes},
    "member_identity_sha256": {m: i["identity_sha256"] for m, i in indexes.items()},
    "compressed_bytes": compressed,
    "uncompressed_bytes": sum(r["hw"][0]*r["hw"][1] for r in rows)*48,
    "data_disk_free_bytes": shutil.disk_usage(ROOT).free,
    "S20_locked_val_canary_parity": parity,
    "shutdown_authorized": True,
    "shutdown_reason": "User requested finishing GPU caches first, then switching to no-GPU mode for CPU work."}
write_json(ROOT / "GPU_CACHE_READY.json", result)
write_json(ROOT / "reports/cpu_audit_deferred.json", {"reason": result["shutdown_reason"], "progress": progress,
    "resume_command": "cd /root/autodl-tmp/HRDecoder_v3_20261008 && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 /root/miniconda3/envs/hrdecoder_q_20261003/bin/python -u cache_audit.py"})
(ROOT / "reports/shutdown_requested.txt").write_text("AUTHORIZED_GPU_CACHE_COMPLETE_SHUTDOWN\n")
print(json.dumps(result), flush=True)
'''

def main():
    DEST.mkdir(exist_ok=True)
    client = connect()
    command = "cd " + ROOT + "\n/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'GPU_PHASE_PY'\n" + REMOTE_CODE + "\nGPU_PHASE_PY\n"
    _, stdout, stderr = client.exec_command(command, timeout=30)
    output, errors = stdout.read().decode(), stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(errors or output)
    summary = json.loads(output)
    files = ["GPU_CACHE_READY.json", "reports/input_audit.json", "reports/S20_canary_parity.json",
        "metadata/inference_weight_receipt.json", "metadata/source_manifest.json", "metadata/data_identity.json",
        "reports/cache_audit_progress.json", "reports/cpu_audit_deferred.json", "reports/shutdown_requested.txt",
        "reports_pipeline.log"]
    files += ["cache/train/" + m + "/index.json" for m in ("A", "U", "S")]
    with client.open_sftp() as ftp:
        for relative in files:
            destination = DEST / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            ftp.get(ROOT + "/" + relative, str(destination))
    completion = {"gpu_cache_complete": True, "receipt_archive_complete": True,
        "full_CPU_audit_deferred": True, "shutdown_authorized": True, "time_unix": time.time()}
    (DEST / "completion.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps({"status": "GPU_CACHE_RECEIPTS_ARCHIVED", "summary": summary}), flush=True)
    (DEST / "shutdown_command_issued.json").write_text(json.dumps({"command": "/bin/bash /usr/bin/shutdown",
        "authorized": True, "time_unix": time.time()}, indent=2), encoding="utf-8")
    client.exec_command("sync\n/bin/bash /usr/bin/shutdown", timeout=20)
    client.close()
    for attempt in range(6):
        time.sleep(5)
        try:
            check = connect()
            check.close()
        except (OSError, paramiko.SSHException) as error:
            observation = {"ssh_unreachable_after_authorized_shutdown": True,
                "time_unix": time.time(), "error": str(error)}
            (DEST / "shutdown_observation.json").write_text(json.dumps(observation, indent=2), encoding="utf-8")
            print(json.dumps({"status": "SERVER_SSH_UNREACHABLE_AFTER_SHUTDOWN"}), flush=True)
            return 0
    print(json.dumps({"status": "SHUTDOWN_SENT_BUT_SSH_STILL_REACHABLE"}), flush=True)
    return 3

if __name__ == "__main__":
    sys.exit(main())
