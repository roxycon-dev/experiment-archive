import base64
import getpass
import gzip
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import paramiko

HOST = "connect.nmb2.seetacloud.com"
PORT = 27443
EXPECTED_FP = "SHA256:yswdgtshuc+hy/m0jR576i7Nu0Es7lZ71jIrFKXnkE0"
REMOTE_DIR = "/root/autodl-tmp/nativepyramid_sp_v3_20261016/run_seed42_launch02"
REMOTE_LOG = REMOTE_DIR + "/train.jsonl"
REMOTE_PROGRESS = REMOTE_DIR + "/progress.json"
OUT_DIR = Path(r"D:\dataset\experiment\HRDecoder_NativePyramid_20261011\review_20261017\recovered_logs")
OUT_GZ = OUT_DIR / "train_metrics_all40000.jsonl.gz"
OUT_JSONL = OUT_DIR / "train_metrics_all40000.jsonl"
OUT_MANIFEST = OUT_DIR / "train_metrics_all40000.manifest.json"
OUT_BOUNDARIES = OUT_DIR / "train_metrics_boundaries_full.json"
TMP_GZ = OUT_DIR / "train_metrics_all40000.attempt3.jsonl.gz.partial"
TMP_JSONL = OUT_DIR / "train_metrics_all40000.jsonl.partial"

# Keep every record field except the very large per-parameter maps in each
# gradient summary. This code is sent through SSH stdin/exec; it creates no
# remote files.
REMOTE_CODE = r"""
import base64, gzip, hashlib, json, os, sys
path = "/root/autodl-tmp/nativepyramid_sp_v3_20261016/run_seed42_launch02/train.jsonl"
progress_path = "/root/autodl-tmp/nativepyramid_sp_v3_20261016/run_seed42_launch02/progress.json"
raw_hash = hashlib.sha256()
count = 0
first_update = None
last_update = None
prev_update = None
seen = set()
anomaly_samples = []
missing_counts = {}
first_raw = None
last_raw = None
projection_hash = hashlib.sha256()
projected_bytes = 0
projected_top_keys = set()
metrics_keys = set()
gradient_keys = set()
required_metrics = ["positive_bce","negative_bce","squared_dice","dense_bce","main_loss","aux_loss","loss","grad_norm_before_clip","clip_coefficient","gradients_before_clip","gradients_after_clip","supervision_positive_pixels","supervision_positive_images"]
required_grad = ["modules","missing_gradient_parameters","missing_gradient_parameter_modules","direct_scale_class","class_order","scale_order"]
def note_missing(prefix, obj, fields):
    for key in fields:
        if not isinstance(obj, dict) or key not in obj:
            missing_counts[prefix + "." + key] = missing_counts.get(prefix + "." + key, 0) + 1
def safe_project(row):
    out = dict(row)
    m = row.get("metrics")
    if isinstance(m, dict):
        m2 = dict(m)
        for gname in ("gradients_before_clip","gradients_after_clip"):
            g = m.get(gname)
            if isinstance(g, dict):
                g2 = dict(g)
                g2.pop("parameter_modules", None)
                m2[gname] = g2
        out["metrics"] = m2
    return out
try:
    st0 = os.stat(path)
except Exception as e:
    sys.stderr.write("REMOTE_ERROR=" + repr(e) + "\n")
    sys.exit(2)
progress = {}
try:
    with open(progress_path, "rb") as pf:
        progress = json.loads(pf.read().decode("utf-8"))
except Exception as e:
    progress = {"read_error": type(e).__name__}
gz = gzip.GzipFile(fileobj=sys.stdout.buffer, mode="wb", compresslevel=1, mtime=0)
with open(path, "rb", buffering=4*1024*1024) as f:
    for raw in f:
        raw_hash.update(raw)
        count += 1
        if first_raw is None:
            first_raw = raw
        last_raw = raw
        try:
            row = json.loads(raw)
        except Exception as e:
            anomaly_samples.append({"line":count,"parse_error":type(e).__name__})
            continue
        u = row.get("update")
        if count == 1:
            first_update = u
        last_update = u
        if u != count:
            anomaly_samples.append({"line":count,"update":u,"expected":count})
        if u in seen:
            anomaly_samples.append({"line":count,"duplicate_update":u})
        seen.add(u)
        projected_top_keys.update(row.keys() if isinstance(row,dict) else [])
        note_missing("top", row, ["update","executed_updates","checkpoint_persisted_updates","lr","seconds_this_process","metrics","training_sids"])
        m = row.get("metrics", {})
        if isinstance(m,dict):
            metrics_keys.update(m.keys())
        note_missing("metrics", m, required_metrics)
        for gname in ("gradients_before_clip","gradients_after_clip"):
            g = m.get(gname, {}) if isinstance(m,dict) else {}
            if isinstance(g,dict):
                gradient_keys.update(g.keys())
            note_missing(gname, g, required_grad)
        projected = safe_project(row)
        line = (json.dumps(projected, ensure_ascii=False, separators=(",",":")) + "\n").encode("utf-8")
        projection_hash.update(line)
        projected_bytes += len(line)
        gz.write(line)
        if count % 5000 == 0:
            sys.stderr.write("REMOTE_PROGRESS rows=%d last_update=%s\n" % (count, str(last_update)))
gz.close()
st1 = os.stat(path)
summary = {
    "remote_path": path,
    "remote_size_bytes_before": st0.st_size,
    "remote_size_bytes_after": st1.st_size,
    "remote_mtime_before": int(st0.st_mtime),
    "remote_mtime_after": int(st1.st_mtime),
    "remote_sha256": raw_hash.hexdigest(),
    "line_count": count,
    "first_update": first_update,
    "last_update": last_update,
    "unique_update_count": len(seen),
    "update_sequence_is_1_to_line_count": count == 40000 and first_update == 1 and last_update == 40000 and len(seen) == 40000 and not any("expected" in a or "duplicate_update" in a for a in anomaly_samples),
    "anomaly_count": len(anomaly_samples),
    "anomaly_samples": anomaly_samples[:100],
    "projected_jsonl_sha256": projection_hash.hexdigest(),
    "projected_jsonl_bytes": projected_bytes,
    "projected_top_keys": sorted(projected_top_keys),
    "projected_metrics_keys": sorted(metrics_keys),
    "projected_gradient_keys": sorted(gradient_keys),
    "removed_fields": ["metrics.gradients_before_clip.parameter_modules", "metrics.gradients_after_clip.parameter_modules"],
    "missing_field_counts": missing_counts,
    "progress_json": progress,
    "first_raw_b64": base64.b64encode(first_raw or b"").decode("ascii"),
    "last_raw_b64": base64.b64encode(last_raw or b"").decode("ascii")
}
sys.stderr.write("SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=False, separators=(",",":")) + "\n")
sys.stderr.flush()
"""
remote_b64 = base64.b64encode(REMOTE_CODE.encode("utf-8")).decode("ascii")
remote_command = None

if any(p.exists() for p in (OUT_GZ, OUT_JSONL, OUT_MANIFEST, OUT_BOUNDARIES, TMP_GZ, TMP_JSONL)):
    raise SystemExit("Output exists; refusing to overwrite.")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_remote(transport, command):
    ch = transport.open_session(timeout=30)
    ch.exec_command(command)
    stdout_chunks = []
    stderr_data = bytearray()
    with TMP_GZ.open("xb") as out:
        compressed_hash = hashlib.sha256()
        compressed_bytes = 0
        while True:
            progressed = False
            if ch.recv_ready():
                data = ch.recv(1024 * 1024)
                if data:
                    out.write(data)
                    compressed_hash.update(data)
                    compressed_bytes += len(data)
                    progressed = True
            if ch.recv_stderr_ready():
                err = ch.recv_stderr(65536)
                if err:
                    stderr_data.extend(err)
                    for line in err.decode("utf-8", errors="replace").splitlines():
                        if line.startswith("REMOTE_PROGRESS"):
                            print(line, flush=True)
                    progressed = True
            if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
                break
            if not progressed:
                time.sleep(0.1)
        out.flush()
        os.fsync(out.fileno())
    status = ch.recv_exit_status()
    ch.close()
    return status, bytes(stderr_data), compressed_hash.hexdigest(), compressed_bytes

def exec_readonly(transport, command, timeout=20):
    ch = transport.open_session(timeout=timeout)
    ch.exec_command(command)
    stdout = bytearray()
    stderr = bytearray()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ch.recv_ready():
            stdout.extend(ch.recv(65536))
        if ch.recv_stderr_ready():
            stderr.extend(ch.recv_stderr(65536))
        if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
            break
        time.sleep(0.05)
    status = ch.recv_exit_status() if ch.exit_status_ready() else None
    ch.close()
    return status, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")

password = getpass.getpass("SSH password (hidden): ")
client = paramiko.SSHClient()
known_hosts = Path.home() / ".ssh" / "known_hosts"
if known_hosts.exists():
    client.load_system_host_keys(str(known_hosts))
task_known_hosts = Path.home() / ".ssh" / "hrdecoder_v3_known_hosts"
if task_known_hosts.exists():
    client.load_host_keys(str(task_known_hosts))
first_use = {"value": False}
class PinnedMissingHostKey(paramiko.MissingHostKeyPolicy):
    def missing_host_key(self, client_obj, hostname, key_obj):
        fp = "SHA256:" + base64.b64encode(hashlib.sha256(key_obj.asbytes()).digest()).decode("ascii").rstrip("=")
        if fp != EXPECTED_FP:
            raise paramiko.SSHException("Unseen host key does not match the recorded first-use fingerprint.")
        first_use["value"] = True
        print("FIRST_USE_TRUST", hostname, key_obj.get_name(), fp, "in_memory_only", flush=True)
        pass
client.set_missing_host_key_policy(PinnedMissingHostKey())
client.connect(
    HOST,
    port=PORT,
    username="root",
    password=password,
    timeout=30,
    banner_timeout=60,
    auth_timeout=60,
    allow_agent=False,
    look_for_keys=False,
)
transport = client.get_transport()
transport.set_keepalive(30)
key = transport.get_remote_server_key()
fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode("ascii").rstrip("=")
print("HOSTKEY", key.get_name(), fingerprint, "first_use", first_use["value"], flush=True)
if fingerprint != EXPECTED_FP:
    client.close()
    raise SystemExit("Host key pin mismatch; stopped before remote reads.")

nvidia_status, nvidia_stdout, nvidia_stderr = exec_readonly(
    transport,
    "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader",
    timeout=30,
)
gpu_fallback = None
if not nvidia_stdout.strip():
    fb_status, fb_out, fb_err = exec_readonly(
        transport,
        "sh -c 'command -v nvidia-smi; nvidia-smi -L 2>&1; echo nvidia_smi_L_status:$?; cat /sys/class/drm/card*/device/mem_info_vram_total /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null || true'",
        timeout=30,
    )
    gpu_fallback = {"exit_status": fb_status, "stdout": fb_out.strip(), "stderr": fb_err.strip()}
gpu = {"nvidia_smi_exit_status": nvidia_status, "stdout": nvidia_stdout.strip(), "stderr": nvidia_stderr.strip(), "fallback": gpu_fallback}
print("GPU_READONLY", json.dumps(gpu, ensure_ascii=False), flush=True)

sftp = paramiko.SFTPClient.from_transport(transport)
st_before = sftp.stat(REMOTE_LOG)
progress_raw = sftp.open(REMOTE_PROGRESS, "rb").read()
try:
    progress_local = json.loads(progress_raw.decode("utf-8"))
except Exception:
    progress_local = {"parse_error": True}
print("REMOTE_FILE", json.dumps({"path":REMOTE_LOG,"size":st_before.st_size,"mtime":st_before.st_mtime,"progress":progress_local},ensure_ascii=False),flush=True)

probe_status, probe_out, probe_err = exec_readonly(transport, "sh -c 'command -v python3 || command -v python || for p in /root/miniconda3/bin/python /root/miniconda/bin/python /usr/local/bin/python3 /usr/bin/python3 /usr/bin/python; do [ -x \"$p\" ] && { echo \"$p\"; break; }; done'", timeout=30)
python_path = next((line.strip() for line in probe_out.splitlines() if line.strip()), None)
if python_path is None:
    raise RuntimeError("Could not resolve a remote Python executable; probe status=%r stdout=%r stderr=%r" % (probe_status, probe_out, probe_err))
print("REMOTE_PYTHON", json.dumps({"path":python_path,"probe_status":probe_status,"stderr":probe_err.strip()}), flush=True)
remote_command = python_path + " -c 'import base64;exec(base64.b64decode(\"" + remote_b64 + "\"))'"
status, stderr_bytes, compressed_sha, compressed_size = run_remote(transport, remote_command)
if status != 0:
    raise RuntimeError("Remote projection failed with exit status %s; stderr tail=%r" % (status, stderr_bytes[-2000:].decode("utf-8", errors="replace")))
summary = None
for line in stderr_bytes.decode("utf-8", errors="replace").splitlines():
    if line.startswith("SUMMARY_JSON="):
        summary = json.loads(line[len("SUMMARY_JSON="):])
if summary is None:
    raise RuntimeError("Projection ended without remote summary; stderr tail=%r" % stderr_bytes[-2000:])
st_after = sftp.stat(REMOTE_LOG)
summary["remote_size_matches_sftp_stat"] = st_before.st_size == st_after.st_size == summary["remote_size_bytes_before"] == summary["remote_size_bytes_after"]
summary["remote_mtime_matches_sftp_stat"] = int(st_before.st_mtime) == int(st_after.st_mtime) == summary["remote_mtime_before"] == summary["remote_mtime_after"]
summary["compressed_gzip_sha256"] = compressed_sha
summary["compressed_gzip_bytes"] = compressed_size
summary["gpu_readonly"] = gpu
summary["host_key_type"] = key.get_name()
summary["host_key_fingerprint"] = fingerprint
summary["projection_excludes_parameter_modules_only"] = True
summary["projection_scope"] = "all original top-level fields and all metrics fields, except parameter_modules within gradients_before_clip and gradients_after_clip"
summary["raw_original_was_not_transferred_or_modified"] = True
if not summary["remote_size_matches_sftp_stat"] or not summary["remote_mtime_matches_sftp_stat"]:
    raise RuntimeError("Remote source changed during read; retaining compressed partial for review.")
os.replace(TMP_GZ, OUT_GZ)

local_proj_hash = hashlib.sha256()
local_proj_bytes = 0
local_rows = 0
with gzip.open(OUT_GZ, "rb") as src, TMP_JSONL.open("xb") as dst:
    for line in src:
        local_proj_hash.update(line)
        local_proj_bytes += len(line)
        local_rows += 1
        dst.write(line)
    dst.flush()
    os.fsync(dst.fileno())
if local_proj_hash.hexdigest() != summary["projected_jsonl_sha256"] or local_proj_bytes != summary["projected_jsonl_bytes"] or local_rows != summary["line_count"]:
    raise RuntimeError("Local decompression/projection checksum or row-count mismatch.")
os.replace(TMP_JSONL, OUT_JSONL)
summary["local_projected_jsonl_sha256"] = local_proj_hash.hexdigest()
summary["local_projected_jsonl_bytes"] = local_proj_bytes
summary["local_projected_line_count"] = local_rows
summary["complete_40000_rows"] = summary["line_count"] == 40000 and summary["update_sequence_is_1_to_line_count"]
summary["boundaries_sidecar"] = str(OUT_BOUNDARIES)
first_raw = base64.b64decode(summary.pop("first_raw_b64"))
last_raw = base64.b64decode(summary.pop("last_raw_b64"))
OUT_BOUNDARIES.write_text(json.dumps({
    "remote_path": REMOTE_LOG,
    "first_record_raw": first_raw.decode("utf-8", errors="replace"),
    "last_record_raw": last_raw.decode("utf-8", errors="replace"),
}, ensure_ascii=False, indent=2), encoding="utf-8")
summary["boundaries_sha256"] = hashlib.sha256(OUT_BOUNDARIES.read_bytes()).hexdigest()
OUT_MANIFEST.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("COMPLETE", json.dumps({
    "gzip":str(OUT_GZ),"gzip_bytes":OUT_GZ.stat().st_size,"gzip_sha256":compressed_sha,
    "projection":str(OUT_JSONL),"projection_bytes":OUT_JSONL.stat().st_size,"projection_sha256":local_proj_hash.hexdigest(),
    "manifest":str(OUT_MANIFEST),"boundaries":str(OUT_BOUNDARIES),"raw_size":summary["remote_size_bytes_before"],
    "raw_sha256":summary["remote_sha256"],"rows":summary["line_count"],"complete_40000":summary["complete_40000_rows"],
    "anomaly_count":summary["anomaly_count"],"gpu":gpu
},ensure_ascii=False),flush=True)
sftp.close()
client.close()








