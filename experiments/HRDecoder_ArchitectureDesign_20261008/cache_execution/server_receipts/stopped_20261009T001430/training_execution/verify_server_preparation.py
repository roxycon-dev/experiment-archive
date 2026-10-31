"""Standard-library source/evidence verification before the GPU stage."""
import hashlib
import json
from pathlib import Path
import py_compile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
manifest = json.loads((HERE/"execution_manifest.json").read_text())
for name, digest in manifest.items():
    path = HERE/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name+": SHA mismatch"
    if path.suffix == ".py": py_compile.compile(str(path), doraise=True)
for name in ("CPU_PREPARATION_PASS.json", "REAL_CACHE_PIPELINE_PASS.json", "OPTIMIZER_RESUME_PASS.json"):
    assert json.loads((HERE/name).read_text())["status"] == "PASS"
for name in ("SERVER_INITIALIZATION_PASS.json", "SERVER_OPTIMIZER_RESUME_PASS.json", "SERVER_REAL_HRNET_ROUTE_PASS.json"):
    value = json.loads((ROOT/"reports"/name).read_text())
    assert value["status"] == "PASS" and value["torch"] == "2.0.1+cu117"
assert json.loads((ROOT/"reports/SERVER_INITIALIZATION_PASS.json").read_text())["all_backbone_tensors_equal_pinned_ImageNet"]
flag = ROOT/"shutdown_after_cache.enabled"
if flag.exists(): flag.replace(ROOT/"shutdown_after_gpu_cache.fulfilled")
result = {"status": "PASS", "all_uploaded_sources_SHA_verified": True,
    "pinned_Python38_syntax_verified": True, "local_CPU_and_real_cache_evidence": "PASS",
    "server_initialization_and_optimizer_resume": "PASS", "GPU_smoke": "PENDING_4090_RESTART",
    "formal_training_started": False, "full_cache_audit": "WAIT_FOR_CACHE_READY_JSON",
    "source_manifest_SHA": hashlib.sha256((HERE/"execution_manifest.json").read_bytes()).hexdigest()}
(ROOT/"reports/TRAINER_PREPARATION_PASS.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result), flush=True)
