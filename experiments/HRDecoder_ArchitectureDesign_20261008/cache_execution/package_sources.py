"""Package exact existing teacher source, metadata and CUDA kernel; no weights."""
from pathlib import Path
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
READY = EXP / "HRDecoder_Ready_20261002"
FINAL = EXP / "HRDecoder_FinalDecision_20261005/execution"
QROOT = EXP / "HRDecoder_Q_20261003"

def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""): d.update(b)
    return d.hexdigest()

def main():
    files = []
    for local, remote in [(READY / "vendor/champion_repo", "vendor/champion_repo"),
                          (READY / "legacy", "vendor/legacy"),
                          (READY / "trainer", "vendor/ready/trainer"),
                          (READY / "tools", "vendor/ready/tools"),
                          (QROOT / "vendor/dinov3", "vendor/dinov3"),
                          (FINAL / "qlib", "vendor/final/qlib")]:
        for path in local.rglob("*"):
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".md", ".txt", ".yaml", ".yml", ".cpp", ".cu", ".h", ".cuh"}:
                files.append((path, remote + "/" + path.relative_to(local).as_posix()))
    for name in ["final_model.py", "spatial.py"]:
        files.append((FINAL / name, "vendor/final/" + name))
    files.extend([
        (FINAL / "server_results/vendor/msda_build/MultiScaleDeformableAttention.so", "vendor/msda_build/MultiScaleDeformableAttention.so"),
        (FINAL / "server_results/reports/data_identity.json", "metadata/data_identity.json"),
        (FINAL / "server_results/reports/final_selection_lock.json", "metadata/final_selection_lock.json"),
        (READY / "configs/champion_server.json", "metadata/original_champion_settings.json"),
        (READY / "best_pipeline/local_weight_manifest.json", "metadata/local_weight_manifest.json"),
        (READY / "configs/resolved/U0_historical_native_export.yaml", "metadata/U0_native_export.yaml"),
        (HERE.parent / "05_v3_机制重构与反驳审查.md", "metadata/design_v3.md"),
    ])
    source_manifest = {target: sha(source) for source, target in files}
    (HERE / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    files.append((HERE / "source_manifest.json", "metadata/source_manifest.json"))
    with tarfile.open(HERE / "teacher_sources.tar.gz", "w:gz") as archive:
        for source, target in files: archive.add(source, arcname=target, recursive=False)
    print(json.dumps({"files": len(files), "bytes": (HERE / "teacher_sources.tar.gz").stat().st_size}))

if __name__ == "__main__": main()
