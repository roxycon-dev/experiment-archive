"""Build/verify the small server handoff bundle; stdlib only, no training."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files():
    for node in ast.parse((HERE / "train.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE_FILES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("Missing training source manifest")


def verify(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        target = (HERE / name).resolve()
        if not target.is_relative_to(HERE) or digest(target) != expected:
            raise ValueError("Changed/missing handoff file: " + name)
    print(json.dumps({"status": "PASS_FILE_IDENTITY_ONLY", "files": len(manifest["files"]),
                      "source_sha256": manifest["source_sha256"], "server_preflight_completed": False}))


def build(output, checks):
    record = json.loads(checks.read_text(encoding="utf-8"))
    material = {name: digest(HERE / name) for name in source_files()}
    source_sha = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    units = record["unit_tests"]
    if (source_sha != record["source_sha256"] or units.get("status") != "PASS"
            or units.get("tests", 0) < 62 or any(units.get(k) != 0 for k in ("failures", "errors", "skipped"))):
        raise ValueError("Code changed or local checks incomplete; rerun the frozen-source checks")
    names = set(source_files()) | {"README.md", "run_checks.ps1", "prepare_delivery.py", "USER_SCOPE_AND_STOP.json"}
    names.update(p.relative_to(HERE).as_posix() for p in (HERE / "preparation_20261016").glob("*.md"))
    names.add("preparation_20261016/data_identity.json")
    names.update(p.relative_to(HERE).as_posix() for p in checks.parent.iterdir() if p.suffix in (".txt", ".json"))
    manifest = {"version": 3, "date": "2026-10-16", "source_sha256": source_sha,
                "status": "STATIC_PREPARATION_COMPLETE_SERVER_PREFLIGHT_PENDING", "formal_updates": 0,
                "test_accessed": False, "files": {name: digest(HERE / name) for name in sorted(names)},
                "external_assets": {"pretrained_sha256": "d2186c55782308392ae14cf179ac4840143b09abdbba81aa3f9274c795ebc960",
                                    "data": "accepted fused-label DDR_split_613, train454/val76/test227"},
                "excludes": ["pretrained weight bytes", "image/label data", "legacy v1 evidence", "training outputs"]}
    manifest_path = HERE / "preparation_20261016/delivery_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False); stream.write("\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(names):
            archive.write(HERE / name, name)
        archive.write(manifest_path, "preparation_20261016/delivery_manifest.json")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt delivery zip")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError("Packaged bytes differ: " + name)
    receipt = {"path": str(output.resolve()), "bytes": output.stat().st_size,
               "sha256": digest(output), "source_sha256": source_sha, "files": len(names)+1,
               "zip_entries_crc_and_sha256_verified": True}
    (HERE / "preparation_20261016/bundle_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--checks", type=Path,
                        default=HERE / "preparation_20261016/static_checks_spec_reaudit/static_checks.json")
    parser.add_argument("--output", type=Path,
                        default=HERE / "preparation_20261016/NativePyramid_SP_v3_server_20261016.zip")
    args = parser.parse_args()
    verify(args.verify) if args.verify else build(args.output, args.checks)
