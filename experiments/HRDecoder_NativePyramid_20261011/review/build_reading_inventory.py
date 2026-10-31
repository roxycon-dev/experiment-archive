"""Inventory and content-deduplicate historical prose; no ML/model imports."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
exclude = ["vendor", "repo", "repo_snapshot", "server_source", "node_modules"]
args = ["rg", "--files", "-g", "*.md", "-g", "!HRDecoder_NativePyramid_20261011/**",
        "-g", "!**/work/HRDecoder/**"]
for item in exclude:
    args += ["-g", f"!**/{item}/**"]
paths = sorted(subprocess.check_output(args, cwd=ROOT).decode("utf-8").splitlines())
groups = {}
for relative in paths:
    raw = (ROOT / relative).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig", errors="replace")
    item = groups.setdefault(digest, {"sha256": digest, "paths": [], "bytes": len(raw),
                                     "characters": len(text), "lines": len(text.splitlines())})
    item["paths"].append(relative.replace("\\", "/"))
early = ("autodl100/", "hrdecoder_plus3/", "HRDecoder_诊断调研_20260910/", "HRDecoder_审查轮_20260916/",
         "HRDecoder_MA残差头与集成_20260915-16/", "HRDecoder_FusionAware_20260917/", "HRDecoder_FullCompetition_20260920/")
later = ("HRDecoder_Ready_20261002/", "HRDecoder_Next_20261003/", "HRDecoder_Q_20261003/",
         "HRDecoder_NextPlan_20261004/", "HRDecoder_FinalDecision_20261005/")
for item in groups.values():
    path = item["paths"][0]
    item["primary_path"] = path
    if path.startswith(early):
        item["assigned"] = "architecture_audit"
    elif path.startswith("HRDecoder_ArchitectureDesign_20261008/"):
        item["assigned"] = "stability_audit"
    elif path.startswith(later):
        item["assigned"] = "protocol_audit"
    else:
        item["assigned"] = "root"
    item["is_directory_index"] = "01_全目录实验清单.md" in path
    item["review_status"] = "pending"
result = {"scope": "All rg-listed historical Markdown outside vendored code and the current output directory",
          "not_claimed": "No claim of reading every binary checkpoint, image, vendored source or raw log",
          "file_count": len(paths), "unique_content_count": len(groups), "entries": list(groups.values())}
(OUT / "reading_inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
for owner in ("root", "architecture_audit", "stability_audit", "protocol_audit"):
    rows = [row for row in groups.values() if row["assigned"] == owner]
    (OUT / f"reading_{owner}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(owner, len(rows), sum(r["characters"] for r in rows), "characters")
print("files", len(paths), "unique", len(groups))
