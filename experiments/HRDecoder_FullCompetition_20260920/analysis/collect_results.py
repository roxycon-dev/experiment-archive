# -*- coding: utf-8 -*-
"""把服务器上本次实验的脚本/测试/日志/契约/CSV/JSON/Markdown 回收到本地。

服务器保留原件（只读下载，不删不改）。不下载任何 checkpoint / 权重。
"""
import os
import sys
import posixpath

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _remote2 as rm  # noqa: E402

R = "/root/autodl-tmp/HRDecoder_FullCompetition_20260920"
LOCAL = r"D:\dataset\experiment\HRDecoder_FullCompetition_20260920\analysis_results\expert_complementarity_20260920"

# (远端路径, 本地子目录) ; 远端可以是文件或目录
ITEMS = [
    # ---- 脚本 ----
    (f"{R}/analysis/analyze_expert_complementarity.py", "scripts/analysis"),
    (f"{R}/analysis/expert_mask_contract.py", "scripts/analysis"),
    (f"{R}/analysis/export_hrplus_expert_masks.py", "scripts/analysis"),
    (f"{R}/analysis/export_fullmodel_masks.py", "scripts/analysis"),
    (f"{R}/analysis/rebuild_hrplus_contracts.py", "scripts/analysis"),
    (f"{R}/analysis/diag_fullmodel.py", "scripts/analysis"),
    (f"{R}/analysis/independent_recompute.py", "scripts/analysis"),
    (f"{R}/analysis/expert_eval_settings.json", "scripts/analysis"),
    (f"{R}/analysis/expert_eval_settings.example.json", "scripts/analysis"),
    (f"{R}/analysis/README_EXPERT_COMPLEMENTARITY.md", "scripts/analysis"),
    # ---- 测试 ----
    (f"{R}/tests/test_packed_mask_contract.py", "scripts/tests"),
    # ---- 日志 ----
    (f"{R}/expert_outputs/hrplus_export.log", "logs"),
    (f"{R}/expert_outputs/hrplus_exit_code.txt", "logs"),
    (f"{R}/logs_expert_comp", "logs"),
    ("/root/logs_run/test_contract.log", "logs"),
    ("/root/logs_run/independent_recompute.log", "logs"),
    # ---- 契约 ----
    (f"{R}/expert_outputs/hrplus", "contracts/hrplus"),
    (f"{R}/expert_outputs/fullmodel", "contracts/fullmodel"),
    (f"{R}/expert_outputs/_shared", "contracts/_shared"),
    (f"{R}/expert_outputs/REBUILD_REPORT.json", "manifests"),
    (f"{R}/expert_outputs/RUN_MANIFEST_hrplus.json", "manifests"),
    (f"{R}/expert_outputs/RUN_MANIFEST_fullmodel.json", "manifests"),
    # ---- 分析产物 ----
    (f"{R}/expert_outputs/analysis", "analysis_outputs"),
    # ---- 独立复算 ----
    ("/root/logs_run/independent_recompute.json", "independent_verification"),
]

DOWNLOAD_MASKS = os.environ.get("DOWNLOAD_MASKS", "1") == "1"


def walk(sftp, remote_dir):
    """返回 (远端文件路径, 相对 remote_dir 的路径) 列表。"""
    out = []
    for entry in sftp.listdir_attr(remote_dir):
        rp = posixpath.join(remote_dir, entry.filename)
        if entry.st_mode & 0o40000:  # dir
            for sub in walk(sftp, rp):
                out.append((sub[0], posixpath.join(entry.filename, sub[1])))
        else:
            out.append((rp, entry.filename))
    return out


def main():
    c = rm.connect()
    sftp = c.open_sftp()
    manifest = []
    n_files = 0
    n_bytes = 0

    targets = list(ITEMS)
    if DOWNLOAD_MASKS:
        for fam in ("hrplus", "fullmodel", "_shared"):
            targets.append((f"{R}/expert_outputs/{fam}", f"masks/{fam}"))

    for remote, sub in targets:
        try:
            st = sftp.stat(remote)
            is_dir = bool(st.st_mode & 0o40000)
        except IOError:
            print("  MISSING:", remote)
            manifest.append({"remote": remote, "status": "missing"})
            continue
        if is_dir:
            files = walk(sftp, remote)
            for rp, rel in files:
                if rp.endswith(".npz") and not DOWNLOAD_MASKS:
                    continue
                lp = os.path.join(LOCAL, sub, *rel.split("/"))
                os.makedirs(os.path.dirname(lp), exist_ok=True)
                sftp.get(rp, lp)
                sz = os.path.getsize(lp)
                n_files += 1
                n_bytes += sz
                manifest.append({"remote": rp, "local": os.path.relpath(lp, LOCAL),
                                 "bytes": sz, "status": "ok"})
            print("  DIR  %-58s %d files" % (sub, len(files)))
        else:
            lp = os.path.join(LOCAL, sub, posixpath.basename(remote))
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            sftp.get(remote, lp)
            sz = os.path.getsize(lp)
            n_files += 1
            n_bytes += sz
            manifest.append({"remote": remote, "local": os.path.relpath(lp, LOCAL),
                             "bytes": sz, "status": "ok"})
            print("  FILE %-58s %d bytes" % (sub + "/" + posixpath.basename(remote), sz))

    sftp.close()
    c.close()

    import json
    import datetime
    rep = {"collected_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "local_root": LOCAL, "n_files": n_files, "n_bytes": n_bytes,
           "download_masks": DOWNLOAD_MASKS, "server_originals_kept": True,
           "entries": manifest}
    os.makedirs(LOCAL, exist_ok=True)
    with open(os.path.join(LOCAL, "COLLECTION_MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print("\ncollected %d files, %.2f MB" % (n_files, n_bytes / 1e6))


if __name__ == "__main__":
    main()
