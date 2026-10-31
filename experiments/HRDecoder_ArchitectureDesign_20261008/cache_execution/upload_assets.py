"""Upload task assets with temporary names; authentication uses the task SSH key."""
from pathlib import Path
import json
import posixpath
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local_preparation"))
from remote import connect

REMOTE = "/root/autodl-tmp/HRDecoder_v3_20261008"
EXP = HERE.parents[1]
READY = EXP / "HRDecoder_Ready_20261002"
FINAL = EXP / "HRDecoder_FinalDecision_20261005/execution"

def mkdirs(sftp, path):
    parts = path.split("/")
    at = ""
    for part in parts:
        if not part: continue
        at += "/" + part
        try: sftp.stat(at)
        except FileNotFoundError: sftp.mkdir(at)

def main():
    client = connect()
    sftp = client.open_sftp()
    started = time.monotonic()
    try:
        mkdirs(sftp, REMOTE)
        assets = [(HERE / "teacher_sources.tar.gz", "teacher_sources.tar.gz")]
        for name in ["cache_common.py", "export_A.py", "export_U.py", "export_S.py", "audit_inputs.py", "cache_audit.py", "cache_pipeline.sh"]:
            if (HERE / name).exists(): assets.append((HERE / name, name))
        assets.append((HERE / "inference_weight_receipt.json", "metadata/inference_weight_receipt.json"))
        assets.append((READY / "best_pipeline/weights/H4_best.pth", "weights/H4_best.pth"))
        assets.append((FINAL / "server_results/runs/S_seed42/val_u20000/007-1774-100.npz", "references/S20_007-1774-100.npz"))
        for local, relative in assets:
            dest = REMOTE + "/" + relative
            mkdirs(sftp, posixpath.dirname(dest))
            if relative.startswith("weights/"):
                try:
                    if sftp.stat(dest).st_size == local.stat().st_size:
                        print(json.dumps({"existing": relative}), flush=True)
                        continue
                except FileNotFoundError: pass
            last = [time.monotonic()]
            def progress(done, total):
                if time.monotonic() - last[0] >= 20:
                    print(json.dumps({"upload": relative, "done": done, "total": total}), flush=True)
                    last[0] = time.monotonic()
            sftp.put(str(local), dest + ".uploading", callback=progress)
            sftp.posix_rename(dest + ".uploading", dest)
            print(json.dumps({"uploaded": relative, "bytes": local.stat().st_size}), flush=True)
            if relative == "teacher_sources.tar.gz":
                command = "cd " + REMOTE + " && /root/miniconda3/bin/python -c \"import tarfile; tarfile.open('teacher_sources.tar.gz').extractall('.')\""
                _, out, err = client.exec_command(command)
                code = out.channel.recv_exit_status()
                if code: raise RuntimeError(err.read().decode())
        mkdirs(sftp, REMOTE + "/weights")
        large = [(READY / "best_pipeline/weights" / name, "weights/" + name)
                 for name in ["B_weights.pth", "D_student_2000.pth", "E4_weights.pth", "E_1500.pth"]]
        large += [(HERE / "inference_weights" / name, "weights/" + name)
                  for name in ["U0_inference.pth", "S20_ema_inference.pth"]]
        def transfer(item):
            local, relative = item
            conn = connect()
            ftp = conn.open_sftp()
            destination = REMOTE + "/" + relative
            try:
                try:
                    if ftp.stat(destination).st_size == local.stat().st_size:
                        print(json.dumps({"existing": relative}), flush=True)
                        return
                except FileNotFoundError: pass
                temporary = destination + ".uploading"
                try: done = ftp.stat(temporary).st_size
                except FileNotFoundError: done = 0
                assert done <= local.stat().st_size
                with local.open("rb") as src, ftp.open(temporary, "ab" if done else "wb") as dst:
                    src.seek(done)
                    dst.set_pipelined(True)
                    last = time.monotonic()
                    while True:
                        block = src.read(256 * 1024)
                        if not block: break
                        dst.write(block)
                        done += len(block)
                        if time.monotonic() - last >= 20:
                            print(json.dumps({"upload": relative, "done": done, "total": local.stat().st_size}), flush=True)
                            last = time.monotonic()
                assert ftp.stat(temporary).st_size == local.stat().st_size
                ftp.posix_rename(temporary, destination)
                print(json.dumps({"uploaded": relative, "bytes": done}), flush=True)
            finally:
                ftp.close(); conn.close()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(transfer, large))
        print(json.dumps({"status": "UPLOAD_COMPLETE", "seconds": time.monotonic() - started}), flush=True)
    finally:
        sftp.close(); client.close()

if __name__ == "__main__": main()
