"""Persist training across SSH disconnects; checkpoint, record outcome, then power off."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT/"training_runs/v3"
REPORTS = ROOT/"reports"
PYTHON = "/root/miniconda3/bin/python"

def write(name, data):
    destination = REPORTS/name
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2)+"\n")
    os.replace(temp, destination)

def main():
    REPORTS.mkdir(exist_ok=True)
    lock = open(REPORTS/"gpu.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, PYTHONPATH=str(ROOT/"vendor/champion_repo"),
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4",
        HRDECODER_WORKERS="12", HRDECODER_PREFETCH="1")
    started = time.time()
    command = [PYTHON, "-u", "train_performance.py", "--train"]
    if (RUN/"last.pt").exists(): command.append("--resume")
    with open(REPORTS/"gpu_telemetry.csv", "a", buffering=1) as telemetry, \
            open(REPORTS/"v3_training.log", "a", buffering=1) as log:
        monitor = subprocess.Popen(["nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits", "--loop=2"], stdout=telemetry, stderr=log)
        process = subprocess.Popen(command, cwd=str(HERE), env=environment, stdout=log, stderr=log)
        write("GPU_CAMPAIGN_RUNNING.json", {"status": "RUNNING", "pid": process.pid,
            "controller_pid": os.getpid(), "start_unix": started,
            "updates": 40000, "batch": 2, "canvas": 2048,
            "shutdown_after_completion": True, "shutdown_after_fatal_failure_grace_seconds": 600,
            "resume": "--resume" in command, "data_hash_rescan": False})
        code = process.wait()
        monitor.terminate()
        try: monitor.wait(timeout=10)
        except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()
    result = {"exit_code": code, "start_unix": started, "end_unix": time.time(),
        "elapsed_hours": (time.time()-started)/3600, "formal_run": "v3"}
    complete = RUN/"TRAINING_COMPLETE.json"
    passed = code == 0 and complete.exists()
    if passed:
        record = json.loads(complete.read_text())
        passed = record.get("status") == "COMPLETE" and record.get("completed") == 40000 and (RUN/"last.pt").exists()
        result["training_completion"] = record
        result["validation_results"] = {p.name: json.loads(p.read_text()) for p in sorted(RUN.glob("val_*.json"))}
        if passed:
            try:
                import torch
                saved = torch.load(RUN/'last.pt',map_location='cpu')
                endpoint = result['validation_results']['val_40000.json']
                passed = saved['completed']==40000 and saved['identity_SHA']==record['identity_SHA']
                passed = passed and all(key in saved for key in ('model','optimizer','scaler','rng'))
                passed = passed and math.isfinite(endpoint['mIoU_percent'])
                result['persistent_checkpoint_verification'] = {'status':'PASS' if passed else 'FAIL',
                    'loaded_completed':saved['completed'],'identity_matches':saved['identity_SHA']==record['identity_SHA'],
                    'path':str(RUN/'last.pt'),'bytes':(RUN/'last.pt').stat().st_size,
                    'optimizer_and_RNG_present':all(key in saved for key in ('optimizer','rng')),
                    'final_validation':endpoint}
                del saved
            except Exception as error:
                passed = False; result['checkpoint_verification_error']=repr(error)
    result["status"] = "COMPLETE" if passed else "FAILED"
    write("GPU_CAMPAIGN_OUTCOME.json", result)
    if not passed:
        # Permit the active agent to repair a startup/runtime error; otherwise stop renting idle GPU.
        deadline = time.time()+600
        while time.time() < deadline:
            if (REPORTS/"shutdown_hold.enabled").exists():
                write("GPU_CAMPAIGN_REPAIR_HOLD.json", {"status": "WAITING_FOR_REPAIR", "time_unix": time.time()})
                return
            time.sleep(5)
    write("TRAINING_SHUTDOWN_COMMAND_ISSUED.json", {"authorized_by_user": True,
        "reason": "Training completed" if passed else "Fatal training exit; evidence saved and 10-minute repair grace expired",
        "time_unix": time.time(), "command": "/bin/bash /usr/bin/shutdown"})
    subprocess.run(["sync"], check=True)
    subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=False)

if __name__ == "__main__": main()
