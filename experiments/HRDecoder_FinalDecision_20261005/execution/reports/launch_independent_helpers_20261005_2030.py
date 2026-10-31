"""Operational recovery only: invoke unchanged helpers outside the app job."""
import os, subprocess, sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qlib.common import write_json, utc_now, sha

expected = {
    "primary_parallel_attendant.py": "f76acf364bdcd56adc290c73ab7d47c9f6641bb817d1b0262b96483d8831578e",
    "parallel_node2_attendant.py": "037d0a4db1838ae991ace0d4c200a84b7ab1b6fc6d5ac1ae58ca0acc5ed7b2a5",
    "parallel_single_seed_coordinator.py": "267f4433f9ac5f0f886d062766a2a10057b00c1995b6ab92b48084e28f81ef5d",
}
for name, digest in expected.items():
    assert sha(ROOT / "reports" / name) == digest, name
flags = subprocess.CREATE_BREAKAWAY_FROM_JOB | subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
rows = []
for role in ("primary", "second", "coordinator"):
    process = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "reports/independent_local_attendant_runner.py"), role],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, close_fds=True,
    )
    time.sleep(2)
    profile = ROOT / "parallel/node2" if role == "second" else ROOT
    isolation = json.loads((profile / "reports" / ("independent_process_isolation_" + role + ".json")).read_text())
    assert isolation["pid"] == process.pid and isolation["query_ok"] and not isolation["in_job_object"], isolation
    assert process.poll() is None, (role, process.returncode)
    rows.append({"role": role, "pid": process.pid, "isolation": isolation})
    write_json(ROOT / "reports/independent_helper_launch_20261005_2030.json", {
        "time": utc_now(), "launcher_pid": os.getpid(), "creationflags": flags,
        "stage": "PARTIAL_LAUNCH_VERIFIED" if len(rows) < 3 else "ALL_THREE_OUTSIDE_APP_JOB_VERIFIED",
        "helpers": rows, "unchanged_helper_sha256": expected, "training_unchanged": True,
    })
print(json.dumps(rows))
