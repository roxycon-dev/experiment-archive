"""Resume existing local-only helpers outside the app process lifetime."""
import sys,runpy,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import write_json,utc_now,sha
role=sys.argv[1]
entries={
 "primary":("primary_parallel_attendant.py",ROOT,"attendant_pid.json"),
 "second":("parallel_node2_attendant.py",ROOT/"parallel/node2","attendant_pid.json"),
 "coordinator":("parallel_single_seed_coordinator.py",ROOT,"parallel_coordinator_pid.json")}
helper,profile,pidfile=entries[role]
log=profile/"reports"/("independent_"+role+"_20261005_2030.log")
err=profile/"reports"/("independent_"+role+"_20261005_2030.err.log")
sys.stdout=log.open("a",encoding="utf-8",buffering=1);sys.stderr=err.open("a",encoding="utf-8",buffering=1)
import ctypes
from ctypes import wintypes
k=ctypes.WinDLL("kernel32",use_last_error=True)
k.GetCurrentProcess.restype=wintypes.HANDLE
k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
k.IsProcessInJob.restype=wintypes.BOOL
in_job=wintypes.BOOL();ok=k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(in_job))
write_json(profile/"reports"/("independent_process_isolation_"+role+".json"),{"pid":os.getpid(),"time":utc_now(),"query_ok":bool(ok),"in_job_object":bool(in_job.value),"required_CREATE_BREAKAWAY_FROM_JOB":True})
assert ok and not in_job.value,"Helper still belongs to a job object; do not report a durable recovery"
write_json(profile/"reports"/pidfile,{"pid":os.getpid(),"time":utc_now(),"role":role,"helper":"reports/"+helper,"runner":"reports/independent_local_attendant_runner.py","helper_sha256":sha(ROOT/"reports"/helper),"launcher":"Direct subprocess.Popen with CREATE_NO_WINDOW, CREATE_NEW_PROCESS_GROUP and CREATE_BREAKAWAY_FROM_JOB; verified outside job before helper invocation","training_unchanged":True})
print(json.dumps({"stage":"EXISTING_HELPER_RECOVERED","role":role,"pid":os.getpid(),"time":utc_now()}),flush=True)
runpy.run_path(str(ROOT/"reports"/helper),run_name="__main__")
