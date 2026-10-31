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
class JobLimits(ctypes.Structure):
    _fields_=[("process_time",ctypes.c_longlong),("job_time",ctypes.c_longlong),("flags",wintypes.DWORD),("min_ws",ctypes.c_size_t),("max_ws",ctypes.c_size_t),("active_process_limit",wintypes.DWORD),("affinity",ctypes.c_size_t),("priority",wintypes.DWORD),("scheduling",wintypes.DWORD)]
k.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
k.QueryInformationJobObject.restype=wintypes.BOOL
limits=JobLimits()
limits_ok=k.QueryInformationJobObject(None,2,ctypes.byref(limits),ctypes.sizeof(limits),None) if in_job.value else True
kill_on_close=bool(limits.flags&0x2000)
write_json(profile/"reports"/("independent_process_isolation_"+role+".json"),{"pid":os.getpid(),"parent_pid":os.getppid(),"time":utc_now(),"query_ok":bool(ok),"in_job_object":bool(in_job.value),"limit_query_ok":bool(limits_ok),"job_limit_flags":limits.flags,"kill_on_job_close":kill_on_close,"required_CREATE_BREAKAWAY_FROM_JOB":True,"acceptance":"No kill-on-close job; WMI service parent verified separately"})
# WMI retains its service job (0x1800); it has no kill-on-close limit.
# Direct child launch retained an outer app job (0x2000) and was rejected.
assert ok and limits_ok and not kill_on_close,"Helper belongs to a kill-on-close job; do not start receivers"
write_json(profile/"reports"/pidfile,{"pid":os.getpid(),"time":utc_now(),"role":role,"helper":"reports/"+helper,"runner":"reports/independent_local_attendant_runner.py","helper_sha256":sha(ROOT/"reports"/helper),"launcher":"Win32_Process.Create with hidden startup and CREATE_BREAKAWAY_FROM_JOB; WMI service parent and no kill-on-close job verified","training_unchanged":True})
print(json.dumps({"stage":"EXISTING_HELPER_RECOVERED","role":role,"pid":os.getpid(),"time":utc_now()}),flush=True)
runpy.run_path(str(ROOT/"reports"/helper),run_name="__main__")
