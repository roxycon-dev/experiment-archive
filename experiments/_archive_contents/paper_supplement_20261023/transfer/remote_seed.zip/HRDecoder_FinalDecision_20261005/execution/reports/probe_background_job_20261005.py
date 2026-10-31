"""Bounded process-lifetime diagnostic, no training or remote changes."""
import ctypes, json, os, sys, time
from ctypes import wintypes
from pathlib import Path
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.GetCurrentProcess.restype=wintypes.HANDLE
k.IsProcessInJob.argtypes=[wintypes.HANDLE,wintypes.HANDLE,ctypes.POINTER(wintypes.BOOL)]
k.IsProcessInJob.restype=wintypes.BOOL
class Limits(ctypes.Structure):
    _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),('flags',wintypes.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active_process_limit',wintypes.DWORD),('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
k.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
k.QueryInformationJobObject.restype=wintypes.BOOL
inside=wintypes.BOOL();ok=k.IsProcessInJob(k.GetCurrentProcess(),None,ctypes.byref(inside));v=Limits();limit_ok=k.QueryInformationJobObject(None,2,ctypes.byref(v),ctypes.sizeof(v),None)
row={'pid':os.getpid(),'ppid':os.getppid(),'in_job':bool(inside.value),'query_ok':bool(ok),'limit_query_ok':bool(limit_ok),'last_error':ctypes.get_last_error(),'limit_flags':v.flags,'kill_on_close':bool(v.flags&0x2000),'breakaway_allowed':bool(v.flags&0x800),'argv':sys.argv}
out=Path(__file__).resolve().parent/('job_probe_'+sys.argv[1]+'.json')
out.write_text(json.dumps(row),encoding='utf-8')
time.sleep(30)
row['bounded_probe_completed']=True
out.write_text(json.dumps(row),encoding='utf-8')
