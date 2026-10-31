"""Shared, disposable decoded NPY bytes in memfd; persistent NPZ cache untouched."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import ctypes
import os
from pathlib import Path
import shutil
import time
import zipfile
import resource

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT/"reports"
GIB = 1024**3
descriptors = []

def memfd(name):
    # These bundled Python builds omit os.memfd_create despite kernel support.
    libc = ctypes.CDLL(None,use_errno=True)
    function = libc.memfd_create
    function.argtypes = [ctypes.c_char_p,ctypes.c_uint]
    function.restype = ctypes.c_int
    descriptor = function(name.encode(),1)
    if descriptor < 0: raise OSError(ctypes.get_errno(), 'memfd_create')
    return descriptor

def write(name, value):
    path = REPORTS/name
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    os.replace(temporary, path)

def main():
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 4096: resource.setrlimit(resource.RLIMIT_NOFILE,(min(4096,hard),hard))
    assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] > 1400
    limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
    while not (REPORTS/"RAM_TUNING_CHECKPOINT_SAVED.json").exists(): time.sleep(1)
    current = int(Path("/sys/fs/cgroup/memory.current").read_text())
    budget = limit-current-8*GIB
    rows = json.loads((ROOT/"metadata/data_identity.json").read_text())["splits"]["train"]["samples"]
    indexes = {member: json.loads((ROOT/"cache/train"/member/"index.json").read_text()) for member in ("A","U","S")}
    selected = sorted(rows, key=lambda row: row["hw"][0]*row["hw"][1])
    used = sum(3*(4*row["hw"][0]*row["hw"][1]*4+4096) for row in selected)
    for unused in range(60):
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        budget = limit-current-8*GIB
        if used < budget: break
        time.sleep(1)  # Wait for the stopped loader's workers and queues to exit.
    assert used < budget, "Decoded cache plus required runtime margin does not fit"
    # If pressure does arise, the disposable RAM owner is the first OOM victim;
    # workers fall back to the persistent original NPZ without changing targets.
    Path('/proc/self/oom_score_adj').write_text('1000')
    started = time.time()
    def extract(row):
        entries = []
        for member in ("A","U","S"):
            source = ROOT/"cache/train"/member/indexes[member]["entries"][row["sid"]]["file"]
            descriptor = memfd("HRDecoder-"+member+"-"+row["sid"])
            descriptors.append(descriptor)
            try:
                with zipfile.ZipFile(source) as archive:
                    assert archive.namelist() == ["prob.npy"]
                    info = archive.getinfo("prob.npy")
                    expected = 4*row["hw"][0]*row["hw"][1]*4
                    assert expected < info.file_size < expected+4096
                    with archive.open(info) as stream, os.fdopen(descriptor,"wb",closefd=False) as destination:
                        shutil.copyfileobj(stream,destination,length=1024**2)
                assert os.fstat(descriptor).st_size == info.file_size
                entries.append((str(source), {"path": "/proc/%d/fd/%d"%(os.getpid(),descriptor),
                    "sid": row["sid"], "member": member, "bytes": info.file_size}))
            except BaseException:
                os.close(descriptor)
                descriptors.remove(descriptor)
                raise
        return entries
    mapping = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(extract,row) for row in selected]
        for completed, future in enumerate(as_completed(futures),1):
            mapping.update(future.result())
            if completed == 1 or completed%10 == 0 or completed == len(selected):
                write("RAM_CACHE_PROGRESS.json", {"status":"BUILDING", "pid":os.getpid(),
                    "completed_images":completed, "selected_images":len(selected),
                    "budget_bytes":budget, "materialized_bytes":sum(row["bytes"] for row in mapping.values()),
                    "seconds":time.time()-started})
    ready = {"status":"READY", "pid":os.getpid(), "selected_images":len(selected),
        "total_images":454, "budget_bytes":budget,
        "bytes":sum(row["bytes"] for row in mapping.values()), "source_probabilities_unchanged":True,
        "storage":"shared memfd, read-only mmap in workers", "hash_rescan":False,
        "seconds":time.time()-started, "mapping":mapping}
    write("RAM_CACHE_READY.json",ready)
    print(json.dumps({key:value for key,value in ready.items() if key != "mapping"}),flush=True)
    evicted = []
    while not (ROOT/"training_runs/v3/TRAINING_COMPLETE.json").exists():
        memory = int(Path('/sys/fs/cgroup/memory.current').read_text())
        if memory > limit-5*GIB:
            reclaim = memory-(limit-8*GIB)
            freed = 0
            for row in sorted(selected,key=lambda row: row['hw'][0]*row['hw'][1],reverse=True):
                if row['sid'] in evicted: continue
                for source, entry in list(mapping.items()):
                    if entry['sid'] == row['sid']:
                        descriptor = int(entry['path'].rsplit('/',1)[-1])
                        os.close(descriptor); descriptors.remove(descriptor)
                        freed += entry['bytes']; del mapping[source]
                evicted.append(row['sid'])
                if freed >= reclaim: break
            ready.update(mapping=mapping, evicted_images=evicted,
                bytes=sum(entry['bytes'] for entry in mapping.values()))
            write('RAM_CACHE_READY.json',ready)
            write('RAM_CACHE_PRESSURE.json',{'memory_bytes':memory,'reclaimed_bytes':freed,
                'evicted_images':evicted,'targets_unchanged':True,'time_unix':time.time()})
        time.sleep(.2)
    for descriptor in descriptors: os.close(descriptor)

if __name__ == "__main__": main()
