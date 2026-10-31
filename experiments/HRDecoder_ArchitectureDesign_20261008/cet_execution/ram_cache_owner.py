"""Campaign-owned 96GiB memfd cache; disk targets and other processes untouched."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import json
import os
from pathlib import Path
import resource
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / 'reports/cet'
GIB = 1024**3
fds = []

def write(name, value):
    path = REPORTS / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)

def memfd(name):
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.memfd_create
    function.argtypes, function.restype = [ctypes.c_char_p, ctypes.c_uint], ctypes.c_int
    fd = function(name.encode(), 1)
    if fd < 0: raise OSError(ctypes.get_errno(), 'memfd_create')
    return fd

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] > 1400
    Path('/proc/self/oom_score_adj').write_text('1000')
    indexes = {m: json.loads((ROOT / 'cache/train' / m / 'index.json').read_text()) for m in ('A', 'U', 'S')}
    rows = json.loads((ROOT / 'metadata/data_identity.json').read_text())['splits']['train']['samples']
    entries = []
    for row in rows:
        for m in ('A', 'U', 'S'):
            path = ROOT / 'cache/train' / m / indexes[m]['entries'][row['sid']]['file']
            entries.append({'source': str(path), 'member': m, 'sid': row['sid'],
                            'raw_bytes': 4*row['hw'][0]*row['hw'][1]*4+128,
                            'compressed_bytes': path.stat().st_size})
    estimated = sum(e['raw_bytes'] if e['member'] != 'S' else e['compressed_bytes'] for e in entries)
    assert estimated < 96*GIB, 'All teachers in mixed RAM format exceed budget'
    selected_s = set()
    for e in sorted((e for e in entries if e['member'] == 'S'), key=lambda e: e['raw_bytes']-e['compressed_bytes'], reverse=True):
        delta = e['raw_bytes']-e['compressed_bytes']
        if estimated+delta <= 96*GIB:
            selected_s.add(e['source']); estimated += delta
    started = time.time()

    def copy(e):
        fmt = 'npy' if e['member'] != 'S' or e['source'] in selected_s else 'npz'
        fd = memfd('CET-'+e['member']+'-'+e['sid']); fds.append(fd)
        with os.fdopen(fd, 'wb', closefd=False) as dest:
            if fmt == 'npy':
                with zipfile.ZipFile(e['source']) as archive:
                    assert archive.namelist() == ['prob.npy']
                    with archive.open('prob.npy') as source: shutil.copyfileobj(source, dest, 1024**2)
            else:
                with open(e['source'], 'rb') as source: shutil.copyfileobj(source, dest, 1024**2)
        return e['source'], {'path': '/proc/%d/fd/%d' % (os.getpid(), fd), 'sid': e['sid'],
                             'member': e['member'], 'format': fmt, 'bytes': os.fstat(fd).st_size}
    mapping = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(copy, e) for e in entries]):
            source, entry = future.result(); mapping[source] = entry
            if len(mapping) % 20 == 0:
                write('RAM_PROGRESS.json', {'status': 'BUILDING', 'pid': os.getpid(),
                      'files': len(mapping), 'expected': 1362, 'seconds': time.time()-started})
    assert len(mapping) == 1362
    total = sum(e['bytes'] for e in mapping.values())
    assert total <= 96*GIB + 1024**2
    ready = {'status': 'READY', 'pid': os.getpid(), 'selected_images': 454,
             'files': 1362, 'bytes': total, 'all_teachers_in_RAM': True,
             'source_probabilities_unchanged': True, 'hash_rescan': False,
             'format_counts': {f: sum(e['format'] == f for e in mapping.values()) for f in ('npy', 'npz')},
             'seconds': time.time()-started, 'time_unix': time.time(), 'mapping': mapping}
    write('RAM_CACHE_READY.json', ready)
    print(json.dumps({k:v for k,v in ready.items() if k != 'mapping'}), flush=True)
    while not (ROOT / 'training_runs/cet/TRAINING_COMPLETE.json').exists() and not (REPORTS / 'STOP_RAM_OWNER.enabled').exists():
        time.sleep(2)

if __name__ == '__main__': main()
