"""Adopt decoded A/U and selected S; keep every remaining S NPZ in shared RAM."""
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import time
import zipfile
from ram_probability_cache import memfd, write, GIB

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT/'reports'
fds = []

def main():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    Path('/proc/self/oom_score_adj').write_text('1000')
    old = json.loads((REPORTS/'RAM_CACHE_READY.json').read_text())
    assert b'ram_probability_cache.py' in Path('/proc/%d/cmdline'%old['pid']).read_bytes()
    rows = json.loads((ROOT/'metadata/data_identity.json').read_text())['splits']['train']['samples']
    indexes = {m: json.loads((ROOT/'cache/train'/m/'index.json').read_text()) for m in ('A','U','S')}
    entries = []
    for row in rows:
        for m in ('A','U','S'):
            source = ROOT/'cache/train'/m/indexes[m]['entries'][row['sid']]['file']
            entries.append({'source':str(source),'sid':row['sid'],'member':m,
                'raw_bytes':4*row['hw'][0]*row['hw'][1]*4+128,'compressed_bytes':source.stat().st_size})
    estimated = sum(e['raw_bytes'] if e['member']!='S' else e['compressed_bytes'] for e in entries)
    raw_s = set()
    for e in sorted((e for e in entries if e['member']=='S'), key=lambda e:e['raw_bytes']-e['compressed_bytes'], reverse=True):
        delta = e['raw_bytes']-e['compressed_bytes']
        if estimated+delta <= 96*GIB and e['source'] in old['mapping']:
            raw_s.add(e['source']); estimated += delta
    mapping = {}
    # Duplicate descriptors, not tensor bytes, before retiring the previous owner.
    for e in entries:
        if e['member']=='S' and e['source'] not in raw_s: continue
        previous = old['mapping'].get(e['source'])
        if previous is None: continue
        fd = os.open(previous['path'], os.O_RDONLY); fds.append(fd)
        assert os.fstat(fd).st_size == previous['bytes']
        mapping[e['source']] = dict(previous, path='/proc/%d/fd/%d'%(os.getpid(),fd),format='npy')
    os.kill(old['pid'], signal.SIGTERM)
    started = time.time()
    for e in entries:
        if e['source'] in mapping: continue
        fmt = 'npy' if e['member']!='S' or e['source'] in raw_s else 'npz'
        fd = memfd('HRDecoder-mixed-'+e['member']+'-'+e['sid']); fds.append(fd)
        with os.fdopen(fd,'wb',closefd=False) as destination:
            if fmt == 'npy':
                with zipfile.ZipFile(e['source']) as archive:
                    assert archive.namelist() == ['prob.npy']
                    with archive.open('prob.npy') as source: shutil.copyfileobj(source,destination,1024**2)
            else:
                with open(e['source'],'rb') as source: shutil.copyfileobj(source,destination,1024**2)
        mapping[e['source']] = {'path':'/proc/%d/fd/%d'%(os.getpid(),fd), 'sid':e['sid'],
            'member':e['member'],'format':fmt,'bytes':os.fstat(fd).st_size}
        if len(mapping)%20==0:
            write('RAM_MIXED_PROGRESS.json',{'status':'BUILDING','pid':os.getpid(),
                'files':len(mapping),'expected':1362,'seconds':time.time()-started})
    assert len(mapping)==1362
    for e in mapping.values(): assert Path(e['path']).stat().st_size == e['bytes']
    ready = {'status':'READY','pid':os.getpid(),'selected_images':454,'total_images':454,
        'files':len(mapping),'all_teachers_in_RAM':True,'bytes':sum(e['bytes'] for e in mapping.values()),
        'storage':'shared memfd; A/U decoded NPY, S lossless NPZ or decoded NPY',
        'format_counts':{f:sum(e['format']==f for e in mapping.values()) for f in ('npy','npz')},
        'source_probabilities_unchanged':True,'hash_rescan':False,'mapping':mapping,
        'seconds':time.time()-started,'time_unix':time.time()}
    assert ready['bytes'] < 96*GIB+1024**2
    write('RAM_CACHE_READY.json',ready)
    write('RAM_MIXED_READY_SUMMARY.json',{k:v for k,v in ready.items() if k!='mapping'})
    print(json.dumps({k:v for k,v in ready.items() if k!='mapping'}),flush=True)
    while not (ROOT/'training_runs/v3/TRAINING_COMPLETE.json').exists(): time.sleep(1)

if __name__=='__main__': main()
