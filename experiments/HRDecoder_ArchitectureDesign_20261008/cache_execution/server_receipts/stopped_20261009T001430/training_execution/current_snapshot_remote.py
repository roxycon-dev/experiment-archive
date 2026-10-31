import json,pathlib,os
r=pathlib.Path('/root/autodl-tmp/HRDecoder_v3_20261008')
a=json.loads((r/'reports/GPU_CAMPAIGN_RUNNING.json').read_text())
j=json.loads((r/'reports/RAM_CACHE_READY.json').read_text())
print(json.dumps({'training_pid_alive':pathlib.Path('/proc/%d/cmdline'%a['pid']).exists(),
'training_cmdline':pathlib.Path('/proc/%d/cmdline'%a['pid']).read_text().replace('\0',' ') if pathlib.Path('/proc/%d/cmdline'%a['pid']).exists() else None,
'RAM_owner_alive':pathlib.Path('/proc/%d'%j['pid']).exists(),'RAM_files':len(j['mapping']),
'RAM_bytes':j['bytes'],'memory_current_bytes':int(pathlib.Path('/sys/fs/cgroup/memory.current').read_text()),
'shutdown_hold':(r/'reports/shutdown_hold.enabled').exists(),'last_checkpoint_bytes':(r/'training_runs/v3/last.pt').stat().st_size}))
