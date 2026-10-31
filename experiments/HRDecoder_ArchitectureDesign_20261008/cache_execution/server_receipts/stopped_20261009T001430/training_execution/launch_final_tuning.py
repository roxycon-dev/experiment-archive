import subprocess,json
root='/root/autodl-tmp/HRDecoder_v3_20261008'
results={}
for name,python in [('pause_at_checkpoint.py','/root/miniconda3/bin/python'),('ram_cache_mixed.py','/root/miniconda3/envs/hrdecoder_q_20261003/bin/python')]:
 f=open(root+'/reports/'+name+'.log','ab',buffering=0)
 p=subprocess.Popen([python,'-u',root+'/training_execution/'+name],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
 results[name]=p.pid
print(json.dumps(results))
