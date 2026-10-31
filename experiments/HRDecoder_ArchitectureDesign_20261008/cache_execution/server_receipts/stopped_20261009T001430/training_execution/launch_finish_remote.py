import subprocess
root='/root/autodl-tmp/HRDecoder_v3_20261008'
f=open(root+'/reports/finish_performance_tuning.log','ab',buffering=0)
p=subprocess.Popen(['/root/miniconda3/bin/python','-u',root+'/training_execution/finish_performance_tuning.py'],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
print(p.pid)
