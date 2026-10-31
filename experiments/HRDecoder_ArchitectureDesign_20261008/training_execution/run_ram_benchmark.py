from pathlib import Path
import os
import sys
import time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'local_preparation'))
from remote import connect
ROOT='/root/autodl-tmp/HRDecoder_v3_20261008'
client=connect()
with client.open_sftp() as ftp:
    for name in ('ram_loader.py','benchmark_ram_loader.py'):
        ftp.put(str(HERE/name),ROOT+'/training_execution/'+name)
while True:
    with client.open_sftp() as ftp:
        try: ftp.stat(ROOT+'/reports/RAM_CACHE_READY.json'); break
        except FileNotFoundError: pass
    time.sleep(10)
counts=os.environ.get('HRDECODER_BENCH_WORKERS','2,4,6')
assert all(value.isdigit() for value in counts.split(','))
command='cd '+ROOT+'/training_execution && PYTHONPATH='+ROOT+'/vendor/champion_repo OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HRDECODER_BENCH_WORKERS='+counts+' /root/miniconda3/bin/python -u benchmark_ram_loader.py > ../reports/ram_pipeline_benchmark.log 2>&1'
_,stdout,stderr=client.exec_command(command)
code=stdout.channel.recv_exit_status(); error=stderr.read().decode()
assert code==0,error+' Read reports/ram_pipeline_benchmark.log'
with client.open_sftp() as ftp:
    destination=HERE.parent/'cache_execution/server_receipts/reports'
    for name in ('RAM_PIPELINE_BENCHMARK_PASS.json','performance_config.json'):
        ftp.get(ROOT+'/reports/'+name,str(destination/name))
print('RAM_PIPELINE_BENCHMARK_PASS',flush=True)
client.close()
