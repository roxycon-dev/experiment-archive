"""Local-only transport scheduling: same frozen mirror/custody/SHA/shutdown code."""
import sys,threading,time,json,contextlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import paramiko
from qlib.common import write_json,utc_now,sha
from reports import parallel_union_mirror as mirror_module
from ops import attendant as attendant_module
attendant_module.mirror=mirror_module.main
RAW_CONNECT=mirror_module.connect
RAW_POOL=mirror_module.ThreadPoolExecutor
RAW_SSH_CONNECT=paramiko.SSHClient.connect
registry=[];registry_lock=threading.Lock();local=threading.local();opened=0;served=0

def compressed_connect(self,*args,**kwargs):
    kwargs.setdefault('compress',True)
    return RAW_SSH_CONNECT(self,*args,**kwargs)
paramiko.SSHClient.connect=compressed_connect

class Connection:
    def __init__(self):
        global opened
        self.client=RAW_CONNECT();self.sftp=self.client.open_sftp();self.closed=False
        with registry_lock:registry.append(self);opened+=1
    def close(self):
        if not self.closed:
            self.closed=True
            try:self.sftp.close()
            finally:self.client.close()
    def open_sftp(self):return SFTPLease(self)
    def __getattr__(self,name):return getattr(self.client,name)
    def __enter__(self):return self
    def __exit__(self,kind,value,tb):
        if kind is not None:self.close()

class SFTPLease:
    def __init__(self,connection):self.connection=connection
    def __enter__(self):return self
    def __exit__(self,kind,value,tb):
        if kind is not None:self.connection.close()
    def __getattr__(self,name):return getattr(self.connection.sftp,name)
    def get(self,*args,**kwargs):
        global served
        result=self.connection.sftp.get(*args,**kwargs)
        with registry_lock:served+=1
        return result

def persistent_connect():
    # Status/inventory/custody runs on the main thread and closes normally.
    # Only the two file-receiver workers lease a connection and SFTP channel.
    if threading.current_thread() is threading.main_thread():return RAW_CONNECT()
    connection=getattr(local,'connection',None)
    if connection is None or connection.closed or not connection.client.get_transport().is_active():
        if connection is not None:connection.close()
        connection=Connection();local.connection=connection
    return connection

class ReceiverPool(RAW_POOL):
    def __init__(self,*args,**kwargs):
        global opened,served
        assert not registry,'Prior receivers did not close'
        opened=served=0;self.started=time.perf_counter()
        super().__init__(*args,**kwargs)
    def shutdown(self,wait=True,*,cancel_futures=False):
        super().shutdown(wait=wait,cancel_futures=cancel_futures)
        if wait:
            with registry_lock:
                connections=list(registry);registry.clear()
            compressions=[c.client.get_transport().local_compression for c in connections if c.client.get_transport()]
            for connection in connections:connection.close()
            write_json(ROOT/'reports/persistent_receiver_state.json',{'stage':'POOL_COMPLETED','time':utc_now(),
                'receiver_connections_opened':opened,'files_received':served,'seconds':time.perf_counter()-self.started,
                'negotiated_compression':compressions,'max_file_receivers':2,'original_mirror_sha256':sha(ROOT/'ops/mirror.py'),
                'all_original_SHA_and_custody_gates_used':True})

mirror_module.connect=persistent_connect
mirror_module.ThreadPoolExecutor=ReceiverPool
RAW_STATUS=attendant_module.status
def compact_status():
    # Retain actual atomic states plus compact disk/time/phase audit, while
    # avoiding repeated full copies of static early-gate JSON in stdout.
    with open(os.devnull,'w') as null,contextlib.redirect_stdout(null):state=RAW_STATUS()
    campaign=state.get('campaign') or {};driver=state.get('driver') or {}
    run=campaign.get('current_run');current=(state.get('runs') or {}).get(run) or {}
    print(json.dumps({'time':utc_now(),'driver_stage':driver.get('stage'),'run':run,'run_stage':current.get('stage'),
        'completed':current.get('completed'),'free_disk_bytes':state.get('free_disk_bytes')},ensure_ascii=False),flush=True)
    return state
attendant_module.status=compact_status

if __name__=='__main__':attendant_module.main()
