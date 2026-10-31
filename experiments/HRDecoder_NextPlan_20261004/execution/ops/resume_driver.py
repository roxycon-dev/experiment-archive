"""Explicit operator-reviewed continuation, without repeating completed GPU acceptance."""
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import assert_source,read_json,utc_now,write_json


def main():
    os.chdir(ROOT);assert_source(ROOT)
    assert read_json(ROOT/'reports/performance_revision.json')['status']=='APPLIED'
    assert not (ROOT/'reports/campaign.lock').exists()
    descriptor=os.open(ROOT/'reports/driver.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(descriptor,str(os.getpid()).encode())
        write_json(ROOT/'reports/driver_state.json',{'stage':'CAMPAIGN','pid':os.getpid(),
            'resume_reason':'Approved performance revision, identity-checked u1600 continuation','time':utc_now()})
        subprocess.run([sys.executable,'-u','campaign.py','--resume-paused'],cwd=ROOT,check=True,
            env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
        write_json(ROOT/'reports/driver_state.json',{'stage':'READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN','time':utc_now()})
    except BaseException as error:
        write_json(ROOT/'reports/driver_state.json',{'stage':'TECHNICAL_PAUSE','reason':str(error),
            'automatic_retry':False,'time':utc_now()});raise
    finally:
        os.close(descriptor);(ROOT/'reports/driver.lock').unlink()


if __name__=='__main__':main()
