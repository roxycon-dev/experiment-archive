"""One server-side job; local backup acknowledgement is required before shutdown."""
import os
from pathlib import Path
import subprocess
import sys
from qlib.common import utc_now,write_json
ROOT=Path(__file__).resolve().parent


def main():
    os.chdir(ROOT)
    write_json(ROOT/'reports/driver_state.json',{'stage':'PREFLIGHT','pid':os.getpid(),'time':utc_now()})
    subprocess.run([sys.executable,'-u','ops/prepare_space.py'],check=True)
    subprocess.run([sys.executable,'-u','preflight_gpu.py'],check=True)
    write_json(ROOT/'reports/driver_state.json',{'stage':'CAMPAIGN','pid':os.getpid(),'time':utc_now()})
    subprocess.run([sys.executable,'-u','campaign.py'],check=True)
    write_json(ROOT/'reports/driver_state.json',{'stage':'READY_FOR_LOCAL_BACKUP_AND_SHUTDOWN','time':utc_now()})


if __name__=='__main__':
    descriptor=None
    try:
        descriptor=os.open(ROOT/'reports/driver.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        os.write(descriptor,str(os.getpid()).encode())
        main()
    except BaseException as error:
        if descriptor is not None:
            write_json(ROOT/'reports/driver_state.json',{'stage':'TECHNICAL_PAUSE','reason':str(error),'automatic_retry':False,'time':utc_now()})
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor);(ROOT/'reports/driver.lock').unlink()
