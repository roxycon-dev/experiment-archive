"""Original completion/shutdown listener with bounded local SFTP receive requests."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops.transfer_acceleration import install
install()
from ops.attendant import main
from qlib.common import utc_now,write_json


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write_json(ROOT/'reports/attendant_state.json',{'stage':'TECHNICAL_PAUSE',
            'reason':str(error),'automatic_retry':False,'time':utc_now()})
        raise
