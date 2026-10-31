"""Explicitly authorized second-node transport; key-only, pinned known hosts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import paramiko
from qlib.common import read_json
from ops.remote import command

def connect_node2():
    s=read_json(ROOT/'configs/parallel.local.json')['server']
    assert s['assignment_confirmed'] and (s['host'],s['port'])==('connect.nmb1.seetacloud.com',40920)
    c=paramiko.SSHClient()
    c.load_host_keys(str(ROOT.parent.parent/'HRDecoder_NextPlan_20261004/execution/ops/known_hosts'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect(s['host'],port=s['port'],username='root',key_filename=s['ssh_key'],allow_agent=False,
              look_for_keys=False,timeout=15,banner_timeout=15,auth_timeout=15,compress=True)
    c.get_transport().set_keepalive(30)
    return c
