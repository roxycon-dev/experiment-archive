"""Remove only the two inventoried task authorizations after verified delivery.

Never shuts down a server or touches weights, predictions, or another user's jobs.
The reverse supervisor is stopped only after all evidence is local; its SSH child
is stopped on a delay so the .59 mutation receipt reaches this machine first.
"""
import hashlib
import json
from pathlib import Path
import shlex
import time

from training_remote_control import connect, execute, ROOT, LAN


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


COMMON = r'''
from pathlib import Path
import base64,datetime,hashlib,json,os,shlex,signal,socket,stat,subprocess,time
def digest(data):return hashlib.sha256(data).hexdigest()
def remove_exact_key(path,expected_line_sha,expected_fp,comment,preserve_sha=None):
    p=Path(path);before=p.read_bytes();original_stat=p.stat()
    lines=before.splitlines(keepends=True)
    matches=[i for i,line in enumerate(lines) if digest(line.rstrip(b'\r\n'))==expected_line_sha]
    assert len(matches)==1,('Task authorization identity changed',path,len(matches))
    index=matches[0];line=lines[index].rstrip(b'\r\n').decode()
    tokens=shlex.split(line);key_index=tokens.index('ssh-ed25519')
    fp=base64.b64encode(hashlib.sha256(base64.b64decode(tokens[key_index+1])).digest()).decode().rstrip('=')
    assert fp==expected_fp and tokens[-1]==comment
    assert 'expiry-time="20261027180000"' in line
    after=b''.join(lines[:index]+lines[index+1:])
    remaining=[digest(v.rstrip(b'\r\n')) for v in after.splitlines(keepends=True)]
    if preserve_sha:assert preserve_sha in remaining
    temporary=p.with_name(p.name+'.paper_supplement_cleanup.tmp')
    fd=os.open(str(temporary),os.O_WRONLY|os.O_CREAT|os.O_EXCL,stat.S_IMODE(original_stat.st_mode))
    with os.fdopen(fd,'wb') as f:
        f.write(after);f.flush();os.fsync(f.fileno())
    os.chmod(temporary,stat.S_IMODE(original_stat.st_mode))
    os.chown(temporary,original_stat.st_uid,original_stat.st_gid)
    assert p.read_bytes()==before,'Concurrent authorized_keys change; refusing replacement'
    os.replace(temporary,p)
    assert p.read_bytes()==after
    assert p.stat().st_mode==original_stat.st_mode
    assert (p.stat().st_uid,p.stat().st_gid)==(original_stat.st_uid,original_stat.st_gid)
    assert remaining==[digest(v.rstrip(b'\r\n')) for i,v in enumerate(lines) if i!=index]
    return {'path':str(p),'removed_full_line_sha256':expected_line_sha,'removed_public_key_fingerprint':fp,
            'before_file_sha256':digest(before),'after_file_sha256':digest(after),
            'all_other_line_bytes_order_and_file_mode_preserved':True,'remaining_line_sha256':remaining}
def proc(pid):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().split()
        return {'pid':pid,'ppid':int(fields[3]),'state':fields[2],'starttime':fields[21],
                'argv':(p/'cmdline').read_bytes().split(b'\0')[:-1],'uid':p.stat().st_uid}
    except FileNotFoundError:return None
def running(pid):
    row=proc(pid);return bool(row and row['state']!='Z')
'''


def remote(client, code, target='root', timeout=45):
    interpreter='/root/miniconda3/bin/python' if target=='root' else '/usr/bin/python3'
    result = execute(client, interpreter+' -c '+shlex.quote(code), target, timeout=timeout)
    assert result['exit_code'] == 0, result
    return json.loads(result['stdout'])


def main():
    receipt_path=ROOT/'results/campaign_completion_receipt.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['overall_campaign_complete'] is True
    assert receipt['status']=='COMPLETE_ALL_AUTHORIZED_EXPERIMENTS_AND_LOCAL_EVIDENCE'
    review=ROOT/'evidence/final_report_review.json'
    assert sha(review)==receipt['report_review_sha256']
    for path,digest in receipt['output_sha256'].items():assert sha(ROOT/path)==digest
    destination=ROOT/'evidence/completed_campaign_auth_cleanup.json'
    assert not destination.exists(),'Cleanup has already been recorded; do not repeat mutations'
    inventory=json.loads((ROOT/'evidence/temporary_auth_cleanup_inventory.json').read_text())
    baseline=json.loads((ROOT/'results/training_live_status.json').read_text())
    assert baseline['baseline']['paired_queue']['stage']=='ALL_SEVEN_EVALUATIONS_COMPLETE_RECOVERY_PENDING'
    assert not baseline['baseline']['paired_queue']['running'] and not baseline['baseline']['paired_queue']['failed']
    assert json.loads((ROOT/'results/terminal_backup_status.json').read_text())['all_terminal_backups_complete']

    # The task-specific supervisor and its existing child were read-only inventoried.
    prior=json.loads((ROOT/'evidence/precleanup_reverse_processes.json').read_text())
    parent=next(p for p in prior['task_reverse_processes'] if p['pid']==292079)
    child=next(p for p in prior['task_reverse_processes'] if p['pid']==292080)
    with connect() as client:
        preflight=remote(client,COMMON+r'''
print(json.dumps({'reverse_listener_open':socket.socket().connect_ex(('127.0.0.1',40560))==0,
                  'uptime_seconds':Path('/proc/uptime').read_text().split()[0]}))
''')
        assert preflight['reverse_listener_open']
        # All checks, including PID start times and exact argv, precede mutations.
        config={'task':LAN,'parent_starttime':parent['starttime'],'child_starttime':child['starttime'],
                'completion_receipt_sha256':sha(receipt_path)}
        code=COMMON+'\nconfig='+repr(config)+r'''
r=Path(config['task']);a=proc(292079);b=proc(292080)
assert a and b and a['starttime']==config['parent_starttime'] and b['starttime']==config['child_starttime']
assert a['uid']==b['uid']==os.getuid() and b['ppid']==a['pid']
assert a['argv']==[b'/usr/bin/python3',b'-u',str(r/'keep_reverse_59.py').encode()]
assert b['argv'][0]==b'ssh' and b'-N' in b['argv']
assert str(Path.home()/'.ssh/codex_supp_reverse_20261023').encode() in b['argv']
assert b'127.0.0.1:40560:127.0.0.1:22' in b['argv']
assert not (r/'logs/stop_reverse_tunnel').exists()
result=remove_exact_key(str(Path.home()/'.ssh/authorized_keys'),
 '280dc1f5ffc142f9131c44ce995b52abfdb5412f129770831e38efbe1de2b991',
 '5W4VoTigT3p+0MjYLVtxf1aC3Y+P5cZFViam1x1tSJc','paper_supplement_transfer_20261023')
(r/'logs/stop_reverse_tunnel').write_text(json.dumps({'campaign_complete_receipt_sha256':config['completion_receipt_sha256']}))
os.kill(a['pid'],signal.SIGTERM)
for i in range(30):
    if not running(a['pid']):break
    time.sleep(.1)
assert not running(a['pid'])
assert proc(b['pid'])['starttime']==config['child_starttime'] and running(b['pid'])
result.update(status='EXACT_TASK_TRANSFER_KEY_REMOVED_AND_OWN_SUPERVISOR_STOPPED',
              reverse_supervisor_pid=a['pid'],supervisor_verified_not_running=True,
              task_stop_flag_written=True,reverse_child_pid=b['pid'],
              reverse_child_starttime=config['child_starttime'],reverse_child_shutdown_delay_seconds=20,
              uptime_seconds=Path('/proc/uptime').read_text().split()[0],servers_shutdown=False)
(r/'logs/final_auth_cleanup_receipt.json').write_text(json.dumps(result,indent=2))
helper=r"""
from pathlib import Path
import json,os,signal,time
r=Path('/home/studentuser001/paper_supplement_20261023')
time.sleep(20)
p=Path('/proc/292080')
if p.exists():
    fields=(p/'stat').read_text().split()
    argv=(p/'cmdline').read_bytes().split(b'\0')[:-1]
    assert fields[21]=='CHILD_START_VALUE' and p.stat().st_uid==os.getuid()
    assert argv[0]==b'ssh' and b'127.0.0.1:40560:127.0.0.1:22' in argv
    assert b'/home/studentuser001/.ssh/codex_supp_reverse_20261023' in argv
    os.kill(292080,signal.SIGTERM)
(r/'logs/final_reverse_tunnel_stop.json').write_text(json.dumps({'status':'TASK_REVERSE_CHILD_SIGTERM_SENT','pid':292080,'servers_shutdown':False}))
""".replace('CHILD_START_VALUE',config['child_starttime'])
with (r/'logs/final_reverse_stop_helper.log').open('ab') as log:
    h=subprocess.Popen(['/usr/bin/python3','-c',helper],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
result['delayed_stop_helper_pid']=h.pid
print(json.dumps(result),flush=True)
'''
        lan_result=remote(client,code,'59',timeout=45)
        # Persist the acknowledged mutation before the delayed task tunnel exit.
        (ROOT/'evidence/acknowledged_59_auth_cleanup.json').write_text(json.dumps(lan_result,indent=2),encoding='utf-8')
        print(json.dumps({'59_acknowledged':lan_result['status'],'supervisor_stopped':True,'tunnel_stop_pending':True}),flush=True)
        for i in range(16):
            listener=remote(client,COMMON+r'''
s=socket.socket();s.settimeout(1);value=s.connect_ex(('127.0.0.1',40560));s.close()
print(json.dumps({'open':value==0}))
''')
            if not listener['open']:break
            time.sleep(2)
        assert not listener['open'],'Own reverse listener still open; leave root authorization for diagnosis'
        root_result=remote(client,COMMON+r'''
result=remove_exact_key('/root/.ssh/authorized_keys',
 '70f8d44cc436afa3f05646392fdd8f8d01c586fb46827d56023c6d03234888d2',
 'MyrKxQlMzI3PfZAd6VzPGMm5WB2AP/OrPJ6+I/F/3ks','paper_supplement_reverse_only_20261023',
 '713bd43634c17070c594759e46cf4c05a5bb60ceda2d595166983f6174ac3f93')
result.update(uptime_seconds=Path('/proc/uptime').read_text().split()[0],servers_shutdown=False)
print(json.dumps(result))
''')
        # This direct connection still works with the pre-existing user key.
        direct=execute(client,'/usr/bin/uptime',timeout=15)
        assert direct['exit_code']==0,direct
    result={'status':'COMPLETE_ONLY_TWO_TASK_AUTH_LINES_REMOVED_SERVERS_KEPT_ON',
            '59':lan_result,'root':root_result,'root_listener40560_verified_closed':True,
            '59_receipt_acknowledged_before_tunnel_exit':True,
            'root_existing_direct_user_key_preserved_and_connection_verified':True,
            'no_shutdown_command_issued':True,'weights_masks_caches_and_other_users_untouched':True,
            'completed_receipt_sha256_before_postcleanup_annotation':sha(receipt_path),
            'finished_unix':time.time()}
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    receipt['post_completion_auth_cleanup_sha256']=sha(destination)
    receipt['temporary_task_authorizations_removed']=True
    receipt_path.write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':result['status'],'servers_shutdown':False}),flush=True)


if __name__=='__main__':main()
