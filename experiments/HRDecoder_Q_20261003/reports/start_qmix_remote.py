"""Launch one authorized registered run after immutable external acceptance."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qlib.common import assert_source,read_json,write_json,sha,utc_now

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',choices=['Q_DDR','Q_MIX'],default='Q_MIX')
    ap.add_argument('--seed',type=int,choices=[42,2024],default=42)
    ap.add_argument('--endpoint',type=int,choices=[16000,24000],default=16000)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    assert ROOT==Path('/root/autodl-tmp/HRDecoder_Q_20261003')
    os.chdir(ROOT)
    py='/root/miniconda3/envs/hrdecoder_q_20261003/bin/python'
    source=assert_source(ROOT)
    protocol=read_json(ROOT/'reports/qmix_continuation_protocol.json')
    assert sha(ROOT/'reports/final_selection.json')==protocol['existing_DDR_final_lock_sha256']
    mix_runtime=read_json(ROOT/'configs/runtime.qmix.server.json')
    assert sha(ROOT/'configs/runtime.qmix.server.json')==protocol['runtime_qmix_sha256']
    assert sha(mix_runtime['external_manifest'])==protocol['external_manifest_server_sha256']
    assert read_json(ROOT/'reports/qmix_remote_asset_receipt.json')['status']=='ALL_REMOTE_ASSET_SHA_VERIFIED'
    for pname in ('formal.pid','qmix_launcher.pid'):
        p=ROOT/'reports'/pname
        if p.exists():
            cmd=Path('/proc')/p.read_text().strip()/'cmdline'
            if cmd.exists():
                value=cmd.read_bytes()
                if b'train_run.py' in value or b'start_qmix_remote.py' in value:
                    # This launcher may have been recorded by its caller; exempt self only.
                    assert int(p.read_text())==os.getpid(), 'Another formal run or launcher is already live'
    assert shutil.disk_usage(ROOT).free>8*1024**3, 'Late cache/checkpoint headroom below registered minimum'
    write_json(ROOT/'reports/qmix_launcher_state.json',{'stage':'IDENTITIES_VERIFIED_GENERATING_FROZEN_STREAM',
        'arm':args.arm,'seed':args.seed,'endpoint':args.endpoint,'updated_at':utc_now()})
    env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    subprocess.run([py,'reports/advance_qmix_campaign.py'],check=True,env=env)
    campaign=read_json(ROOT/'reports/campaign_state.json')
    assert not campaign['architecture_stopped']
    assert args.endpoint==campaign['common_endpoint']
    if args.seed==2024:assert campaign['review_seed_authorized'] and campaign['discovery_complete']
    runtime='configs/runtime.qmix.server.json' if args.arm=='Q_MIX' else 'configs/runtime.server.json'
    stream=ROOT/'reports'/f'stream_{args.arm}_seed{args.seed}.jsonl'
    if not stream.exists():
        subprocess.run([py,'-u','make_stream.py','--runtime',runtime,'--arm',args.arm,'--seed',str(args.seed)],check=True,env=env)
    info=read_json(stream.with_suffix('.summary.json'))
    assert info['status']=='PASS' and sha(stream)==info['file_sha256']
    if args.arm=='Q_MIX':
        assert info['exposures']['16000']=={'DDR':80000,'IDRiD':24000,'MAPLES':24000}
    gate=read_json(ROOT/'reports/formal_readiness.json')
    original=ROOT/'reports/formal_readiness.original_DDR.json'
    if not original.exists():write_json(original,gate)
    assert gate['source_sha256']==source['sha256'] and gate['status']=='READY'
    gate.update(runtime_sha256=sha(ROOT/runtime),created_at=utc_now(),
        external_manifest_sha256=protocol['external_manifest_server_sha256'],
        acceptance_basis='Existing identical architecture GPU preflight reused; new external file/geometry/validity/dedup/overlay acceptance passed',
        formal_training_started=False)
    write_json(ROOT/'reports/formal_readiness.json',gate)
    run=ROOT/'runs'/f'{args.arm}_seed{args.seed}'
    assert (run/'last.pth').exists()==args.resume
    command=[py,'-u','train_run.py','--runtime',runtime,'--arm',args.arm,'--seed',str(args.seed),'--endpoint',str(args.endpoint)]
    if args.resume:command.append('--resume')
    log_path=ROOT/'reports'/f'formal_{args.arm}_seed{args.seed}.log'
    with log_path.open('ab' if args.resume else 'wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 cwd=ROOT,env=env,start_new_session=True)
    (ROOT/'reports/formal.pid').write_text(str(process.pid)+'\n')
    receipt={'stage':'FORMAL_PROCESS_LAUNCHED','pid':process.pid,'arm':args.arm,'seed':args.seed,'endpoint':args.endpoint,
        'runtime_sha256':sha(ROOT/runtime),'external_manifest_sha256':protocol['external_manifest_server_sha256'],
        'continuation_protocol_sha256':sha(ROOT/'reports/qmix_continuation_protocol.json'),
        'stream_sha256':info['file_sha256'],'existing_DDR_final_lock_unchanged':True,'created_at':utc_now()}
    write_json(ROOT/'reports/qmix_formal_launch_receipt.json',receipt)
    write_json(ROOT/'reports/qmix_launcher_state.json',receipt)
    print('FORMAL_PROCESS_LAUNCHED',process.pid,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write_json(ROOT/'reports/qmix_launcher_state.json',{'stage':'TECHNICAL_PAUSE','reason':str(error),
                   'automatic_restart':False,'updated_at':utc_now()})
        raise
