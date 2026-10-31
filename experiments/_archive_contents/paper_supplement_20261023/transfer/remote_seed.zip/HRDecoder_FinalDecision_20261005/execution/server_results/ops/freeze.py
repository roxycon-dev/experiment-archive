"""Freeze scientific code/data/config and build one reviewed deployment package."""
import sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import read_json,write_json,sha,digest,utc_now

def main():
    for file in ('cpu_acceptance.json','window_acceptance.json','protocol_acceptance.json','storage_acceptance.json'):
        assert read_json(ROOT/'reports'/file)['status']=='PASS'
    manifest=read_json(ROOT/'reports/TJDR_manifest.json');assert manifest['status']=='READY'
    read_json(ROOT/'reports/stream_identity.json')
    files=sorted([*ROOT.glob('*.py'),*(ROOT/'qlib').glob('*.py'),*(ROOT/'tests').glob('*.py'),*(ROOT/'ops').glob('*.py'),
        *(ROOT/'vendor/msda_compat/src').rglob('*.cpp'),*(ROOT/'vendor/msda_compat/src').rglob('*.cu'),*(ROOT/'vendor/msda_compat/src').rglob('*.h'),
        *[p for p in (ROOT/'configs').iterdir() if p.is_file() and not p.name.startswith('runtime.')]])
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    write_json(ROOT/'reports/source_identity.json',{'sha256':digest(hashes),'files':hashes,'time':utc_now()})
    package=ROOT/'reports/HRDecoder_20261005_prepared.zip'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,p.relative_to(ROOT).as_posix())
        for sub in ('configs','reports'):
            for p in (ROOT/sub).iterdir():
                if not p.is_file() or p==package or p.suffix not in ('.json','.jsonl','.md','.log'):continue
                if p.name in ('package_identity.json','source_identity.json'):continue
                if p in files:continue
                z.write(p,p.relative_to(ROOT).as_posix())
        z.write(ROOT/'reports/source_identity.json','reports/source_identity.json')
    write_json(ROOT/'reports/package_identity.json',{'sha256':sha(package),'bytes':package.stat().st_size,'time':utc_now(),
        'inputs_uploaded_separately_with_manifest_SHA':True,'server_runtime_assignment_finalized_before_GPU_gate':True})
    assigned=read_json(ROOT/'configs/runtime.local.json')['server']['assignment_confirmed']
    write_json(ROOT/'reports/preparation_status.json',{'stage':'LOCAL_READY_SERVER_ASSIGNED_GPU_GATE_PENDING' if assigned else 'LOCAL_READY_AWAITING_USER_SERVER_ASSIGNMENT','time':utc_now(),
        'data_acceptance':'PASS','cpu_architecture_acceptance':'PASS','packed_geometry_acceptance':'PASS',
        'saved_streams':'READY','source_sha256':digest(hashes),'paid_gpu_started':False,'server_assigned':assigned,
        'remaining_server_gates':['native1024 and parent fullval','CUDA kernel/reference forward/backward','full trainable gradients and EMA/RNG resume',
            '100complete updates and fullval timing per C/S/D/SD','audited100GiB initial local reserve and bounded remote endpoint custody','verified P0/R identity']})
    print('LOCAL_PREPARATION_READY',sha(package))

if __name__=='__main__':main()
