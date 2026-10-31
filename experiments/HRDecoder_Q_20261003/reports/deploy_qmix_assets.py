"""Freeze and deploy accepted external assets without changing completed DDR identities."""
import gzip
import hashlib
import json
import pathlib
import sys
import tarfile
import time
from datetime import datetime,timezone

ROOT=pathlib.Path(__file__).resolve().parents[1]
REMOTE=pathlib.PurePosixPath('/root/autodl-tmp/HRDecoder_Q_20261003')
sys.path.insert(0,str(ROOT))
from ops.remote import connect

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def write(path,data):
    p=pathlib.Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(data,indent=2),encoding='utf-8');q.replace(p)

def progress(stage,**kw):
    write(ROOT/'reports/qmix_deployment_progress.json',{'stage':stage,'updated_at':datetime.now(timezone.utc).isoformat(),**kw})

def main():
    manifest=json.loads((ROOT/'reports/external_acceptance/manifest.local.accepted.json').read_text())
    assert manifest['status']=='READY' and manifest['dedup_status']=='PASS' and manifest['visual_review_count']>=20
    files={}
    def remap(path,expected=None):
        p=pathlib.Path(path).resolve()
        if p.is_relative_to(ROOT):
            rel=p.relative_to(ROOT).as_posix()
        elif p.is_relative_to(pathlib.Path('D:/dataset/best_segmentation').resolve()):
            rel='assets/external/IDRiD_user_images/'+p.name
        else:raise ValueError('Unapproved source asset path')
        actual=sha(p)
        if expected and actual!=expected:raise ValueError('Accepted source asset changed: '+p.name)
        files[rel]={'local':str(p),'sha256':actual,'bytes':p.stat().st_size}
        return str(REMOTE/rel)
    progress('FREEZING_ASSET_IDENTITIES')
    server=json.loads(json.dumps(manifest))
    for r in server['samples']:
        if r['excluded_duplicate']:continue
        r['image']=remap(r['image'],r['image_sha256'])
        for entry in r['classes'].values():entry['mask']=remap(entry['mask'],entry['sha256'])
        if r['source']=='MAPLES':
            for key,p in list(r['uncertain'].items()):r['uncertain'][key]=remap(p,r['uncertain_sha256'][key])
    server_path=ROOT/'reports/external_acceptance/manifest.server.accepted.json'
    write(server_path,server)
    runtime=json.loads((ROOT/'configs/runtime.server.json').read_text())
    runtime.update(external_manifest=str(REMOTE/'reports/external_acceptance/manifest.server.accepted.json'),stage='Q_MIX_ACCEPTED_FIXED_PAIRED_PROTOCOL')
    runtime_path=ROOT/'configs/runtime.qmix.server.json';write(runtime_path,runtime)
    lock={
        'status':'FROZEN_BEFORE_Q_MIX_TRAINING','created_at':datetime.now(timezone.utc).isoformat(),
        'authority':'Original two user plan files plus direct user IDRiD source override',
        'source_override_sha256':sha(ROOT/'reports/external_source_user_override.json'),
        'external_manifest_server_sha256':sha(server_path),'runtime_qmix_sha256':sha(runtime_path),
        'core_protocol_sha256':sha(ROOT/'configs/protocol.json'),
        'source_sha256':json.loads((ROOT/'reports/source_identity.json').read_text())['sha256'],
        'existing_DDR_final_lock_sha256':sha(ROOT/'reports/final_selection.json'),
        'existing_DDR_final_test_completed':True,
        'test_evidence_boundary':'Exploratory continuation after already completed Q_DDR test; new decisions use val only. Never overwrite or repeat old test selection.',
        'primary_supervision_comparison':'Q_MIX minus Q_DDR single-model mIoU at the same common final update, paired seeds 42/2024',
        'IDRiD_source_quality':'User-authorized stored square JPEGs and polygon-rasterized masks; official native/pixel-GT claim waived; empty unverified classes ignored',
        'MAPLES_source_quality':'Official original MESSIDOR-1 paired by ID to official MAPLES train138 masks with uncertain ignore',
        'required_seeds':[42,2024],'effective_batch':8,'micro_batch':2,'accumulation':4,'dtype':'bfloat16',
        'base_endpoint':16000,'max_endpoint':24000,
        'extension_rule':'Any discovery single 12k->16k gain >=0.3 pp extends both arms and both reviews to 24k, unchanged 24k scheduler',
        'review_rule':'Original same-formula adjacent-late P0+0.5 signal; completed DDR promotion retained; complete common discovery endpoint first',
        'severe_failure_rule':'At 8k all: single<35, fixed fusion<P0-1, single4k->8k gain<0.2; technical faults pause immediately',
        'deployment_candidates':['single','0.75 P0 + 0.25 Q'],'deployment_seed':42,
        'new_selection_rule':'Two-seed mean common-late val once; tie earlier then single; all registered arm candidates, no test-based choices',
        'test_rule':'Both selected seeds>P0 and mean>=P0+0.5; separate immutable continuation lock. Reuse existing counts for an identical already-tested identity; new identity evaluated once.',
        'no_extra_seeds_models_losses_thresholds_TTA':True,
        'DDR_and_MIX_runtime_resources_identical':runtime['resources']==json.loads((ROOT/'configs/runtime.server.json').read_text())['resources'],
        'asset_files':[{k:v for k,v in row.items() if k!='local'}|{'path':rel} for rel,row in sorted(files.items())]
    }
    assert lock['DDR_and_MIX_runtime_resources_identical']
    lock_path=ROOT/'reports/qmix_continuation_protocol.json'
    if lock_path.exists():raise ValueError('Continuation already frozen; do not regenerate identities silently')
    write(lock_path,lock)
    for p in (server_path,runtime_path,lock_path,ROOT/'reports/advance_qmix_campaign.py',ROOT/'reports/start_qmix_remote.py',
              ROOT/'reports/external_source_user_override.json',ROOT/'reports/execution_scope_override.json',ROOT/'reports/original_plan_completion.json'):
        rel=p.relative_to(ROOT).as_posix();files[rel]={'local':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    archive=ROOT/'assets/Q_MIX_accepted_assets.tar.gz'
    progress('PACKAGING_ACCEPTED_TRAIN_ASSETS',files=len(files),asset_bytes=sum(x['bytes'] for x in files.values()))
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for rel,row in sorted(files.items()):tar.add(row['local'],arcname=rel,recursive=False)
    transport={'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
               'files':[{k:v for k,v in row.items() if k!='local'}|{'path':rel} for rel,row in sorted(files.items())]}
    write(ROOT/'reports/qmix_transport_manifest.json',transport)
    progress('UPLOADING',archive_bytes=archive.stat().st_size)
    with connect() as c:
        with c.open_sftp() as s:
            previous=[time.monotonic()]
            def callback(done,total):
                if time.monotonic()-previous[0]>5:
                    progress('UPLOADING',uploaded_bytes=done,archive_bytes=total);previous[0]=time.monotonic()
            s.put(str(archive),str(REMOTE/'assets/Q_MIX_accepted_assets.tar.gz'),callback=callback)
            s.put(str(ROOT/'reports/qmix_transport_manifest.json'),str(REMOTE/'reports/qmix_transport_manifest.json'))
        progress('REMOTE_EXTRACT_AND_SHA_ACCEPTANCE')
        command="/root/miniconda3/envs/hrdecoder_q_20261003/bin/python - <<'PY'\n"+r'''
import hashlib,json,pathlib,tarfile,shutil
from datetime import datetime,timezone
root=pathlib.Path('/root/autodl-tmp/HRDecoder_Q_20261003').resolve()
manifest=json.loads((root/'reports/qmix_transport_manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
archive=root/'assets/Q_MIX_accepted_assets.tar.gz'
assert sha(archive)==manifest['archive_sha256']
required=sum(x['bytes'] for x in manifest['files'])+8*1024**3
assert shutil.disk_usage(root).free>required, 'Insufficient accepted-asset plus late-cache/checkpoint space'
expected={x['path'] for x in manifest['files']}
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers()
 assert {x.name for x in members}==expected
 for m in members:
  p=(root/m.name).resolve()
  assert m.isfile() and p.is_relative_to(root) and '..' not in pathlib.PurePosixPath(m.name).parts
  assert m.name.startswith(('assets/external/','assets/external_derived/','reports/','configs/runtime.qmix.'))
  p.parent.mkdir(parents=True,exist_ok=True)
  with tar.extractfile(m) as src,p.open('wb') as out:
   for b in iter(lambda:src.read(1024**2),b''):out.write(b)
for row in manifest['files']:
 p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
receipt={'status':'ALL_REMOTE_ASSET_SHA_VERIFIED','file_count':len(manifest['files']),
 'archive_sha256':manifest['archive_sha256'],'free_bytes_after_assets':shutil.disk_usage(root).free,
 'completed_at':datetime.now(timezone.utc).isoformat()}
(root/'reports/qmix_remote_asset_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
PY'''
        _,out,err=c.exec_command(command,timeout=300)
        stdout=out.read().decode();stderr=err.read().decode();code=out.channel.recv_exit_status()
        if code:raise RuntimeError(stderr)
        with c.open_sftp() as s:s.get(str(REMOTE/'reports/qmix_remote_asset_receipt.json'),str(ROOT/'reports/qmix_remote_asset_receipt.json'))
        print(stdout,flush=True)
    progress('REMOTE_ASSETS_READY')

if __name__=='__main__':main()
