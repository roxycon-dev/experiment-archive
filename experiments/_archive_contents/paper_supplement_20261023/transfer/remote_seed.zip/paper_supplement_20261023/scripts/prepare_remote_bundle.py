"""Prepare source/data bundle and an exact EMA-only C export; no credentials included."""
import hashlib
import json
from pathlib import Path
import zipfile
import torch

ROOT=Path(__file__).resolve().parents[1]; EXP=ROOT.parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    out=ROOT/'transfer';out.mkdir(exist_ok=True)
    source=EXP/'HRDecoder_FinalDecision_20261005/execution/server_results/runs/C_seed42/checkpoint_u20000.pth'
    compact=out/'C20_ema_inference.pth'
    torch.set_num_threads(2)
    c=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    if not compact.exists():
        torch.save({'ema':c['ema'],'source_sha256':sha(source),'source_completed':c['completed'],
                    'source_state_key':'ema','source_ema_completed':20000},compact)
    check=torch.load(compact,map_location='cpu',mmap=True,weights_only=False)
    assert set(c['ema'])==set(check['ema']) and all(torch.equal(c['ema'][k],check['ema'][k]) for k in c['ema'])
    (out/'C20_export_receipt.json').write_text(json.dumps({'source_sha256':sha(source),'compact_sha256':sha(compact),
        'EMA_tensors_bitwise_equal':True,'tensor_count':len(c['ema']),'completed':20000},indent=2),encoding='utf-8')
    manifest=json.loads((ROOT/'evidence/external_holdout_manifest.json').read_text(encoding='utf-8'))
    roots=[EXP/'HRDecoder_Ready_20261002/legacy',EXP/'HRDecoder_Ready_20261002/vendor/champion_repo',
           EXP/'HRDecoder_Ready_20261002/trainer',EXP/'HRDecoder_FinalDecision_20261005/execution',
           EXP/'HRDecoder_Q_20261003/vendor/dinov3',EXP/'HRDecoder_ArchitectureDesign_20261008/training_execution/python_deps']
    files={p:p.relative_to(EXP).as_posix() for folder in roots for p in folder.rglob('*.py')}
    p=EXP/'HRDecoder_Ready_20261002/configs/champion_server.json';files[p]=p.relative_to(EXP).as_posix()
    for r in manifest['samples']:
        files[Path(r['image_path'])]='data/idrid/images/testing/'+r['sid']+'.jpg'
        files[Path(r['polygon_path'])]='data/idrid/labels/testing/'+r['sid']+'.txt'
        for m in r['masks'].values():
            p=Path(m['path']);files[p]=p.relative_to(EXP).as_posix()
    image=Path('D:/HRDecoder/data/DDR_split_613/images/val/007-1774-100.jpg')
    files[image]='data/canary/007-1774-100.jpg'
    for folder in [ROOT/'scripts',ROOT/'plan',ROOT/'evidence']:
        for p in folder.glob('*'):
            if p.is_file() and p.suffix in ['.py','.md','.json']:
                files[p]=p.relative_to(EXP).as_posix()
    p=ROOT/'results/ddr_test_local_v1/manifest.json'; files[p]='paper_supplement_20261023/evidence/source_DDR_manifest.json'
    p=out/'C20_export_receipt.json';files[p]='paper_supplement_20261023/evidence/C20_export_receipt.json'
    bundle=out/'remote_seed.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p,name in sorted(files.items()):z.write(p,name)
    print(json.dumps({'bundle':str(bundle),'bytes':bundle.stat().st_size,'files':len(files),
                      'compact_C_bytes':compact.stat().st_size}),flush=True)


if __name__=='__main__':main()
