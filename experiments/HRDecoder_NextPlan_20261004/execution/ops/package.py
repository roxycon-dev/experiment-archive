"""Small deployable code/evidence bundle. References existing server weights and data."""
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qlib.common import digest,sha,utc_now,write_json,read_json


def main():
    assert read_json(ROOT/'reports/cpu_acceptance.json')['status']=='PASS'
    streams={p.name:sha(p) for p in (ROOT/'reports').glob('stream_*.jsonl')}
    write_json(ROOT/'reports/stream_identity.json',streams)
    paths=list(ROOT.glob('*.py'))+list((ROOT/'qlib').glob('*.py'))+list((ROOT/'ops').glob('*.py'))
    paths+=list((ROOT/'configs').glob('*.json'))
    paths+=list((ROOT/'historical').rglob('*.json'))+[ROOT/'historical/train_run.py']
    paths+=list(ROOT.glob('*.md'))+list(ROOT.glob('*.ps1'))
    paths+=list((ROOT/'reports').glob('*.json'))+list((ROOT/'reports').glob('*.jsonl'))
    paths=[p for p in paths if p.name not in ('package_identity.json','source_identity.json') and 'runtime.local' not in p.name]
    paths=sorted(set(paths))
    files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    # Server preflight may freeze a shared micro1x8 F runtime; the accepted runtime
    # is hashed separately in gpu_readiness, so runtime.server is excluded here.
    immutable_reports={'stream_identity.json','historical_implementation.json','negative_diff.local.json',
        'negative_diff.server.json','negative_input_checks.json','data_identity.json','pretraining_download.json',
        'pretraining_identity.json','affected_window_counts.json','cpu_acceptance.json','local_initialization_receipt.json',
        'manifest.local.negative_fixed.json','manifest.server.negative_fixed.json','preparation.json'}
    frozen={rel:h for rel,h in files.items() if rel!='configs/runtime.server.json' and
            (not rel.startswith('reports/') or Path(rel).name in immutable_reports or Path(rel).name.startswith(('stream_','affected_windows_seed')))}
    source={'files':frozen,'sha256':digest(frozen),'created_at':utc_now()}
    write_json(ROOT/'reports/source_identity.json',source)
    target=ROOT/'reports/HRDecoder_20261004_prepared.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in paths+[ROOT/'reports/source_identity.json']:
            archive.write(path,path.relative_to(ROOT).as_posix())
    write_json(ROOT/'reports/package_identity.json',{'sha256':sha(target),'bytes':target.stat().st_size,
               'source_sha256':source['sha256'],'existing_weights_not_duplicated':True,'created_at':utc_now()})
    print(json.dumps({'package':str(target),'bytes':target.stat().st_size,'sha256':sha(target)}))


if __name__=='__main__':main()
