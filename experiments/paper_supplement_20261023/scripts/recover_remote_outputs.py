"""Recover this run's outputs without writing into historical model directories."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'transfer/external_results.zip'
records = []
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None, 'Archive CRC failure'
    for item in z.infolist():
        name = PurePosixPath(item.filename)
        assert not name.is_absolute() and '..' not in name.parts
        if item.is_dir():
            continue
        if item.filename == 'HRDecoder_Ready_20261002/best_pipeline/weights/exact_transport_receipt.json':
            target = ROOT / 'evidence/remote_exact_transport_receipt.json'
        else:
            assert name.parts[0] == ROOT.name
            relative = PurePosixPath(*name.parts[1:])
            allowed = (str(relative).startswith('results/idrid_holdout_v1/')
                       or str(relative) == 'evidence/remote_branch_audit.json'
                       or (len(relative.parts) == 1 and relative.name.startswith('log_') and relative.suffix == '.txt'))
            assert allowed, item.filename
            target = ROOT / str(relative)
        assert target.resolve().is_relative_to(ROOT.resolve())
        data = z.read(item)
        if target.exists():
            assert target.read_bytes() == data, f'Existing output differs: {target}'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        records.append({'archive_entry': item.filename, 'local_path': str(target),
                        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
receipt = {'status': 'COMPLETE', 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
           'archive_bytes': archive.stat().st_size, 'files': records,
           'historical_model_files_overwritten': False}
(ROOT / 'evidence/remote_output_recovery.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({'status': 'COMPLETE', 'recovered_files': len(records), 'archive_bytes': receipt['archive_bytes']}))
