"""List archived checkpoint candidates without extracting or modifying archives."""
import csv
import json
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent


def main():
    rows = list(csv.DictReader((EXP/'paper_writing_20261023/evidence/archive_inventory.csv').open(encoding='utf-8-sig')))
    results = []
    for row in rows:
        p = EXP/row['path']
        entry = {'path': str(p), 'checkpoint_members': []}
        try:
            if zipfile.is_zipfile(p):
                with zipfile.ZipFile(p) as archive:
                    members = [(x.filename, x.file_size) for x in archive.infolist() if not x.is_dir()]
            else:
                with tarfile.open(p, 'r:*') as archive:
                    members = [(x.name,x.size) for x in archive if x.isfile()]
            entry['members'] = len(members)
            entry['checkpoint_members'] = [{'name':n,'bytes':s} for n,s in members if n.lower().endswith(('.pth','.pt','.ckpt','.safetensors'))]
            entry['status'] = 'HEADERS_LISTED_NO_EXTRACTION'
        except (tarfile.TarError, zipfile.BadZipFile, OSError) as error:
            entry.update(status='NOT_A_SUPPORTED_CONTAINER', error=str(error))
        results.append(entry)
        print(json.dumps({'archive': row['path'], 'status': entry['status'], 'checkpoints':len(entry['checkpoint_members'])},ensure_ascii=False),flush=True)
    out = ROOT/'evidence/archive_checkpoint_audit.json'
    out.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__ == '__main__':
    main()
