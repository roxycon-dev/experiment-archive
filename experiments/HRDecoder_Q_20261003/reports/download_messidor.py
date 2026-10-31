"""Fetch user-authorized MESSIDOR archives and retain only MAPLES split images."""
import concurrent.futures
import hashlib
import http.cookiejar
import json
import pathlib
import threading
import time
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / 'assets/messidor_download'
DEST = ROOT / 'assets/external/MESSIDOR_MAPLES_originals'
MAPLES = ROOT.parent / 'HRDecoder_Ready_20261002/assets/MAPLES-DR.zip'
LOCK = threading.Lock()
STATE = {'stage': 'DOWNLOADING', 'archives': {}, 'started_at': datetime.now(timezone.utc).isoformat()}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def update(name=None, **values):
    with LOCK:
        target = STATE if name is None else STATE['archives'].setdefault(name, {})
        target.update(values)
        STATE['updated_at'] = datetime.now(timezone.utc).isoformat()
        p = ROOT / 'reports/messidor_download_progress.json'
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(STATE, indent=2), encoding='utf-8')
        tmp.replace(p)

def opener():
    jar = http.cookiejar.MozillaCookieJar(str(WORK / 'cookies.private'))
    jar.load(ignore_discard=True, ignore_expires=False)
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

def fetch(entry, wanted):
    label = entry['name'].split()[0]
    archive = WORK / (label + '.zip')
    part = archive.with_suffix('.zip.part')
    update(label, stage='DOWNLOADING', downloaded_bytes=0)
    if not archive.exists():
        # A bounded single retry handles an interrupted network transfer; no credentials in logs.
        for attempt in range(2):
            try:
                request = urllib.request.Request(entry['url'], headers={'User-Agent': 'Mozilla/5.0'})
                with opener().open(request, timeout=90) as response, part.open('wb') as out:
                    if response.status != 200:
                        raise RuntimeError('unexpected HTTP status ' + str(response.status))
                    declared = int(response.headers.get('Content-Length', '0'))
                    update(label, expected_bytes=declared, attempt=attempt + 1)
                    size, previous = 0, time.monotonic()
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
                        size += len(block)
                        if time.monotonic() - previous >= 5:
                            update(label, downloaded_bytes=size)
                            previous = time.monotonic()
                if declared and size != declared:
                    raise RuntimeError('download length mismatch')
                with zipfile.ZipFile(part) as z:
                    z.infolist()  # Verify the complete central directory before accepting the archive.
                part.replace(archive)
                break
            except Exception as error:
                update(label, stage='RETRY_PENDING' if attempt == 0 else 'FAILED', error_type=type(error).__name__)
                if attempt == 1:
                    raise RuntimeError(label + ': ' + type(error).__name__) from None
    archive_sha = sha(archive)
    rows = []
    update(label, stage='EXTRACTING', downloaded_bytes=archive.stat().st_size, archive_sha256=archive_sha)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            filename = pathlib.PurePosixPath(member.filename).name
            case = pathlib.Path(filename).stem
            if case not in wanted or pathlib.Path(filename).suffix.lower() not in ('.tif', '.tiff', '.jpg', '.jpeg', '.png'):
                continue
            dest = DEST / filename
            if dest.exists():
                raise RuntimeError('duplicate MAPLES image ID: ' + case)
            temp = dest.with_suffix(dest.suffix + '.part')
            # ZipExtFile checks the member CRC while reading to EOF.
            with z.open(member) as src, temp.open('wb') as out:
                for block in iter(lambda: src.read(1024 * 1024), b''):
                    out.write(block)
            temp.replace(dest)
            from PIL import Image
            with Image.open(dest) as im:
                im.load()
                dimensions = list(im.size)
            if dimensions != wanted[case]['dimensions']:
                raise RuntimeError('native image/mask shape mismatch: ' + case)
            rows.append({'id': case, 'split': wanted[case]['split'], 'image': str(dest),
                         'sha256': sha(dest), 'bytes': dest.stat().st_size, 'dimensions': dimensions,
                         'archive': label, 'archive_sha256': archive_sha,
                         'zip_member': member.filename, 'zip_crc32': member.CRC})
    update(label, stage='COMPLETE', matched_images=len(rows))
    print(label + ': complete; matched ' + str(len(rows)), flush=True)
    return rows

def main():
    WORK.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    wanted = {}
    with zipfile.ZipFile(MAPLES) as z:
        for name in z.namelist():
            parts = name.split('/')
            if len(parts) == 3 and parts[0] in ('train', 'test') and parts[1] == 'Exudates' and name.endswith('.png'):
                with z.open(name) as f, Image.open(f) as im:
                    wanted[pathlib.Path(name).stem] = {'split': parts[0], 'dimensions': list(im.size)}
    assert sum(x['split'] == 'train' for x in wanted.values()) == 138
    assert sum(x['split'] == 'test' for x in wanted.values()) == 60
    entries = [x for x in json.loads((WORK / 'links.private').read_text(encoding='utf-8')) if 'images (zip)' in x['name']]
    assert len(entries) == 12
    update(required_images=len(wanted), training_images=138, quarantined_test_images=60,
           provenance='User-authorized ADCIS MESSIDOR-1 download; actual access URLs and cookies are private')
    rows, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, entry, wanted): entry['name'] for entry in entries}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as error:
                errors.append(str(error))
    ids = [x['id'] for x in rows]
    missing = sorted(set(wanted) - set(ids))
    receipt = {'stage': 'COMPLETE' if not errors and not missing and len(ids) == len(set(ids)) else 'FAILED',
               'source_host': 'www.adcis.net', 'maples_annotation_archive_sha256': sha(MAPLES),
               'images': sorted(rows, key=lambda x: x['id']), 'missing_ids': missing, 'errors': errors,
               'test_images_quarantined_from_training': True, 'completed_at': datetime.now(timezone.utc).isoformat()}
    (ROOT / 'reports/messidor_download_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    update(stage=receipt['stage'], extracted_images=len(rows), missing_count=len(missing), errors=errors)
    if receipt['stage'] != 'COMPLETE':
        raise RuntimeError('MESSIDOR acquisition incomplete; inspect private progress and sanitized receipt')
    print('COMPLETE: 138 training + 60 quarantined test originals; CRC, SHA and native shape verified.', flush=True)

if __name__ == '__main__':
    main()
