"""Download public assets with resume, integrity checks, and recorded provenance."""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CONVNEXT = 'https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth'

def download(url, dest, expected_size=None, md5=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    if not dest.exists():
        for attempt in range(8):
            offset = part.stat().st_size if part.exists() else 0
            try:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                # Fresh redirect avoids a cached, expired signed download URL.
                sep = '&' if '?' in url else '?'
                with requests.get(url + sep + f'prep={time.time_ns()}', headers=headers,
                                  stream=True, timeout=(20, 60)) as r:
                    r.raise_for_status()
                    if offset and r.status_code != 206:
                        offset = 0
                    with part.open('ab' if offset else 'wb') as f:
                        for chunk in r.iter_content(256 * 1024):
                            f.write(chunk)
                if expected_size and part.stat().st_size != expected_size:
                    raise ValueError('Incomplete download')
                part.replace(dest)
                break
            except (requests.RequestException, ValueError) as e:
                print(f'download retry {attempt + 1}: {type(e).__name__}', flush=True)
        else:
            raise RuntimeError(f'Download failed: {url}')
    content_md5 = hashlib.md5(dest.read_bytes()).hexdigest()
    if md5 and content_md5 != md5:
        raise ValueError(f'MD5 mismatch: {dest}')
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    return {'url': url, 'file': str(dest), 'size': dest.stat().st_size,
            'sha256': sha, 'md5': content_md5}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset', choices=['maples', 'convnext'], required=True)
    a = ap.parse_args()
    if a.asset == 'maples':
        metadata = requests.get('https://api.figshare.com/v2/articles/24328660', timeout=30).json()
        file = next(f for f in metadata['files'] if f['name'] == 'MAPLES-DR.zip')
        result = download(file['download_url'], ROOT/'assets/MAPLES-DR.zip',
                          file['size'], file['computed_md5'])
        with zipfile.ZipFile(result['file']) as z:
            target = ROOT/'assets/maples_labels'
            for member in z.infolist():
                resolved = (target/member.filename).resolve()
                if not resolved.is_relative_to(target.resolve()):
                    raise ValueError('Unsafe zip path')
            z.extractall(target)
        result.update(figshare_version=metadata['version'], license=metadata['license'],
                      images_available=False)
    else:
        result = download(CONVNEXT, ROOT/'assets/convnext_small_1k_224_ema.pth')
    (ROOT/f'reports/{a.asset}_asset.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
