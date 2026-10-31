import getpass
import sys
from pathlib import Path
from urllib.parse import urlsplit
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qlib.common import sha, utc_now, write_json


root = Path(__file__).resolve().parents[1]
url = getpass.getpass('Authorized official ViT-B/16 URL (not echoed): ')
parts = urlsplit(url)
if parts.hostname != 'dinov3.llamameta.net' or not parts.path.endswith('dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'):
    raise ValueError('Not the authorized official ViT-B/16 LVD asset')
target = root/'assets'/Path(parts.path).name
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix('.download')
try:
    with urllib.request.urlopen(url, timeout=40) as response, temporary.open('wb') as f:
        total = 0
        for chunk in iter(lambda: response.read(4*1024**2), b''):
            f.write(chunk)
            total += len(chunk)
            if total % (32*1024**2) == 0:
                print(f'Downloaded {total//1024**2} MiB', flush=True)
    url = None
    checksum = sha(temporary)
    if not checksum.startswith('73cec8be'):
        raise ValueError('Official filename hash prefix does not match downloaded bytes')
    temporary.replace(target)
    write_json(root/'reports/pretraining_download.json', {'status': 'DOWNLOADED_SHA_VERIFIED', 'model': 'dinov3_vitb16',
        'pretraining': 'LVD1689M', 'file': target.name, 'bytes': target.stat().st_size, 'sha256': checksum,
        'source': 'https://'+parts.hostname+parts.path, 'signed_url_stored': False,
        'license': 'DINOv3 License', 'license_sha256': sha(root/'vendor/dinov3/LICENSE.md'),
        'authorization': 'User supplied official access-granted email and signed link on 2026-10-03',
        'created_at': utc_now()})
    print('Official pretrained SHA verified:', checksum, flush=True)
except Exception as exc:
    print('Download failed:', type(exc).__name__, flush=True)
    sys.exit(1)
