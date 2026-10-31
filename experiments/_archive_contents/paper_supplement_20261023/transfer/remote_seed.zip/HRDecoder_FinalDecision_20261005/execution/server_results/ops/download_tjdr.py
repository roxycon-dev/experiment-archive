"""Recover the author-published official train448 only. No test labels or images."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import sys
import threading
import time
import gdown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qlib.common import read_json, write_json, utc_now


def main():
    inventory = read_json(ROOT / 'reports/TJDR_download_inventory.json')['files']
    assert len(inventory) == 896
    expected = ROOT / 'inputs/TJDR/train'
    state = {'stage': 'DOWNLOADING', 'started_at': utc_now(), 'expected_files': 896,
             'completed_files': [], 'failures': [], 'official_test_downloaded': False,
             'source': 'Author README GoogleDrive official train folder',
             'usage_terms_acceptance': 'PENDING_DATA_INSPECTION', 'workers': 6}
    receipt = ROOT / 'reports/TJDR_download_receipt.json'
    lock = threading.Lock()

    def recover(item):
        path = Path(item['local_path']).resolve()
        assert path.is_relative_to(expected.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            for attempt in range(3):
                try:
                    result = gdown.download(id=item['id'], output=str(path), quiet=True,
                                            use_cookies=False, resume=True)
                    if result is None:
                        raise OSError('Official download returned no file')
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2**(attempt + 1))
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(block)
        result = {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                  'sha256': h.hexdigest(), 'official_public_file_id': item['id']}
        with lock:
            state['completed_files'].append(result)
            state['updated_at'] = utc_now()
            write_json(receipt, state)
            if len(state['completed_files']) % 20 == 0:
                print('TJDR', len(state['completed_files']), '/', len(inventory), flush=True)
        return result

    # Pair images/masks early so visual acceptance can overlap inbound transfers.
    ordered = sorted(inventory, key=lambda x: (Path(x['local_path']).name,
                                             Path(x['local_path']).parent.name))
    write_json(receipt, state)
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = {pool.submit(recover, item): item for item in ordered}
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as error:
                with lock:
                    state['failures'].append({'path': tasks[future]['path'], 'reason': str(error)})
                    write_json(receipt, state)
    state.update(stage='DOWNLOAD_COMPLETE' if not state['failures'] else 'DOWNLOAD_INCOMPLETE',
                 completed_at=utc_now())
    write_json(receipt, state)
    print(state['stage'], len(state['completed_files']), flush=True)
    if state['failures']:
        raise RuntimeError('Some author-provided files could not be recovered; inspect receipt')


if __name__ == '__main__':
    main()
