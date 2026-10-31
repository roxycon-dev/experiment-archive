"""Bounded, read-only RGB/index-mask cache shared by Linux fork workers.

Cache four bytes per original pixel instead of materializing eight redundant
target/valid planes. Exact window tensors are checked against the inherited path.
"""
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from qlib.common import read_json
from qlib.data import DDR, read_rgb
from qlib.geometry import Tile, rgb_window, transform, supervision


class TJDR:
    source = 'TJDR'

    def __init__(self, root, manifest):
        self.root = Path(root)
        assert manifest['status'] == 'READY'
        assert manifest['mapping_verified'] and manifest['dedup_status'] == 'PASS'
        assert manifest['visual_review_count'] >= 20
        self.rows = {row['sid']: row for row in manifest['samples'] if not row['excluded_duplicate']}
        self.ids = sorted(self.rows)
        self.images = {sid: self.root / 'image' / (sid + '.png') for sid in self.ids}
        self.labels = {sid: self.root / 'annotation' / (sid + '.png') for sid in self.ids}
        assert self.ids

    def packed(self, sid):
        rgb = read_rgb(self.images[sid])
        with Image.open(self.labels[sid]) as im:
            label = np.asarray(im).copy()
        assert label.shape == rgb.shape[:2] and np.isin(label, [0, 1, 2, 3, 4]).all()
        # Official EX1 HE2 MA3 SE4 -> DDR EX1 HE2 SE3 MA4.
        mapped = np.array([0, 1, 2, 4, 3], dtype=np.uint8)[label]
        return rgb, mapped

    def load(self, sid):
        rgb, label = self.packed(sid)
        return rgb, np.stack([label == c for c in range(1, 5)], axis=-1).astype(np.uint8), np.ones((*label.shape, 4), np.uint8)


def packed(dataset, sid):
    if isinstance(dataset, TJDR):
        return dataset.packed(sid)
    assert isinstance(dataset, DDR) and dataset.split == 'train'
    rgb = read_rgb(dataset.images[sid])
    with Image.open(dataset.root / 'labels/train' / (sid + '.png')) as im:
        label = np.asarray(im).copy()
    assert label.shape == rgb.shape[:2] and np.isin(label, [0, 1, 2, 3, 4, 255]).all()
    return rgb, label


def cache_sources(datasets, limit_bytes, progress=None):
    tasks, expected = [], 0
    for source, dataset in datasets.items():
        for sid in dataset.ids:
            with Image.open(dataset.images[sid]) as im:
                w, h = im.size
            expected += 4 * w * h
            tasks.append((source, sid, dataset))
    assert expected <= limit_bytes, 'Explicit shared cache budget exceeded; review before formal training'
    cached = {source: {} for source in datasets}
    started = time.perf_counter()

    def decode(item):
        source, sid, dataset = item
        result = packed(dataset, sid)
        for array in result:
            array.flags.writeable = False
        return source, sid, result

    actual = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n, (source, sid, result) in enumerate(pool.map(decode, tasks), 1):
            cached[source][sid] = result
            actual += sum(array.nbytes for array in result)
            if progress and (n % 50 == 0 or n == len(tasks)):
                progress({'images': n, 'total': len(tasks), 'bytes': actual})
    return cached, {'bytes': actual, 'limit_bytes': limit_bytes, 'expected_bytes': expected,
                    'images': len(tasks), 'seconds': time.perf_counter() - started,
                    'storage': 'RGB uint8 + mapped label uint8; read-only pages shared by fork'}


class Windows:
    def __init__(self, cached, rows, start, end):
        self.cached, self.rows, self.start, self.end = cached, rows, start, end

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        row = self.rows[self.start + index]
        rgb, label = self.cached[row['source']][row['sid']]
        rgb, label = transform(rgb, row), transform(label, row)
        assert list(rgb.shape[:2]) == row['hw']
        h, w = row['hw']
        tile = Tile(row['y'], row['x'], min(768, h - row['y']), min(768, w - row['x']))
        label_center = np.full((768, 768), 255, dtype=np.uint8)
        label_center[:tile.height, :tile.width] = label[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width]
        target = np.stack([label_center == c for c in range(1, 5)]).astype(np.float32)
        valid = np.broadcast_to((label_center != 255)[None], target.shape).astype(np.float32)
        if row['foreground_class'] is not None:
            assert (target[row['foreground_class']] * valid[row['foreground_class']]).any()
        x = rgb_window(rgb, tile, row['brightness'], row['contrast'])
        legal = valid.sum(axis=(1, 2), dtype=np.float32)
        positive = (target * valid).sum(axis=(1, 2), dtype=np.float32)
        return tuple(torch.from_numpy(x) for x in (x, target, valid, legal, positive))


def worker_init(_):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def loader_for(cached, rows, source, completed, endpoint, resources, seed):
    slots = 8 if source == 'DDR' else 4
    workers = resources['workers'] if source == 'DDR' else resources['source_workers']
    assert source == 'DDR' or endpoint <= 20000
    return iter(DataLoader(Windows(cached, rows, completed * slots, endpoint * slots),
        batch_size=2, shuffle=False, num_workers=workers, pin_memory=True,
        generator=torch.Generator().manual_seed(seed + (333 if source == 'DDR' else 777)),
        prefetch_factor=resources['prefetch_factor'], persistent_workers=True,
        multiprocessing_context='fork', worker_init_fn=worker_init))
