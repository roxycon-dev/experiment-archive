from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import CLASSES
from .geometry import Tile, rgb_window, supervision, transform, transform_points


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


class DDR:
    source = 'DDR'

    def __init__(self, root, split, manifest=None, test_authorized=False):
        if split == 'test' and not test_authorized:
            raise ValueError('Test is sealed until final selection and two-seed gate')
        self.root, self.split = Path(root), split
        self.images = {p.stem: p for p in (self.root/'images'/split).glob('*.jpg')}
        self.ids = sorted(self.images)
        if len(self.ids) != {'train': 454, 'val': 76, 'test': 227}[split]:
            raise ValueError('Fixed DDR split count mismatch')
        if manifest and self.ids != [r['sid'] for r in manifest['samples']]:
            raise ValueError('Fixed DDR identities differ')
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = 512*1024**2

    def load(self, sid):
        if sid in self.cache:
            self.cache.move_to_end(sid)
            return self.cache[sid]
        rgb = read_rgb(self.images[sid])
        with Image.open(self.root/'labels'/self.split/(sid+'.png')) as im:
            label = np.asarray(im)
        if label.shape != rgb.shape[:2] or not np.isin(label, [0, 1, 2, 3, 4, 255]).all():
            raise ValueError('DDR label/coordinate mismatch: '+sid)
        target = np.stack([label == c for c in range(1, 5)], axis=-1).astype(np.uint8)
        valid = np.broadcast_to((label != 255)[..., None], target.shape).astype(np.uint8)
        value = rgb, target, valid
        size = sum(x.nbytes for x in value)
        while self.cache and self.cache_bytes+size > self.cache_limit:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= sum(x.nbytes for x in old)
        if size <= self.cache_limit:
            self.cache[sid] = value
            self.cache_bytes += size
        return value


class External:
    def __init__(self, source, accepted_manifest):
        if accepted_manifest.get('status') != 'READY' or source not in ('IDRiD', 'MAPLES'):
            raise ValueError('External supervision has not passed acceptance')
        if accepted_manifest.get('dedup_status') != 'PASS' or accepted_manifest.get('visual_review_count', 0) < 20:
            raise ValueError('External dedup/overlay acceptance incomplete')
        self.source = source
        self.rows = {r['sid']: r for r in accepted_manifest['samples'] if r['source'] == source and r['split'] == 'train' and not r['excluded_duplicate']}
        self.ids = sorted(self.rows)
        if not self.ids:
            raise ValueError('No accepted external training images')

    def load(self, sid):
        row = self.rows[sid]
        rgb = read_rgb(row['image'])
        g = np.zeros((*rgb.shape[:2], 4), np.uint8)
        v = np.zeros_like(g)
        for c, name in enumerate(CLASSES):
            entry = row['classes'][name]
            status = entry['review_status']
            if status not in ('pixel_mask', 'confirmed_negative', 'unknown'):
                raise ValueError('Unknown supervision completeness status')
            if status == 'pixel_mask':
                with Image.open(entry['mask']) as im:
                    a = np.asarray(im)
                if a.shape != rgb.shape[:2]:
                    raise ValueError('Official mask does not align to original image')
                g[:, :, c] = a > 0
            if status != 'unknown':
                v[:, :, c] = 1
        if self.source == 'MAPLES':
            for key, channels in (('RedUncertains', (1, 3)), ('BrightUncertains', (0, 2))):
                path = row['uncertain'][key]
                if path is None and not row['uncertain_confirmed_absent'][key]:
                    raise ValueError('Missing uncertain metadata is not confirmed absence')
                if path is not None:
                    with Image.open(path) as im:
                        uncertain = np.asarray(im) > 0
                    if uncertain.shape != rgb.shape[:2]:
                        raise ValueError('Uncertain mask geometry differs')
                    for c in channels:
                        v[:, :, c][uncertain] = 0
        return rgb, g, v


def catalog(dataset):
    """Class-positive images and components, from training GT only."""
    if isinstance(dataset, DDR) and dataset.split != 'train':
        raise ValueError('Sampling never reads val/test GT')
    result = {}
    positive = [[] for _ in range(4)]
    for sid in dataset.ids:
        rgb, target, valid = dataset.load(sid)
        components = []
        for c in range(4):
            mask = np.ascontiguousarray(target[:, :, c] & valid[:, :, c])
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            entries = []
            for k in range(1, n):
                x, y, w, h, area = map(int, stats[k])
                py, px = np.nonzero(labels[y:y+h, x:x+w] == k)
                entries.append({'bbox': [y, x, h, w], 'points': np.stack((py+y, px+x), axis=1).astype(np.int32)})
            components.append(entries)
            if entries:
                positive[c].append(sid)
        result[sid] = {'hw': list(rgb.shape[:2]), 'components': components}
    if any(not p for p in positive):
        raise ValueError('A required source/class has no legal foreground sample')
    return result, positive


def draw_row(dataset, metadata, positives, seed, update, slot, foreground_class=None):
    source_index = {'DDR': 0, 'IDRiD': 1, 'MAPLES': 2}[dataset.source]
    # Stable random streams let both arms reuse their corresponding DDR slots.
    rng = np.random.default_rng(np.random.SeedSequence([seed, update, slot, source_index]))
    choices = dataset.ids if foreground_class is None else positives[foreground_class]
    sid = choices[int(rng.integers(len(choices)))]
    meta = metadata[sid]
    row = {'sid': sid, 'source': dataset.source, 'update': update, 'slot': slot,
           'rotation': int(rng.integers(4)), 'flip_h': bool(rng.integers(2)), 'flip_v': bool(rng.integers(2)),
           'foreground_class': foreground_class, 'brightness': 1., 'contrast': 1.}
    h, w = meta['hw']
    if foreground_class is not None:
        entries = meta['components'][foreground_class]
        comp = entries[int(rng.integers(len(entries)))]
        y, x, bh, bw = comp['bbox']
        corners = np.array([[y, x], [y+bh-1, x+bw-1]])
        corners, (h, w) = transform_points(corners, meta['hw'], row)
        lo, hi = corners.min(0), corners.max(0)
        if np.all(hi-lo+1 <= 768):
            ylo, yhi = max(0, int(hi[0])-767), min(int(lo[0]), max(0, h-768))
            xlo, xhi = max(0, int(hi[1])-767), min(int(lo[1]), max(0, w-768))
        else:
            anchor = comp['points'][int(rng.integers(len(comp['points'])))]
            anchor, _ = transform_points(anchor[None], meta['hw'], row)
            ay, ax = map(int, anchor[0])
            ylo, yhi = max(0, ay-767), min(ay, max(0, h-768))
            xlo, xhi = max(0, ax-767), min(ax, max(0, w-768))
        y = int(rng.integers(ylo, yhi+1))
        x = int(rng.integers(xlo, xhi+1))
    else:
        if row['rotation'] % 2:
            h, w = w, h
        y, x = int(rng.integers(max(0, h-768)+1)), int(rng.integers(max(0, w-768)+1))
    row.update(y=y, x=x, hw=[h, w])
    if rng.random() < .5:
        row.update(brightness=float(rng.uniform(.9, 1.1)), contrast=float(rng.uniform(.9, 1.1)))
    return row


def make_stream(datasets, catalogs, seed, arm, updates=24000):
    if arm not in ('Q_DDR', 'Q_MIX'):
        raise ValueError('Unregistered arm')
    if arm == 'Q_MIX' and set(datasets) != {'DDR', 'IDRiD', 'MAPLES'}:
        raise ValueError('Q_MIX requires both complete accepted external sources')
    rows = []
    for update in range(1, updates+1):
        mix = arm == 'Q_MIX' and update <= 12000
        sources = ['DDR', 'DDR', 'IDRiD', 'MAPLES'] if mix else ['DDR']*4
        for slot in range(8):
            c = None if slot < 4 else slot-4
            # Uniform external slots fixed; foreground source-class relationship
            # rotates across EX/HE/SE/MA while all four occur each update.
            source = sources[slot] if slot < 4 else sources[(c-(update-1)%4) % 4]
            meta, positive = catalogs[source]
            rows.append(draw_row(datasets[source], meta, positive, seed, update, slot, c))
    return rows


class WindowDataset:
    def __init__(self, datasets, rows, start=0, end=None):
        self.datasets, self.rows, self.start = datasets, rows, start
        self.end = len(rows) if end is None else end

    def __len__(self):
        return self.end-self.start

    def __getitem__(self, index):
        import torch
        row = self.rows[self.start+index]
        rgb, g, v = self.datasets[row['source']].load(row['sid'])
        rgb, g, v = transform(rgb, row), transform(g, row), transform(v, row)
        if list(rgb.shape[:2]) != row['hw']:
            raise ValueError('Replay coordinate drift')
        h, w = row['hw']
        tile = Tile(row['y'], row['x'], min(768, h-row['y']), min(768, w-row['x']))
        target, valid = supervision(g, v, tile)
        if row['foreground_class'] is not None:
            c = row['foreground_class']
            if not (target[c]*valid[c]).any():
                raise ValueError('Foreground moved outside supervised center')
        return tuple(torch.from_numpy(x) for x in (rgb_window(rgb, tile, row['brightness'], row['contrast']), target, valid))
