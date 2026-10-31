"""Exact paired data stream, independent of model random-number consumption."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image

from .common import read_json, write_json, sha, digest
from .geometry import augment_full, canvas, Tile


class DDR:
    def __init__(self, root, split, expected=None):
        if split not in ('train', 'val'):
            raise ValueError('Only train and val are available to this campaign')
        self.root, self.split = Path(root), split
        self.images = {p.stem: p for p in (self.root/'images'/split).glob('*.jpg')}
        self.stems = sorted(self.images)
        if len(self.stems) != {'train': 454, 'val': 76}[split]:
            raise ValueError(f'Wrong fixed DDR split size: {split}, {len(self.stems)}')
        if expected is not None and set(self.stems) != set(expected):
            raise ValueError('DDR val identities differ from historical 76-image protocol')
        self.cache = OrderedDict()
        self.cache_bytes = 0

    def __len__(self):
        return len(self.stems)

    def load(self, sid):
        if sid in self.cache:
            self.cache.move_to_end(sid)
            return self.cache[sid]
        with Image.open(self.images[sid]) as im:
            rgb = np.array(im.convert('RGB'))
        with Image.open(self.root/'labels'/self.split/(sid+'.png')) as im:
            label = np.array(im)
        if label.ndim != 2 or label.shape != rgb.shape[:2] or label.max() > 4:
            raise ValueError(f'GT coordinate/value contract failed: {sid}')
        size = rgb.nbytes + label.nbytes
        while self.cache and self.cache_bytes+size > 512*1024**2:
            _, (a, b) = self.cache.popitem(last=False)
            self.cache_bytes -= a.nbytes+b.nbytes
        self.cache[sid] = rgb, label
        self.cache_bytes += size
        return rgb, label

    def fingerprint(self):
        rows = []
        for sid in self.stems:
            rgb, label = self.load(sid)
            rows.append(dict(sid=sid, hw=list(label.shape),
                             image_sha256=sha(self.images[sid]),
                             label_sha256=sha(self.root/'labels'/self.split/(sid+'.png'))))
        return {'split': self.split, 'samples': rows, 'sha256': digest(rows)}


def make_stream(dataset, path, seed, geometry, updates=8000, batch=8, center=768, halo=128):
    """Store every chosen center and full-image D4 transform before either arm trains."""
    path = Path(path)
    n = updates*batch
    if path.exists():
        info = read_json(path)
        if (info['seed'], info['geometry'], len(info['rows']), info['center'], info['halo']) != (
                seed, geometry, n, center, halo):
            raise ValueError('Existing stream disagrees with locked pairing')
        return info
    rng = np.random.default_rng(seed)
    rows = []
    # Read each mask once. An image LRU here would re-decode tens of thousands
    # of full images while merely preparing a 64k-row sample manifest.
    prepared = {}
    for sid in dataset.stems:
        rgb,label=dataset.load(sid)
        _,label=canvas(rgb,label,geometry)
        coords={c:np.argwhere(label==c).astype(np.int32) for c in range(1,5) if (label==c).any()}
        prepared[sid]=label.shape,coords
    for draw in range(n):
        sid = dataset.stems[int(rng.integers(len(dataset)))]
        # Cache small coordinate arrays, not one transformed full image per draw.
        (h0, w0), coords = prepared[sid]
        rot = int(rng.integers(4))
        fh, fv = bool(rng.integers(2)), bool(rng.integers(2))
        h, w = (w0, h0) if rot % 2 else (h0, w0)
        anchor = int(rng.choice(list(coords))) if coords and rng.random() < .5 else 0
        if anchor:
            y, x = map(int, coords[anchor][int(rng.integers(len(coords[anchor])))])
            if rot == 1: y, x = w0-1-x, y
            elif rot == 2: y, x = h0-1-y, w0-1-x
            elif rot == 3: y, x = x, h0-1-y
            if fh: x = w-1-x
            if fv: y = h-1-y
            ylo, yhi = max(0, y-center+1), min(y, max(0, h-center))
            xlo, xhi = max(0, x-center+1), min(x, max(0, w-center))
            y, x = int(rng.integers(ylo, yhi+1)), int(rng.integers(xlo, xhi+1))
        else:
            y = int(rng.integers(max(0, h-center)+1))
            x = int(rng.integers(max(0, w-center)+1))
        rows.append(dict(draw=draw, sid=sid, rotation=rot, flip_h=fh, flip_v=fv,
                         y=y, x=x, hw=[h, w], anchor_class=anchor))
    info = dict(version=1, seed=seed, geometry=geometry, center=center, halo=halo,
                effective_batch=batch, updates=updates, rows=rows, rows_sha256=digest(rows))
    write_json(path, info)
    return info


def replay(dataset, row, geometry, center=768, halo=128):
    rgb, label = dataset.load(row['sid'])
    rgb, label = canvas(rgb, label, geometry)
    rgb, label = augment_full(rgb, row), augment_full(label, row)
    if list(label.shape) != row['hw']:
        raise ValueError('Replay shape drift')
    tile = Tile(row['y'], row['x'], min(center, label.shape[0]-row['y']),
                min(center, label.shape[1]-row['x']), center, halo)
    if row['anchor_class'] and not (label[tile.y:tile.y+tile.height,
                                          tile.x:tile.x+tile.width] == row['anchor_class']).any():
        raise ValueError('Foreground anchor is outside the supervised center')
    return rgb, label, tile
