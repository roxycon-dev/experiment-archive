from dataclasses import dataclass

import numpy as np


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Tile:
    y: int
    x: int
    height: int
    width: int
    center: int = 768
    halo: int = 128

    @property
    def window(self):
        return self.center + 2 * self.halo


def tiles(h, w, center=768, halo=128):
    for y in range(0, h, center):
        for x in range(0, w, center):
            yield Tile(y, x, min(center, h-y), min(center, w-x), center, halo)


def transform(a, row):
    a = np.rot90(a, row['rotation'], axes=(0, 1))
    if row['flip_h']:
        a = a[:, ::-1]
    if row['flip_v']:
        a = a[::-1]
    return a


def transform_points(points, hw, row):
    points = np.asarray(points, dtype=np.int64).copy()
    y, x = points[:, 0].copy(), points[:, 1].copy()
    h, w = hw
    k = row['rotation']
    if k == 1:
        y, x = w-1-x, y
    elif k == 2:
        y, x = h-1-y, w-1-x
    elif k == 3:
        y, x = x, h-1-y
    if k % 2:
        h, w = w, h
    if row['flip_h']:
        x = w-1-x
    if row['flip_v']:
        y = h-1-y
    return np.stack((y, x), axis=1), (h, w)


def rgb_window(rgb, tile, brightness=1., contrast=1.):
    # Padding is zero in NORMALIZED coordinates, not black RGB.
    patch = np.zeros((3, tile.window, tile.window), dtype=np.float32)
    y0, x0 = tile.y-tile.halo, tile.x-tile.halo
    ya, xa = max(y0, 0), max(x0, 0)
    yb, xb = min(y0+tile.window, rgb.shape[0]), min(x0+tile.window, rgb.shape[1])
    region = rgb[ya:yb, xa:xb].astype(np.float32) / 255.
    # Fixed RGB contrast around 0.5 makes the full-image transform crop-local
    # without a crop-dependent image mean or unnecessary full-image copies.
    region = np.clip(((region - .5) * contrast + .5) * brightness, 0., 1.)
    region = ((region - MEAN) / STD).transpose(2, 0, 1)
    patch[:, ya-y0:yb-y0, xa-x0:xb-x0] = region
    return patch


def supervision(target, valid, tile):
    g = np.zeros((4, tile.center, tile.center), dtype=np.float32)
    v = np.zeros_like(g)
    y, x, h, w = tile.y, tile.x, tile.height, tile.width
    g[:, :h, :w] = target[y:y+h, x:x+w].transpose(2, 0, 1)
    v[:, :h, :w] = valid[y:y+h, x:x+w].transpose(2, 0, 1)
    if not v.any():
        raise ValueError('All-invalid sample; stop data acceptance')
    return g, v


def center_logits(z, center=768, halo=128):
    if z.shape[-2:] != (center+2*halo, center+2*halo):
        raise ValueError('Interpolate logits to the full window BEFORE center crop')
    return z[..., halo:halo+center, halo:halo+center]
