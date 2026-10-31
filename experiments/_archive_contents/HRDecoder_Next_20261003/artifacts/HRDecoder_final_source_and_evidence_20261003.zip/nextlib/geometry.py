"""One coordinate ledger for sampling, supervision, alignment and inference."""
from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import LOCAL_MEAN, LOCAL_STD, GLOBAL_MEAN, GLOBAL_STD


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
        return self.center + self.halo * 2


def tiles(h, w, center=768, halo=128):
    # Nonoverlapping centers: every real pixel is returned exactly once.
    for y in range(0, h, center):
        for x in range(0, w, center):
            yield Tile(y, x, min(center, h-y), min(center, w-x), center, halo)


def augment_full(arr, row):
    out = np.rot90(arr, int(row['rotation']), axes=(0, 1))
    if row['flip_h']:
        out = out[:, ::-1]
    if row['flip_v']:
        out = out[::-1]
    return np.ascontiguousarray(out)


def normalized(rgb, mean, std):
    return ((rgb.astype(np.float32) - np.asarray(mean, np.float32)) /
            np.asarray(std, np.float32)).transpose(2, 0, 1).copy()


def patch_tensor(rgb, tile, device='cpu'):
    # Identical train/eval padding: zeros after local normalization.
    patch = np.zeros((3, tile.window, tile.window), np.float32)
    y0, x0 = tile.y-tile.halo, tile.x-tile.halo
    ya, xa = max(y0, 0), max(x0, 0)
    yb, xb = min(y0+tile.window, rgb.shape[0]), min(x0+tile.window, rgb.shape[1])
    patch[:, ya-y0:yb-y0, xa-x0:xb-x0] = normalized(rgb[ya:yb, xa:xb], LOCAL_MEAN, LOCAL_STD)
    return torch.from_numpy(patch).unsqueeze(0).to(device)


def supervision(label, tile, device='cpu'):
    target = np.zeros((4, tile.center, tile.center), np.uint8)
    valid = np.zeros_like(target)
    lab = label[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width]
    for c in range(4):
        target[c, :tile.height, :tile.width] = lab == c+1
        valid[c, :tile.height, :tile.width] = lab != 255
    return (torch.from_numpy(target).unsqueeze(0).to(device),
            torch.from_numpy(valid).unsqueeze(0).to(device))


def global_view(rgb, long_side=1024, device='cpu'):
    h, w = rgb.shape[:2]
    scale = long_side / max(h, w)
    rh, rw = max(1, round(h*scale)), max(1, round(w*scale))
    view = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    ph, pw = math.ceil(rh/32)*32, math.ceil(rw/32)*32
    padded = np.zeros((3, ph, pw), np.float32)
    padded[:, :rh, :rw] = normalized(view, GLOBAL_MEAN, GLOBAL_STD)
    ledger = dict(native_hw=[h, w], resized_hw=[rh, rw], padded_hw=[ph, pw],
                  scale_y=rh/h, scale_x=rw/w, long_side=long_side,
                  resize='opencv_linear', pad='normalized_zero_bottom_right',
                  align_corners=False)
    return torch.from_numpy(padded).unsqueeze(0).to(device), ledger


def align_global(feature, ledger, tile, stride=4):
    """Pixel centers in the native window -> resized padded feature lattice."""
    size = tile.window // stride
    dtype, device = feature.dtype, feature.device
    y = tile.y-tile.halo + (torch.arange(size, device=device).float()+.5)*stride
    x = tile.x-tile.halo + (torch.arange(size, device=device).float()+.5)*stride
    ph, pw = ledger['padded_hw']
    gy = 2*y*ledger['scale_y']/ph - 1
    gx = 2*x*ledger['scale_x']/pw - 1
    yy, xx = torch.meshgrid(gy, gx, indexing='ij')
    grid = torch.stack((xx, yy), -1).unsqueeze(0)
    # Alignment is FP32 even if a cache later stores exact fp32 features.
    aligned = F.grid_sample(feature.float(), grid, mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    h, w = ledger['native_hw']
    mask = ((y >= 0) & (y < h))[:, None] & ((x >= 0) & (x < w))[None, :]
    return (aligned * mask[None, None]).to(dtype)


def center_logits(logits, tile):
    q = tile.halo
    return logits[..., q:q+tile.center, q:q+tile.center]


def canvas(rgb, label=None, geometry='native'):
    if geometry == 'native':
        return rgb, label
    if geometry == 'square2048':
        hw = (2048, 2048)
    elif geometry == 'long2048':
        h, w = rgb.shape[:2]
        r = 2048/max(h, w)
        hw = (round(w*r), round(h*r))
    else:
        raise ValueError(f'Unknown frozen canvas {geometry}')
    im = cv2.resize(rgb, hw, interpolation=cv2.INTER_LINEAR)
    lab = None if label is None else cv2.resize(label, hw, interpolation=cv2.INTER_NEAREST)
    return im, lab
