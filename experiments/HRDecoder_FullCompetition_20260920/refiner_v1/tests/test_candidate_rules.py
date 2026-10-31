#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU unit tests for the frozen candidate rule.  No GPU, no champion, no data.

Run:  python tests/test_candidate_rules.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refiner import candidate as C      # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print("[PASS] %s" % name)
    except Exception as e:                                  # noqa: BLE001
        FAIL.append((name, repr(e)))
        print("[FAIL] %s: %r" % (name, e))


def t_threshold_is_strict():
    p_ma = np.zeros((8, 8), np.float32)
    p_he = np.zeros((8, 8), np.float32)
    p_ma[1, 1] = 0.10                      # not > 0.10
    p_he[2, 2] = 0.1000001
    s = C.candidate_seeds(p_ma, p_he, 0.10)
    assert s.sum() == 1, s.sum()
    assert s[2, 2] and not s[1, 1]


def t_seed_is_union_of_classes():
    p_ma = np.zeros((4, 4), np.float32); p_ma[0, 0] = 0.9
    p_he = np.zeros((4, 4), np.float32); p_he[3, 3] = 0.9
    s = C.candidate_seeds(p_ma, p_he, 0.2)
    assert s.sum() == 2


def t_grow_radius_is_exact():
    m = np.zeros((301, 301), bool); m[150, 150] = True
    for r in (0, 1, 4, 16, 56):
        g = C.grow_mask(m, r)
        ys, xs = np.nonzero(g)
        assert ys.min() == 150 - r and ys.max() == 150 + r, (r, ys.min(), ys.max())
        assert xs.min() == 150 - r and xs.max() == 150 + r
    # radius 0 must be a no-op that returns the input unchanged
    assert C.grow_mask(m, 0).dtype == bool
    assert np.array_equal(C.grow_mask(m, 0), m)


def t_tiles_cover_the_box():
    for box in [(0, 0, 40, 40), (10, 20, 700, 300), (0, 0, 1200, 900),
                (5, 5, 512, 512), (0, 0, 513, 513)]:
        tiles = C.tile_patch(box, 512, 384)
        x0, y0, x1, y1 = box
        m = C.mask_from_tiles(tiles, (y1 + 10, x1 + 10))
        sub = m[y0:y1, x0:x1]
        assert sub.all(), ("uncovered pixels", box)
        for t in tiles:
            assert t[2] - t[0] <= 512 and t[3] - t[1] <= 512
            assert t[0] >= x0 and t[1] >= y0 and t[2] <= x1 and t[3] <= y1


def t_tiles_are_disjoint_from_other_boxes():
    a = C.tile_patch((0, 0, 200, 200), 512, 384)
    assert a == [(0, 0, 200, 200)]


def t_candidate_tiles_end_to_end():
    p_ma = np.zeros((600, 700), np.float32)
    p_he = np.zeros((600, 700), np.float32)
    p_ma[100:110, 100:110] = 0.9
    p_he[400:410, 400:410] = 0.9
    tiles, boxes = C.candidate_tiles(p_ma, p_he, 0.10, 0)
    assert len(boxes) == 2, boxes
    m = C.mask_from_tiles(tiles, (600, 700))
    assert m[105, 105] and m[405, 405]
    assert not m[300, 300]
    assert m.sum() <= 2 * (10 + 2 * C.MARGIN) ** 2


def t_hann_weights_positive_and_symmetric():
    w = C.hann2d(64, 96)
    assert w.shape == (64, 96)
    assert (w > 0).all()
    assert abs(w[0, :].mean() - w[-1, :].mean()) < 1e-6
    assert w[32, :].mean() > w[0, :].mean()


def t_stitch_single_tile_is_identity_on_that_region():
    tiles = [(10, 20, 40, 60)]
    patch = np.arange(2 * 40 * 30, dtype=np.float32).reshape(2, 40, 30)
    out = C.stitch_tiles(tiles, patch[None], shape=(80, 80))
    assert out.shape == (2, 80, 80)
    assert np.allclose(out[:, 20:60, 10:40], patch, atol=1e-5)
    assert np.allclose(out[:, :20], 0.0)


def t_stitch_overlap_is_weighted_mean():
    tiles = [(0, 0, 40, 40), (20, 0, 60, 40)]        # 20 px overlap in x
    a = np.full((1, 40, 40), 0.2, np.float32)
    b = np.full((1, 40, 40), 0.8, np.float32)
    out = C.stitch_tiles(tiles, np.stack([a, b]), shape=(40, 60))
    # outside the overlap the single covering tile wins exactly
    assert abs(float(out[0, 5, 5]) - 0.2) < 1e-6
    assert abs(float(out[0, 5, 55]) - 0.8) < 1e-6
    # inside the overlap the result is the fixed Hann-weighted mean, which is a
    # convex combination strictly between the two values
    v = float(out[0, 20, 30])
    assert 0.2 < v < 0.8, v
    w = C.hann2d(40, 40)[0]
    exp = (w[30] * 0.2 + w[10] * 0.8) / (w[30] + w[10])
    assert abs(v - exp) < 1e-5, (v, exp)


def t_stitch_normalises_constant_input():
    """Fixed weighted average must reproduce a constant exactly (weights sum to 1)."""
    tiles = [(0, 0, 40, 40), (10, 0, 50, 40), (20, 0, 60, 40)]
    p = np.full((3, 1, 40, 40), 0.37, np.float32)
    out = C.stitch_tiles(tiles, p, shape=(40, 60))
    assert np.allclose(out[:, :, :], 0.37, atol=1e-6), float(out[0, 20, 30])
    assert np.isfinite(out).all()


def t_stitch_rejects_undersized_patch():
    try:
        C.stitch_tiles([(0, 0, 60, 40)], np.ones((1, 1, 40, 40), np.float32),
                       shape=(40, 60))
        raise AssertionError("undersized patch not rejected")
    except C.ContractError:
        pass


def t_stitch_uniform_weights_average():
    tiles = [(0, 0, 4, 4), (2, 0, 6, 4)]
    a = np.full((1, 4, 4), 0.2, np.float32)
    b = np.full((1, 4, 4), 0.8, np.float32)
    out = C.stitch_tiles(tiles, np.stack([a, b]), weight_fn=lambda h, w: np.ones((h, w), np.float32),
                         shape=(4, 6))
    assert abs(float(out[0, 2, 3]) - 0.5) < 1e-6
    assert abs(float(out[0, 2, 0]) - 0.2) < 1e-6


def t_stitch_is_nan_free_when_region_untouched():
    tiles = [(0, 0, 8, 8)]
    out = C.stitch_tiles(tiles, np.ones((1, 1, 8, 8), np.float32), shape=(16, 16))
    assert np.isfinite(out).all()
    assert float(out[0, 15, 15]) == 0.0


def t_forbidden_path_guard():
    for bad in ("/root/data/test", "/root/test/x.npz", "C:\\x\\TEST\\y",
                "/a/test.json", "/tmp/test"):
        try:
            C.check_no_forbidden_paths(bad)
            raise AssertionError("guard did not reject %r" % bad)
        except C.ContractError:
            pass
    for ok in ("/root/data/train", "/root/data/val", "/root/latest/x.npz",
               "/root/contest/y.json", "/root/attest/z"):
        C.check_no_forbidden_paths(ok)


def t_mismatched_prob_shapes_rejected():
    try:
        C.candidate_seeds(np.zeros((4, 4), np.float32), np.zeros((4, 5), np.float32), 0.1)
        raise AssertionError("shape mismatch not rejected")
    except C.ContractError:
        pass


def main():
    check("threshold_is_strict_gt", t_threshold_is_strict)
    check("seed_is_union_of_classes", t_seed_is_union_of_classes)
    check("grow_radius_is_exact", t_grow_radius_is_exact)
    check("tiles_cover_the_box", t_tiles_cover_the_box)
    check("small_box_single_tile", t_tiles_are_disjoint_from_other_boxes)
    check("candidate_tiles_end_to_end", t_candidate_tiles_end_to_end)
    check("hann_weights_positive_and_symmetric", t_hann_weights_positive_and_symmetric)
    check("stitch_single_tile_identity", t_stitch_single_tile_is_identity_on_that_region)
    check("stitch_overlap_weighted_mean", t_stitch_overlap_is_weighted_mean)
    check("stitch_normalises_constant", t_stitch_normalises_constant_input)
    check("stitch_uniform_weights_average", t_stitch_uniform_weights_average)
    check("stitch_rejects_undersized_patch", t_stitch_rejects_undersized_patch)
    check("stitch_nan_free", t_stitch_is_nan_free_when_region_untouched)
    check("forbidden_path_guard", t_forbidden_path_guard)
    check("prob_shape_mismatch_rejected", t_mismatched_prob_shapes_rejected)
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for n, e in FAIL:
        print("  FAILED %s: %s" % (n, e))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
