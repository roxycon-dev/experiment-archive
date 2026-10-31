#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU unit tests for the MA/HE refiner: model, loss, assembly, guards.

No GPU, no champion checkpoint, no dataset, no network access.

Run:  python tests/test_refiner_units.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refiner import assemble as A          # noqa: E402
from refiner import candidate as C          # noqa: E402
from refiner import guards as G             # noqa: E402
from refiner import haar as H               # noqa: E402
from refiner import losses as L             # noqa: E402
from refiner import tiled_infer as TI       # noqa: E402
from refiner.model import HaarHeteroRefiner  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print("[PASS] %s" % name)
    except Exception as e:                                   # noqa: BLE001
        import traceback
        FAIL.append((name, repr(e)))
        print("[FAIL] %s: %r" % (name, e))
        traceback.print_exc()


def _champ(h=64, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(1, 2, h, w, generator=g) * 0.98 + 0.01)


# ---------------------------------------------------------------------------
# 1. identity
# ---------------------------------------------------------------------------

def t_identity_is_bitwise():
    torch.manual_seed(0)
    m = HaarHeteroRefiner().eval()
    rgb = torch.rand(2, 3, 64, 64)
    p = _champ(64, 64, 3).repeat(2, 1, 1, 1)
    with torch.no_grad():
        r, refined = m.forward_refine(rgb, p)
    assert torch.count_nonzero(r) == 0, "head is not zero at init"
    assert torch.equal(refined, p), "identity is not bitwise (max|d|=%.3e)" % (
        (refined - p).abs().max().item())
    # EX/SE path: the refiner never emits them
    assert refined.shape[1] == 2


def t_identity_holds_after_randomising_other_layers():
    """The identity comes from the ZERO HEAD, not from a lucky whole-model init."""
    torch.manual_seed(1)
    m = HaarHeteroRefiner().eval()
    for name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Conv2d) and name != "head":
            torch.nn.init.normal_(mod.weight, 0, 0.5)
        if isinstance(mod, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(mod.weight, 1.0, 0.3)
            torch.nn.init.normal_(mod.bias, 0.0, 0.3)
    rgb = torch.rand(1, 3, 48, 48)
    p = _champ(48, 48, 7)
    with torch.no_grad():
        _r, refined = m.forward_refine(rgb, p)
    assert torch.equal(refined, p)


def t_residual_is_bounded():
    m = HaarHeteroRefiner(residual_scale=0.5)
    p = torch.full((1, 2, 8, 8), 0.5)
    big = torch.full((1, 2, 8, 8), 1e4)
    small = torch.full((1, 2, 8, 8), -1e4)
    up = m.refine(p, big)
    dn = m.refine(p, small)
    assert torch.allclose(up, torch.full_like(p, 1.0), atol=1e-6)
    assert torch.allclose(dn, torch.full_like(p, 0.0), atol=1e-6)
    assert float((m.refine(p, big) - p).abs().max()) <= 0.5 + 1e-6


def t_gradients_reach_the_head():
    """tanh'(0)=1: the zero-initialised head is NOT a dead branch."""
    torch.manual_seed(0)
    m = HaarHeteroRefiner()
    rgb = torch.rand(2, 3, 64, 64)
    p = _champ(64, 64, 5).repeat(2, 1, 1, 1)
    tgt = torch.zeros(2, 2, 64, 64); tgt[:, :, 20:24, 20:24] = 1.0
    valid = torch.ones(2, 1, 64, 64)
    _r, refined = m.forward_refine(rgb, p)
    loss, _ = L.refiner_losses(refined, tgt, valid)
    loss.backward()
    g = m.head.weight.grad
    assert g is not None and torch.isfinite(g).all(), "head weight grad missing/non-finite"
    assert float(g.abs().sum()) > 0.0, "head weight grad is exactly zero (dead branch)"
    for n, prm in m.named_parameters():
        assert prm.grad is not None and torch.isfinite(prm.grad).all(), "bad grad: %s" % n


# ---------------------------------------------------------------------------
# 2. Haar features
# ---------------------------------------------------------------------------

def t_haar_level_is_invertible():
    x = torch.rand(2, 3, 32, 48)
    a, h, v, d = H.haar_level(x)
    back = H.haar_inverse(a, h, v, d)
    assert back.shape == x.shape, (back.shape, x.shape)
    assert torch.allclose(back, x, atol=1e-5), (back - x).abs().max().item()


def t_haar_invertible_on_odd_sizes():
    x = torch.rand(1, 1, 15, 21)
    a, h, v, d = H.haar_level(x)
    back = H.haar_inverse(a, h, v, d)[:, :, :15, :21]
    assert torch.allclose(back, x, atol=1e-5)


def t_haar_hf_channel_count_and_shape():
    g = torch.rand(2, 1, 64, 96)
    for lv in (1, 2, 3):
        y = H.haar_hf_channels(g, levels=lv)
        assert y.shape == (2, 3 * lv, 64, 96), y.shape
    assert H.NUM_SUB_BANDS == 6


def t_haar_hf_has_no_parameters_or_grad():
    g = torch.rand(1, 1, 32, 32, requires_grad=True)
    y = H.haar_hf_channels(g, 2)
    assert y.requires_grad
    y.sum().backward()
    assert torch.isfinite(g.grad).all()


def t_build_input_channel_count():
    rgb = torch.rand(1, 3, 32, 32)
    p = _champ(32, 32, 1)
    x = HaarHeteroRefiner.build_input(rgb, p, 2)
    assert x.shape == (1, 11, 32, 32), x.shape
    assert torch.allclose(x[:, 0:3], rgb)
    assert torch.allclose(x[:, 3:5], p)


# ---------------------------------------------------------------------------
# 3. spatial alignment / assembly
# ---------------------------------------------------------------------------

def _synth_champ(h=200, w=200, seed=11):
    g = np.random.default_rng(seed)
    ch = g.random((4, h, w), dtype=np.float32) * 0.9 + 0.05
    return ch


def _ma_he(ch):
    """(2,H,W) view in the refiner's class order (MA, HE).

    NOTE the champion/contract class order is EX=0, HE=1, SE=2, MA=3, so MA and
    HE are NOT adjacent -- slicing ch[3:5] silently yields one channel.
    """
    return np.stack([ch[3], ch[1]], axis=0)


def t_assemble_outside_candidate_is_bitwise():
    ch = _synth_champ()
    tiles = [(20, 20, 70, 70), (120, 100, 170, 150)]
    patches = np.stack([_ma_he(ch)[:, t[1]:t[3], t[0]:t[2]] for t in tiles])
    out = A.assemble(ch, tiles, patches)
    rep = A.assert_bitwise_unchanged(out, ch, tiles)
    for c in C.CLASSES:
        assert rep["%s_changed_outside" % c] == 0, (c, rep)
    assert rep["EX_total_changed"] == 0 and rep["SE_total_changed"] == 0


def t_assemble_identity_when_patches_equal_champion():
    """If the refiner returns the champion probability, the assembled map is
    bitwise the champion INSIDE the candidate too."""
    ch = _synth_champ()
    tiles = [(10, 10, 60, 60), (40, 40, 90, 90)]
    patches = np.stack([_ma_he(ch)[:, t[1]:t[3], t[0]:t[2]] for t in tiles])
    out = A.assemble(ch, tiles, patches)
    assert np.array_equal(out, ch), "assembled != champion (max %.3e)" % (
        np.abs(out - ch).max())


def t_assemble_uses_patch_centre_when_context_is_present():
    """A patch extracted with context must be centre-cropped, otherwise the
    result would be spatially shifted."""
    ch = _synth_champ(120, 120)
    tile = (40, 40, 80, 80)
    ctx = 7
    x0, y0, x1, y1 = tile
    big = _ma_he(ch)[:, y0 - ctx:y1 + ctx, x0 - ctx:x1 + ctx].copy()
    out = A.assemble(ch, [tile], big[None])
    assert np.array_equal(out, ch), "context patch mis-aligned (max %.3e)" % (
        np.abs(out - ch).max())


def t_assemble_rejects_undersized_patch():
    ch = _synth_champ(60, 60)
    try:
        A.assemble(ch, [(0, 0, 50, 50)], np.zeros((1, 2, 10, 10), np.float32))
        raise AssertionError("undersized patch not rejected")
    except C.ContractError:
        pass


def t_assemble_handles_empty_tiles():
    ch = _synth_champ(40, 40)
    out = A.assemble(ch, [], np.zeros((0, 2, 1, 1), np.float32))
    assert np.array_equal(out, ch)


def t_assemble_overlap_is_weighted_average():
    ch = _synth_champ(80, 80)
    tiles = [(0, 0, 40, 40), (20, 0, 60, 40)]
    a = np.full((1, 2, 40, 40), 0.2, np.float32)
    b = np.full((1, 2, 40, 40), 0.8, np.float32)
    out = A.assemble(ch, tiles, np.concatenate([a, b], 0))
    v = float(out[3, 20, 30])
    assert 0.2 < v < 0.8, v
    assert abs(float(out[3, 5, 5]) - 0.2) < 1e-6
    assert abs(float(out[3, 5, 55]) - 0.8) < 1e-6
    # untouched channel is still bitwise champion everywhere
    assert np.array_equal(out[0], ch[0])


def t_binarize_threshold_is_fixed():
    p = np.array([[[0.5, 0.5000001], [0.4999999, 0.9]]], np.float32)
    b = A.binarize(p)
    assert b.dtype == bool
    assert not b[0, 0, 0] and b[0, 0, 1] and not b[0, 1, 0] and b[0, 1, 1]
    try:
        A.binarize(p, 0.4)
        raise AssertionError("non-fixed threshold accepted")
    except C.ContractError:
        pass


def t_end_to_end_refiner_assembly_roundtrip():
    """Full path: champion -> tiles -> patches -> assemble, with a zero-init
    model. Output must be bitwise the champion everywhere."""
    torch.manual_seed(0)
    m = HaarHeteroRefiner().eval()
    ch = _synth_champ(160, 160)
    tiles, boxes = C.candidate_tiles(ch[3], ch[1], 0.10, 0)
    assert tiles, "no candidate tiles produced"
    rgb = torch.rand(1, 3, 160, 160)
    patches = []
    with torch.no_grad():
        for t in tiles:
            x0, y0, x1, y1 = t
            sub_rgb = rgb[:, :, y0:y1, x0:x1]
            sub_p = torch.from_numpy(_ma_he(ch)[:, y0:y1, x0:x1])[None]
            _r, ref = m.forward_refine(sub_rgb, sub_p)
            patches.append(ref[0].numpy())
    out = A.assemble(ch, tiles, np.stack(patches))
    assert np.array_equal(out, ch), "roundtrip not bitwise (max %.3e)" % (
        np.abs(out - ch).max())


# ---------------------------------------------------------------------------
# 3b. RAGGED tiles: interior + 4 borders + 4 corners, different patch sizes
#
# Regression for the formal-run failure `ValueError: all input arrays must have
# the same shape` in eval_refiner.py.  Real numbers from the frozen val cache:
# only 34/1599 candidate patches are full 512x512, there are 892 distinct
# candidate window shapes and every one of the 76 val images mixes 2..51 of
# them, so patches can never be stacked into one array.
# ---------------------------------------------------------------------------

_RAGGED_H, _RAGGED_W = 260, 340


def _ragged_tiles():
    """One interior tile, one per border, one per corner: all different sizes.

    Each of these is a legal (x0, y0, x1, y1) candidate tile; the windows the
    evaluator derives from them are clipped at the image border, which is what
    makes the patch sizes differ.
    """
    return [(60, 40, 190, 150),        # interior
            (0, 60, 90, 180),          # left border
            (250, 30, 340, 140),       # right border
            (110, 0, 240, 80),         # top border
            (150, 190, 260, 260),      # bottom border
            (0, 0, 70, 50),            # top-left corner
            (270, 0, 340, 40),         # top-right corner
            (0, 210, 60, 260),         # bottom-left corner
            (280, 220, 340, 260)]      # bottom-right corner


def _ragged_arrays(seed=3):
    """rgb + the champion MA/HE the refiner consumes + the 4-channel champion.

    `ma_he` is DERIVED from `ch` (class order is EX, HE, SE, MA), exactly like
    the evaluator derives it from the champion's (4, H, W) map.
    """
    g = np.random.default_rng(seed)
    rgb = (g.random((_RAGGED_H, _RAGGED_W, 3)) * 255).astype(np.uint8)
    ch = _synth_champ(_RAGGED_H, _RAGGED_W, seed)
    return rgb, _ma_he(ch), ch


def _random_head_model(seed=7):
    """A refiner that is NOT the identity, so a comparison is not vacuous."""
    torch.manual_seed(seed)
    m = HaarHeteroRefiner().eval()
    for _n, mod in m.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            torch.nn.init.normal_(mod.weight, 0.0, 0.3)
            if mod.bias is not None:
                torch.nn.init.normal_(mod.bias, 0.0, 0.1)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def t_windows_contain_their_tile_and_are_exact():
    """window_for clips at the image border but never loses the tile, and the
    extracted patch is bitwise the real pixels of that window."""
    rgb, ma_he, _ch = _ragged_arrays()
    tiles = _ragged_tiles()
    wins = C.windows_for(tiles, _RAGGED_H, _RAGGED_W, C.PATCH_CONTEXT, 768)
    assert len(wins) == len(tiles)
    for t, w in zip(tiles, wins):
        x0, y0, x1, y1 = t
        ex0, ey0, ex1, ey1 = w
        assert ex0 <= x0 and ey0 <= y0 and ex1 >= x1 and ey1 >= y1, (t, w)
        assert 0 <= ex0 < ex1 <= _RAGGED_W and 0 <= ey0 < ey1 <= _RAGGED_H, w
        for c in range(3):                       # no padding: real pixels only
            assert np.array_equal(
                C.extract_exact(rgb, ma_he, w)[0][c],
                rgb[ey0:ey1, ex0:ex1, c].astype(np.float32) / 255.0), (t, w, c)
        assert np.array_equal(C.extract_exact(rgb, ma_he, w)[1],
                              ma_he[:, ey0:ey1, ex0:ex1]), (t, w)
    shapes = {(w[3] - w[1], w[2] - w[0]) for w in wins}
    assert len(shapes) >= 6, ("expected many distinct patch shapes", shapes)


def t_ragged_buckets_are_grouped_by_shape_without_padding():
    tiles = _ragged_tiles()
    wins = C.windows_for(tiles, _RAGGED_H, _RAGGED_W, C.PATCH_CONTEXT, 768)
    buckets = C.bucket_windows_by_shape(wins)
    seen = []
    for (h, w), idxs in buckets:
        assert idxs, "empty bucket"
        seen.extend(idxs)
        for i in idxs:
            assert (wins[i][3] - wins[i][1], wins[i][2] - wins[i][0]) == (h, w)
    assert sorted(seen) == list(range(len(tiles))), "buckets lost/duplicated tiles"
    rgb, ma_he, _ch = _ragged_arrays()
    # a bucket is homogeneous, so the batch really is padding-free
    for shape, idxs in buckets:
        br, bp = C.extract_exact_batch(rgb, ma_he, wins, idxs)
        assert br.shape == (len(idxs), 3) + shape, br.shape
        assert bp.shape == (len(idxs), 2) + shape, bp.shape
    # mixing shapes must fail closed instead of silently padding
    try:
        C.extract_exact_batch(rgb, ma_he, wins, [0, 1])
        raise AssertionError("mixed-shape batch was not rejected")
    except C.ContractError:
        pass


def t_ragged_bucketed_inference_matches_per_patch_reference():
    """Batched (bucket) inference == one-patch-at-a-time reference."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _random_head_model().to(dev)
    rgb, ma_he, ch = _ragged_arrays(5)
    tiles = _ragged_tiles()
    pat8, _w, st = TI.refine_tiles(m, rgb, ma_he, tiles, _RAGGED_H, _RAGGED_W,
                                   batch=8, device=dev)
    pat1, _w1, st1 = TI.refine_tiles(m, rgb, ma_he, tiles, _RAGGED_H, _RAGGED_W,
                                     batch=1, device=dev)
    assert st["n_distinct_shapes"] == len({(w[3] - w[1], w[2] - w[0])
                                           for w in _w})
    assert st1["mode"] == st["mode"]
    for i, t in enumerate(tiles):
        want = (2, t[3] - t[1], t[2] - t[0])
        assert pat8[i].shape == want and pat1[i].shape == want, (i, pat8[i].shape)
        d = float(np.abs(pat8[i] - pat1[i]).max())
        assert d < 1e-5, "batched != per-patch at tile %d (max|d|=%.3e)" % (i, d)
    # the patches really are ragged -- this is the shape of the original crash
    try:
        np.stack(pat8)
        raise AssertionError("expected ragged patches (np.stack must fail)")
    except ValueError:
        pass
    # and the write-back is identical for both, i.e. placement follows tile order
    a = A.assemble(ch, tiles, pat8)
    b = A.assemble(ch, tiles, pat1)
    assert np.allclose(a, b, atol=1e-5), float(np.abs(a - b).max())
    # order independence: a shuffled tile list must assemble to the same map
    order = [5, 0, 8, 1, 7, 2, 6, 3, 4]
    ts = [tiles[i] for i in order]
    ps, _w2, _s2 = TI.refine_tiles(m, rgb, ma_he, ts, _RAGGED_H, _RAGGED_W,
                                   batch=4, device=dev)
    assert np.allclose(A.assemble(ch, ts, ps), a, atol=1e-5)


def t_ragged_end_to_end_identity_is_bitwise():
    """Zero-init refiner over ragged patches: the assembled map is bitwise the
    champion, EX/SE never move and nothing outside a candidate tile moves."""
    m = HaarHeteroRefiner().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    rgb, ma_he, ch = _ragged_arrays(9)
    tiles = _ragged_tiles()
    pat, _w, _st = TI.refine_tiles(m, rgb, ma_he, tiles, _RAGGED_H, _RAGGED_W,
                                   batch=4, device="cpu")
    out = A.assemble(ch, tiles, pat)
    assert np.array_equal(out, ch), "ragged roundtrip not bitwise (max %.3e)" % (
        np.abs(out - ch).max())
    m2 = _random_head_model()
    pat2, _w2, _s2 = TI.refine_tiles(m2, rgb, ma_he, tiles, _RAGGED_H,
                                     _RAGGED_W, batch=4, device="cpu")
    out2 = A.assemble(ch, tiles, pat2)
    rep = A.assert_bitwise_unchanged(out2, ch, tiles)
    for c in C.CLASSES:
        assert rep["%s_changed_outside" % c] == 0, (c, rep)
    assert rep["EX_total_changed"] == 0 and rep["SE_total_changed"] == 0, rep
    assert np.isfinite(out2).all()


def t_assemble_accepts_ragged_patch_list():
    """A ragged list (one array per tile) writes each patch to its own tile."""
    ch = _synth_champ(120, 160, 4)
    tiles = [(0, 0, 40, 30), (90, 60, 160, 120), (50, 10, 90, 70)]
    plist = [np.full((2, 30, 40), 0.25, np.float32),
             np.full((2, 60, 70), 0.75, np.float32),
             np.full((2, 60, 40), 0.5, np.float32)]
    out = A.assemble(ch, tiles, plist)                     # ragged, not stacked
    assert abs(float(out[3, 5, 5]) - 0.25) < 1e-6
    assert abs(float(out[3, 100, 120]) - 0.75) < 1e-6
    assert abs(float(out[3, 40, 70]) - 0.5) < 1e-6
    rep = A.assert_bitwise_unchanged(out, ch, tiles)
    for c in C.CLASSES:
        assert rep["%s_changed_outside" % c] == 0, (c, rep)
    # also accept the stacked form when every patch happens to be the same size
    st = np.stack([np.full((2, 40, 40), 0.1, np.float32)] * 2)
    out2 = A.assemble(ch, [(0, 0, 40, 40), (40, 40, 80, 80)], st)
    assert abs(float(out2[3, 1, 1]) - 0.1) < 1e-6
    # wrong arity is still rejected
    try:
        A.assemble(ch, tiles, plist[:2])
        raise AssertionError("patch/tile count mismatch accepted")
    except C.ContractError:
        pass


def t_ragged_overlap_is_still_weighted_average():
    """Overlapping RAGGED tiles: exact outside the overlap, fixed Hann mean in."""
    ch = _synth_champ(80, 90, 6)
    tiles = [(0, 0, 30, 40), (20, 0, 60, 50)]      # different sizes, overlap 10 px
    pa = np.full((2, 40, 30), 0.2, np.float32)
    pb = np.full((2, 50, 40), 0.8, np.float32)
    out = A.assemble(ch, tiles, [pa, pb])
    assert abs(float(out[3, 5, 5]) - 0.2) < 1e-6
    assert abs(float(out[3, 45, 55]) - 0.8) < 1e-6
    v = float(out[3, 20, 25])                      # inside the 10 px overlap
    assert 0.2 < v < 0.8, v
    wa = C.hann2d(40, 30)[20, 25]                  # patch A local coords
    wb = C.hann2d(50, 40)[20, 5]                   # patch B local coords (x-20)
    exp = (wa * 0.2 + wb * 0.8) / (wa + wb)
    assert abs(v - exp) < 1e-5, (v, exp)
    assert np.array_equal(out[0], ch[0])


# ---------------------------------------------------------------------------
# 4. losses
# ---------------------------------------------------------------------------

def t_ignore_pixels_do_not_contribute():
    torch.manual_seed(0)
    p = torch.rand(1, 2, 32, 32) * 0.9 + 0.05
    tgt = (torch.rand(1, 2, 32, 32) > 0.5).float()
    valid = torch.ones(1, 1, 32, 32)
    valid[..., 0:8, :] = 0.0                      # ignored band
    l1, _ = L.refiner_losses(p, tgt, valid)
    tgt2 = tgt.clone()
    tgt2[..., 0:8, :] = 1.0 - tgt2[..., 0:8, :]   # flip the ignored band only
    l2, _ = L.refiner_losses(p, tgt2, valid)
    assert torch.allclose(l1, l2, atol=1e-7), (float(l1), float(l2))


def t_fully_ignored_patch_is_zero_and_finite():
    p = torch.rand(1, 2, 16, 16)
    tgt = torch.zeros(1, 2, 16, 16)
    valid = torch.zeros(1, 1, 16, 16)
    loss, parts = L.refiner_losses(p, tgt, valid)
    assert torch.isfinite(loss), loss
    assert abs(float(loss)) < 1e-12, float(loss)
    loss.backward() if loss.requires_grad else None


def t_empty_lesion_is_finite():
    p = torch.rand(2, 2, 32, 32) * 0.5
    tgt = torch.zeros(2, 2, 32, 32)
    valid = torch.ones(2, 1, 32, 32)
    loss, parts = L.refiner_losses(p, tgt, valid, alpha=0.75, gamma=2.0)
    assert torch.isfinite(loss), loss
    assert np.isfinite(parts["focal"]) and np.isfinite(parts["dice"])


def t_perfect_prediction_gives_low_loss():
    tgt = torch.zeros(1, 2, 32, 32)
    tgt[:, :, 10:14, 10:14] = 1.0
    p = torch.where(tgt > 0.5, torch.full_like(tgt, 0.999),
                    torch.full_like(tgt, 0.0005))
    valid = torch.ones(1, 1, 32, 32)
    loss, parts = L.refiner_losses(p, tgt, valid)
    # exact soft-dice loss for this construction
    npos = float(tgt.sum())
    ntot = float(tgt.numel())
    psum = npos * 0.999 + (ntot - npos) * 0.0005
    exp = 1.0 - (2.0 * npos * 0.999) / (psum + npos)
    assert abs(parts["dice"] - exp) < 1e-5, (parts["dice"], exp)
    assert parts["dice"] < 0.05, parts
    assert parts["focal"] < 0.01, parts


def t_loss_is_two_terms_only():
    p = torch.rand(1, 2, 16, 16) * 0.9 + 0.05
    tgt = (torch.rand(1, 2, 16, 16) > 0.5).float()
    valid = torch.ones(1, 1, 16, 16)
    total, parts = L.refiner_losses(p, tgt, valid, w_focal=2.0, w_dice=3.0)
    exp = 2.0 * parts["focal"] + 3.0 * parts["dice"]
    assert abs(float(total) - exp) < 1e-6, (float(total), exp)


_CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "configs")


def _load_cfg(name):
    import json
    with open(os.path.join(_CFG_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def t_frozen_objective_is_30_focal_plus_dice():
    """The formal objective must be exactly 30.0*focal + 1.0*soft_dice, in the
    main config AND in both stage overrides, and nowhere else."""
    base = _load_cfg("refiner_v1.json")
    s1 = _load_cfg("stage1.json")
    s2 = _load_cfg("stage2.json")
    for nm, cfg in (("refiner_v1.json", base), ("stage1.json", s1), ("stage2.json", s2)):
        rep = L.assert_frozen_objective(cfg["loss"])
        assert rep["objective"] == L.FROZEN_OBJECTIVE_STR, nm
        assert cfg["loss"]["weights"] == {"focal_bce": 30.0, "soft_dice": 1.0}, \
            (nm, cfg["loss"]["weights"])
        assert list(cfg["loss"]["terms"]) == ["focal_bce", "soft_dice"], nm
        assert float(cfg["loss"]["focal"]["alpha"]) == 0.75, nm
        assert float(cfg["loss"]["focal"]["gamma"]) == 2.0, nm
    assert base["loss"]["weights"] == s1["loss"]["weights"] == s2["loss"]["weights"]
    assert base["loss"]["focal"] == s1["loss"]["focal"] == s2["loss"]["focal"]
    assert base["loss"]["terms"] == s1["loss"]["terms"] == s2["loss"]["terms"]
    # no third term sneaked in
    for cfg in (base, s1, s2):
        assert len(cfg["loss"]["terms"]) == 2, cfg["loss"]["terms"]


def t_frozen_objective_fails_closed_on_drift():
    ok = _load_cfg("refiner_v1.json")["loss"]
    L.assert_frozen_objective(ok)
    for mutate in (
        lambda c: c.__setitem__("weights", {"focal_bce": 20.0, "soft_dice": 1.0}),
        lambda c: c.__setitem__("weights", {"focal_bce": 30.0, "soft_dice": 1.0,
                                            "tv": 0.1}),
        lambda c: c.__setitem__("terms", ["focal_bce", "soft_dice", "boundary"]),
        lambda c: c.__setitem__("focal", {"alpha": 0.5, "gamma": 2.0}),
        lambda c: c.__setitem__("focal", {"alpha": 0.75, "gamma": 3.0}),
    ):
        bad = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list)
                   else v) for k, v in ok.items()}
        mutate(bad)
        try:
            L.assert_frozen_objective(bad)
            raise AssertionError("drifted objective accepted: %r" % (bad,))
        except ValueError:
            pass


def t_loss_total_and_recorded_focal_share():
    """Total == configured weighted sum, and the recorded focal share is the
    weighted focal contribution over the total (this is what the CSV logs)."""
    p = torch.rand(2, 2, 24, 24) * 0.9 + 0.05
    tgt = (torch.rand(2, 2, 24, 24) > 0.6).float()
    valid = torch.ones(2, 1, 24, 24)
    w_f = L.FROZEN_WEIGHTS["focal_bce"]
    w_d = L.FROZEN_WEIGHTS["soft_dice"]
    total, parts = L.refiner_losses(p, tgt, valid, alpha=L.FROZEN_ALPHA,
                                    gamma=L.FROZEN_GAMMA)
    assert abs(float(total) - (w_f * parts["focal"] + w_d * parts["dice"])) < 1e-6
    assert abs(parts["focal_weighted"] - w_f * parts["focal"]) < 1e-9
    assert abs(parts["dice_weighted"] - w_d * parts["dice"]) < 1e-9
    assert abs(parts["total"] - float(total)) < 1e-6
    exp = parts["focal_weighted"] / (parts["focal_weighted"] + parts["dice_weighted"])
    assert abs(parts["focal_share"] - exp) < 1e-9
    assert 0.0 <= parts["focal_share"] <= 1.0
    # the whole point of R1: with the frozen weights focal is not ~0.5% any more
    assert parts["focal_share"] > 0.002, parts["focal_share"]


def t_validation_nodes_are_frozen():
    base = _load_cfg("refiner_v1.json")
    s2 = _load_cfg("stage2.json")
    s1 = _load_cfg("stage1.json")
    assert list(s2["stages"]["stage2_main"]["checkpoint_nodes"]) == [5, 10, 15, 20, 25, 30]
    assert list(s1["stages"]["stage1_overfit"]["checkpoint_nodes"]) == [200]
    for cfg in (base, s2):
        st = cfg["stages"]["stage2_main"]
        n = list(st["checkpoint_nodes"])
        assert n == list(range(st["eval_every_epochs"], st["epochs"] + 1,
                               st["eval_every_epochs"])), (n, st["eval_every_epochs"])
        assert n[-1] == st["epochs"], n
        assert len(n) == 6, "GO_NO_GO R8 budgets 6 val evaluations, got %d" % len(n)
    for cfg in (base, s1):
        st = cfg["stages"]["stage1_overfit"]
        assert list(st["checkpoint_nodes"]) == [st["epochs"]]


def t_dice_rewards_region_overlap():
    tgt = torch.zeros(1, 2, 32, 32)
    tgt[:, :, 8:24, 8:24] = 1.0
    valid = torch.ones(1, 1, 32, 32)
    good = L.soft_dice(tgt * 0.99 + 0.005, tgt, valid)
    bad = L.soft_dice(torch.zeros_like(tgt) + 0.005, tgt, valid)
    assert float(good) < float(bad)


# ---------------------------------------------------------------------------
# 5. gradient isolation / guards / checkpointing
# ---------------------------------------------------------------------------

def t_only_refiner_gets_gradients():
    torch.manual_seed(0)
    frozen = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1))
    for p in frozen.parameters():
        p.requires_grad_(False)
    frozen.eval()
    refiner = HaarHeteroRefiner()
    rep = G.assert_only_refiner_trainable(refiner, [frozen])
    assert rep["non_refiner_trainable"] == 0
    assert rep["refiner_trainable"] > 0

    rgb = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        feat = frozen(rgb)
    p_champ = torch.sigmoid(feat[:, :2])
    _r, refined = refiner.forward_refine(rgb, p_champ)
    tgt = torch.zeros(1, 2, 32, 32); tgt[:, :, 10:14, 10:14] = 1
    loss, _ = L.refiner_losses(refined, tgt, torch.ones(1, 1, 32, 32))
    loss.backward()
    for p in frozen.parameters():
        assert p.grad is None, "frozen reference received a gradient"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in refiner.parameters())


def t_frozen_reference_guard():
    m = torch.nn.Linear(3, 3)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    G.assert_frozen_reference(m)
    m.train()
    try:
        G.assert_frozen_reference(m)
        raise AssertionError("train-mode reference not rejected")
    except C.ContractError:
        pass
    m.eval()
    m.weight.requires_grad_(True)
    try:
        G.assert_frozen_reference(m)
        raise AssertionError("trainable reference not rejected")
    except C.ContractError:
        pass


def t_no_test_paths_at_entry_points():
    for bad in ("/x/test", "/x/test/idx.json", "/data/TEST/y.npz"):
        try:
            G.assert_no_test_paths(bad)
            raise AssertionError("not rejected: %s" % bad)
        except C.ContractError:
            pass
    G.assert_no_test_paths("/x/val", "/x/train")
    G.assert_val_only_inference("val", "/x/val/idx.json")
    G.assert_val_only_inference("train", "/x/train/idx.json")
    for bad_split in ("test", "TEST", "all", "holdout"):
        try:
            G.assert_split(bad_split)
            raise AssertionError("split %r accepted" % bad_split)
        except C.ContractError:
            pass


def t_checkpoint_strict_load():
    torch.manual_seed(0)
    m = HaarHeteroRefiner()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "refiner.pth")
        torch.save({"model": m.state_dict(), "stage": "unit"}, p)
        m2 = HaarHeteroRefiner()
        ck = torch.load(p, map_location="cpu", weights_only=True)
        missing, unexpected = m2.load_state_dict(ck["model"], strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        s1 = {k: v.clone() for k, v in m.state_dict().items()}
        for k, v in s1.items():
            assert torch.equal(v, m2.state_dict()[k]), k
        # strict load must fail on a mismatched state dict
        bad = dict(ck["model"])
        bad.pop(sorted(k for k in bad if k.endswith("head.bias"))[0])
        try:
            HaarHeteroRefiner().load_state_dict(bad, strict=True)
            raise AssertionError("strict load accepted a missing key")
        except RuntimeError:
            pass


def t_gpu_stages_get_fragmentation_safe_allocator():
    """The supervisor must hand GPU stages the allocator policy that keeps the
    frozen stage-2 batch (16 patches zero-padded to the batch maximum, up to
    576x576, peak 21.34 GiB of the 23.57 GiB torch-visible) from losing the run
    to caching-allocator fragmentation.  Regression for the OOM at step 52 of
    the first formal run.
    """
    import run_formal_v1 as R
    assert R.CUDA_ALLOC_CONF == "max_split_size_mb:128", R.CUDA_ALLOC_CONF
    plain = {"PATH": "/usr/bin"}
    for kind in ("gpu", "eval"):
        env = R.stage_env({"kind": kind, "name": "stage2_train"}, plain)
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == R.CUDA_ALLOC_CONF, env
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["PATH"] == "/usr/bin"
    # a CPU stage must not be given a CUDA policy it does not need
    cpu = R.stage_env({"kind": "cpu", "name": "preflight_tests"}, plain)
    assert cpu.get("PYTORCH_CUDA_ALLOC_CONF") is None, cpu
    # an operator override always wins
    over = {"PATH": "/usr/bin", "PYTORCH_CUDA_ALLOC_CONF": "operator_override"}
    got = R.stage_env({"kind": "gpu", "name": "x"}, over)
    assert got["PYTORCH_CUDA_ALLOC_CONF"] == "operator_override", got
    # the base environment is never mutated in place
    assert plain == {"PATH": "/usr/bin"}, plain
    assert over == {"PATH": "/usr/bin",
                    "PYTORCH_CUDA_ALLOC_CONF": "operator_override"}, over


def t_parameter_count_is_small():
    m = HaarHeteroRefiner()
    n = sum(p.numel() for p in m.parameters())
    assert n < 200_000, "refiner has %d params (too large)" % n
    print("      [info] trainable params = %d" % n)


def main():
    check("identity_is_bitwise", t_identity_is_bitwise)
    check("identity_from_zero_head_only", t_identity_holds_after_randomising_other_layers)
    check("residual_is_bounded", t_residual_is_bounded)
    check("gradients_reach_head", t_gradients_reach_the_head)
    check("haar_level_invertible", t_haar_level_is_invertible)
    check("haar_invertible_odd_sizes", t_haar_invertible_on_odd_sizes)
    check("haar_hf_shape", t_haar_hf_channel_count_and_shape)
    check("haar_hf_no_params", t_haar_hf_has_no_parameters_or_grad)
    check("build_input_11ch", t_build_input_channel_count)
    check("assemble_outside_bitwise", t_assemble_outside_candidate_is_bitwise)
    check("assemble_identity_inside", t_assemble_identity_when_patches_equal_champion)
    check("assemble_context_alignment", t_assemble_uses_patch_centre_when_context_is_present)
    check("assemble_rejects_undersized", t_assemble_rejects_undersized_patch)
    check("assemble_empty_tiles", t_assemble_handles_empty_tiles)
    check("assemble_overlap_weighted", t_assemble_overlap_is_weighted_average)
    check("binarize_fixed_threshold", t_binarize_threshold_is_fixed)
    check("e2e_refiner_roundtrip_bitwise", t_end_to_end_refiner_assembly_roundtrip)
    check("ragged_windows_exact", t_windows_contain_their_tile_and_are_exact)
    check("ragged_buckets_no_padding", t_ragged_buckets_are_grouped_by_shape_without_padding)
    check("ragged_bucketed_equals_per_patch", t_ragged_bucketed_inference_matches_per_patch_reference)
    check("ragged_e2e_identity_bitwise", t_ragged_end_to_end_identity_is_bitwise)
    check("ragged_assemble_placement", t_assemble_accepts_ragged_patch_list)
    check("ragged_overlap_weighted", t_ragged_overlap_is_still_weighted_average)
    check("ignore_pixels_excluded", t_ignore_pixels_do_not_contribute)
    check("fully_ignored_zero", t_fully_ignored_patch_is_zero_and_finite)
    check("empty_lesion_finite", t_empty_lesion_is_finite)
    check("perfect_prediction_low_loss", t_perfect_prediction_gives_low_loss)
    check("loss_is_two_terms", t_loss_is_two_terms_only)
    check("objective_30focal_1dice", t_frozen_objective_is_30_focal_plus_dice)
    check("objective_frozen_fail_closed", t_frozen_objective_fails_closed_on_drift)
    check("loss_total_and_focal_share", t_loss_total_and_recorded_focal_share)
    check("validation_nodes_frozen", t_validation_nodes_are_frozen)
    check("dice_rewards_overlap", t_dice_rewards_region_overlap)
    check("only_refiner_grads", t_only_refiner_gets_gradients)
    check("frozen_reference_guard", t_frozen_reference_guard)
    check("no_test_paths", t_no_test_paths_at_entry_points)
    check("checkpoint_strict_load", t_checkpoint_strict_load)
    check("gpu_stages_allocator_policy",
          t_gpu_stages_get_fragmentation_safe_allocator)
    check("param_count_small", t_parameter_count_is_small)
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for n, e in FAIL:
        print("  FAILED %s: %s" % (n, e))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
