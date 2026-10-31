import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hrplus_core import (  # noqa: E402
    _axis_positions,
    tiled_forward,
    blend_probs,
    resize_logit,
    pad_to_size,
    crop_arrays,
    foreground_crop_bbox,
    lovasz_grad,
    lovasz_hinge_flat,
    multiclass_lovasz_hinge,
)

torch.set_num_threads(2)


def _linear_forward(scale=1.0, bias=0.0):
    """Deterministic forward_fn: per-pixel linear map of the patch, per class."""
    def fn(patch):
        b, _, th, tw = patch.shape
        out = scale * patch.mean(dim=1, keepdim=True) + bias
        return out.expand(b, 4, th, tw).contiguous()
    return fn


class TestAxisPositions(unittest.TestCase):
    def test_full_irregular_coverage(self):
        # irregular size where (total - tile) is not a multiple of stride
        for total, tile, stride in [(10, 4, 3), (11, 4, 3), (10, 4, 4), (7, 4, 3)]:
            pos = _axis_positions(total, tile, stride)
            self.assertEqual(pos[-1], total - tile)
            self.assertTrue(all(p >= 0 for p in pos))
            covered = np.zeros(total, dtype=int)
            for p in pos:
                covered[p:p + tile] += 1
            self.assertTrue((covered > 0).all(), f'uncovered for {total},{tile},{stride}')

    def test_small_image_single_position(self):
        self.assertEqual(_axis_positions(3, 4, 2), [0])
        self.assertEqual(_axis_positions(4, 4, 2), [0])

    def test_invalid_stride(self):
        with self.assertRaises(AssertionError):
            _axis_positions(10, 4, 0)
        with self.assertRaises(AssertionError):
            _axis_positions(10, 4, 5)
        with self.assertRaises(AssertionError):
            tiled_forward(torch.zeros(1, 3, 8, 8), _linear_forward(),
                          tile_size=4, stride=5, num_classes=4)


class TestTiledForward(unittest.TestCase):
    def test_full_irregular(self):
        img = torch.randn(1, 3, 10, 7)
        out = tiled_forward(img, _linear_forward(2.0, 0.5), tile_size=(4, 4),
                            stride=3, num_classes=4)
        self.assertEqual(out.shape, (1, 4, 10, 7))
        self.assertEqual(out.dtype, torch.float32)
        # every output pixel equals 2*mean+0.5 of its pixel
        expected = 2.0 * img.mean(dim=1, keepdim=True) + 0.5
        self.assertTrue(torch.allclose(out[:, :1], expected.expand(1, 1, 10, 7), atol=1e-5))

    def test_small_image_padded_and_cropped(self):
        img = torch.randn(1, 3, 3, 5)
        seen = []

        def fn(patch):
            seen.append(patch.shape)
            return _linear_forward(1.0, 0.0)(patch)

        out = tiled_forward(img, fn, tile_size=4, stride=2, num_classes=4)
        self.assertEqual(out.shape, (1, 4, 3, 5))
        self.assertEqual(seen[0], (1, 3, 4, 4))  # patched to full tile
        expected = img.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(out[:, :1], expected, atol=1e-5))

    def test_overlap_exact_averaging(self):
        img = torch.randn(1, 3, 8, 8)
        out = tiled_forward(img, _linear_forward(1.0, 0.0), tile_size=4, stride=2,
                            num_classes=4)
        expected = img.mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(out[:, :1], expected, atol=1e-5))


class TestBlendProbs(unittest.TestCase):
    def test_weighted_mix(self):
        a = torch.rand(1, 4, 4, 4)
        b = torch.rand(1, 4, 4, 4)
        out = blend_probs(a, b, weight_a=0.3)
        self.assertTrue(torch.allclose(out, 0.3 * a + 0.7 * b))

    def test_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            blend_probs(torch.rand(1, 4, 4, 4), torch.rand(1, 4, 5, 4))


class TestResizeLogit(unittest.TestCase):
    def test_align_corners(self):
        x = torch.rand(1, 4, 2, 2)
        y = resize_logit(x, (4, 4), align_corners=True)
        self.assertEqual(y.shape, (1, 4, 4, 4))
        self.assertTrue(torch.allclose(y[:, :, 0, 0], x[:, :, 0, 0]))


class TestCropHelpers(unittest.TestCase):
    def test_pad_alignment_values(self):
        img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        gt = np.array([[1, 2, 0], [0, 3, 255]], dtype=np.uint8)
        pi, pg = pad_to_size(img, gt, (4, 5), img_pad=7, seg_pad=255)
        self.assertEqual(pi.shape, (4, 5, 3))
        self.assertEqual(pg.shape, (4, 5))
        self.assertTrue((pi[2:, :] == 7).all() and (pi[:2, 3:] == 7).all())
        self.assertTrue((pg[2:, :] == 255).all() and (pg[:2, 3:] == 255).all())
        self.assertTrue((pg[:2, :3] == gt).all())

    def test_no_pad_when_larger(self):
        img = np.zeros((5, 5, 3), np.uint8)
        gt = np.zeros((5, 5), np.uint8)
        pi, pg = pad_to_size(img, gt, (3, 3))
        self.assertEqual(pi.shape, (5, 5, 3))
        self.assertEqual(pg.shape, (5, 5))

    def test_crop_aligned(self):
        img = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
        gt = np.arange(36, dtype=np.uint8).reshape(6, 6)
        ci, cg = crop_arrays(img, gt, 2, 3, 3, 2)
        self.assertTrue(np.array_equal(ci, img[2:5, 3:5]))
        self.assertTrue(np.array_equal(cg, gt[2:5, 3:5]))

    def test_guaranteed_positive_crop(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        gt[0, 0] = 2  # lesion pixel at corner -> only one valid crop
        rng = np.random.RandomState(0)
        for _ in range(20):
            y0, x0 = foreground_crop_bbox(gt, 8, fg_prob=0.0, rng=rng)
            window = gt[y0:y0 + 8, x0:x0 + 8]
            self.assertIn(2, window)

    def test_fg_prob_one_is_uniform(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        gt[0, 0] = 1
        rng = mock.MagicMock()
        rng.random.return_value = 1.0  # >= fg_prob -> uniform branch
        rng.randint.side_effect = lambda lo, hi=None: (lo if hi is None else lo)
        # With mocked rng, uniform branch calls randint(0, h-cs+1) -> returns 0
        y0, x0 = foreground_crop_bbox(gt, 8, fg_prob=0.5, rng=rng)
        self.assertEqual((y0, x0), (0, 0))

    def test_all_random_when_no_lesion(self):
        gt = np.zeros((10, 10), dtype=np.int64)
        rng = np.random.RandomState(1)
        y0, x0 = foreground_crop_bbox(gt, 5, fg_prob=0.0, rng=rng)
        self.assertTrue(0 <= y0 <= 5 and 0 <= x0 <= 5)


class TestLovasz(unittest.TestCase):
    def _logits_for(self, labels_tensor):
        """Hand-built 4-channel logits; channels 0..3 <-> labels 1..4."""
        b, h, w = labels_tensor.shape
        logits = torch.zeros(b, 4, h, w, dtype=torch.float64, requires_grad=True)
        return logits

    def test_perfect_logits_loss_zero(self):
        # labels 1..4 map to channels 0..3; hand-built logits:
        # channel c gets +10 where label==c+1, -10 elsewhere (on valid pixels).
        tgt = torch.zeros(1, 8, 8, dtype=torch.long)
        tgt[0, :2, :] = 1
        tgt[0, 2:4, :] = 2
        tgt[0, 4:6, :] = 3
        tgt[0, 6:, :] = 4
        logits = torch.full((1, 4, 8, 8), -10.0, dtype=torch.float64, requires_grad=True)
        for c in range(4):
            logits = logits.clone()
            logits[0, c][tgt[0] == c + 1] = 10.0
        logits = logits.detach().requires_grad_(True)
        loss = multiclass_lovasz_hinge(logits, tgt)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_wrong_class_loss_positive(self):
        tgt = torch.zeros(1, 4, 4, dtype=torch.long)
        tgt[0, :, :] = 1  # all class 1
        # channel 0 predicts wrong (negative where label 1), others correct
        logits = torch.full((1, 4, 4, 4), -10.0, dtype=torch.float64)
        logits[0, 0] = -10.0  # wrong on all class-1 pixels
        logits[0, 1:] = 10.0  # correct: label never 2..4
        loss = multiclass_lovasz_hinge(logits, tgt)
        self.assertGreater(float(loss), 0.0)

    def test_each_channel_independent_grad_nonzero(self):
        tgt = torch.zeros(1, 6, 6, dtype=torch.long)
        tgt[0, :3, :] = 1
        tgt[0, 3:, :] = 3
        logits = torch.zeros(1, 4, 6, 6, dtype=torch.float64, requires_grad=True)
        logits = logits + torch.tensor([[[[1.0]]]])  # break symmetry
        logits = logits.detach().requires_grad_(True)
        # build asymmetric logits
        base = torch.zeros(1, 4, 6, 6, dtype=torch.float64)
        for c in range(4):
            base[0, c] = 0.1 * (c + 1)
        base = base.requires_grad_(True)
        loss = multiclass_lovasz_hinge(base, tgt)
        loss.backward()
        self.assertIsNotNone(base.grad)
        for c in range(4):
            g = base.grad[0, c]
            self.assertGreater(float(g.abs().sum()), 0.0, f'channel {c} grad zero')

    def test_ignore_perturbation_invariance_and_zero_grad(self):
        tgt = torch.zeros(1, 4, 4, dtype=torch.long)
        tgt[0, :, :] = 2
        tgt[0, 0, 0] = 255
        tgt[0, 3, 3] = 255

        def make(pert):
            base = torch.zeros(1, 4, 4, 4, dtype=torch.float64)
            for c in range(4):
                base[0, c] = 0.05 * (c + 1)
            base[0, 0, 0, 0] = pert   # perturb only IGNORE pixels (any channel)
            base[0, 3, 3, 3] = -pert
            return base.requires_grad_(True)

        l1 = make(100.0)
        loss1 = multiclass_lovasz_hinge(l1, tgt)
        loss1.backward()
        l2 = make(-100.0)
        loss2 = multiclass_lovasz_hinge(l2, tgt)
        loss2.backward()
        self.assertAlmostEqual(float(loss1), float(loss2), places=6)
        # gradient is zero exactly at ignore pixel locations
        for c in range(4):
            self.assertEqual(float(l1.grad[0, c, 0, 0]), 0.0)
            self.assertEqual(float(l1.grad[0, c, 3, 3]), 0.0)
        # ignore pixels are NOT replaced by a class (they only affect via exclusion)
        self.assertGreater(float(l1.grad.abs().sum()), 0.0)

    def test_all_ignore_graph_zero(self):
        tgt = torch.full((1, 3, 3), 255, dtype=torch.long)
        logits = torch.randn(1, 4, 3, 3, dtype=torch.float64, requires_grad=True)
        loss = multiclass_lovasz_hinge(logits, tgt)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_absent_class_averaged_hand_values(self):
        # only class 1 present; hand-compute per-pixel binary lovasz for channel 0.
        # 2 valid pixels, one fg one bg. logits ch0: [0.0, 1.0].
        # labels: [1, 0]; errors = 1 - logit*sign, sign=+1 for fg, -1 for bg:
        #   fg pixel: 1 - 0.0*1 = 1.0 ; bg pixel: 1 - 1.0*(-1) = 2.0
        # sorted desc: [2.0 (bg, gt=0), 1.0 (fg, gt=1)]
        # lovasz_grad: gts=1; intersection = 1 - cumsum([0,1]) = [1, 0]
        #              union = 1 + cumsum([1,0]) = [2, 2]; jacc=[0.5, 1.0]
        #              jacc diff (p>1): [0.5, 0.5]
        # loss = dot(relu([2.0,1.0]), [0.5,0.5]) = 1.0 + 0.5 = 1.5
        # channels 1..3: labels all 0, errors with logits 0 -> [1,1] sorted,
        #   gt_sorted=[0,0]; gts=0; intersection=[0,0]; union=[1,2]; jacc=[1,1]
        #   diff -> [1, 0]; relu*grad = 1*1 + 1*0 = 1.0 each
        # total = 1.5 + 1.0 + 1.0 + 1.0 = 4.5 ; /4 = 1.125
        tgt = torch.tensor([[[1, 0]]], dtype=torch.long)
        logits = torch.zeros(1, 4, 1, 2, dtype=torch.float64)
        logits[0, 0, 0, 0] = 0.0
        logits[0, 0, 0, 1] = 1.0
        logits = logits.requires_grad_(True)
        loss = multiclass_lovasz_hinge(logits, tgt)
        self.assertAlmostEqual(float(loss), 1.125, places=6)

    def test_singleton_loss(self):
        tgt = torch.ones(1, 1, 1, dtype=torch.long)
        logits = torch.zeros(1, 4, 1, 1, dtype=torch.float64, requires_grad=True)
        loss = multiclass_lovasz_hinge(logits, tgt)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestAllClassesReachable(unittest.TestCase):
    def test_each_class_produces_channel_grad(self):
        for cls in range(1, 5):
            tgt = torch.full((1, 4, 4), cls, dtype=torch.long)
            logits = torch.zeros(1, 4, 4, 4, dtype=torch.float64, requires_grad=True)
            loss = multiclass_lovasz_hinge(logits, tgt)
            loss.backward()
            g = logits.grad[0, cls - 1]
            self.assertGreater(float(g.abs().sum()), 0.0, f'class {cls} unreachable')


if __name__ == '__main__':
    unittest.main()
