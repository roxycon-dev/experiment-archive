# -*- coding: utf-8 -*-
"""Round-5 unit tests: weighted fusion loss equivalence, curriculum
schedule, and train_arm extension safety (defaults preserved)."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_core as fc                     # noqa: E402
import run_round5 as r5                       # noqa: E402


def _make(ref_val=0.3):
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8)
    label = torch.randint(0, 3, (2, 8, 8))     # class map (N,H,W)
    ref = torch.full((2, 4, 8, 8), ref_val)
    return logits, label, ref


class TestLossPairW(unittest.TestCase):
    def test_w05_equals_armB_exactly(self):
        """w=0.5 must reproduce fusion_core.loss_pair(arm='B') BITWISE."""
        logits, label, ref = _make()
        got = r5.loss_pair_w(logits, label, ref, 0.5)
        want = fc.loss_pair(logits, label, ref, 'B')
        self.assertTrue(torch.equal(got, want))

    def test_w0_equals_armA_exactly(self):
        logits, label, ref = _make()
        got = r5.loss_pair_w(logits, label, ref, 0.0)
        want = fc.loss_pair(logits, label, ref, 'A')
        self.assertTrue(torch.equal(got, want))

    def test_w1_is_pure_fused(self):
        logits, label, ref = _make()
        probs = torch.sigmoid(logits)
        fused = 0.5 * probs + 0.5 * ref
        got = r5.loss_pair_w(logits, label, ref, 1.0)
        want = fc.dice_prob(fused, label)
        self.assertTrue(torch.equal(got, want))

    def test_intermediate_is_convex_combination(self):
        logits, label, ref = _make()
        a = r5.loss_pair_w(logits, label, ref, 0.0)
        b = r5.loss_pair_w(logits, label, ref, 1.0)
        for w in (0.25, 0.75):
            got = r5.loss_pair_w(logits, label, ref, w)
            want = (1 - w) * a + w * b
            self.assertTrue(torch.allclose(got, want, atol=1e-7))

    def test_ref_validated_and_detached(self):
        logits, label, ref = _make()
        bad = ref.clone().requires_grad_(True)
        with self.assertRaises(Exception):
            r5.loss_pair_w(logits, label, bad * 5.0, 0.5)   # out of [0,1]
        # gradient must NOT flow into ref
        r = ref.clone().requires_grad_(True)
        loss = r5.loss_pair_w(logits.requires_grad_(True), label, r, 0.5)
        loss.backward()
        self.assertIsNone(r.grad)


class TestCurriculum(unittest.TestCase):
    def test_schedule_boundaries(self):
        self.assertEqual(r5.w_v2(0), 0.0)
        self.assertEqual(r5.w_v2(999), 0.0)
        self.assertEqual(r5.w_v2(1000), 0.5)
        self.assertEqual(r5.w_v2(1999), 0.5)
        self.assertEqual(r5.w_v2(2000), 1.0)
        self.assertEqual(r5.w_v2(2999), 1.0)
        self.assertEqual(r5.w_v1(1234), 1.0)


class TestTrainArmExtension(unittest.TestCase):
    def test_defaults_preserved(self):
        """loss_fn/val_points default to the original quarter behavior."""
        import inspect
        import run_protocol as rp
        sig = inspect.signature(rp.train_arm)
        self.assertIsNone(sig.parameters['loss_fn'].default)
        self.assertIsNone(sig.parameters['val_points'].default)
        src = inspect.getsource(rp.train_arm)
        self.assertIn('val_points = [T * q // 4 for q in (1, 2, 3, 4)]',
                      src)
        self.assertIn('rt.loss_pair(lg, lb, rb, arm)', src)

    def test_round5_constants(self):
        self.assertEqual(r5.T5, 3000)
        self.assertEqual(r5.VAL_POINTS, [1000, 2000, 3000])
        self.assertEqual(r5.G1_FLOOR, 0.4185)
        self.assertEqual(r5.G0_MARGIN, 0.002)
        self.assertEqual(r5.G3_MARGIN, 0.002)


if __name__ == '__main__':
    unittest.main()
