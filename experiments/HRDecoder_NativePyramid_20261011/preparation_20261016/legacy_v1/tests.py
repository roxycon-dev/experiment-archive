"""Regression tests for the previously observed failure mechanisms."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch

from data import ResumableBatches, augment
from evaluate import validate_lock
from losses import SparseLesionLoss
from metrics import accumulate, summarize, TrainProbeGuard, validation_collapse
from model import PyramidDecoder
from train import learning_rate, update_once

PROTOCOL = json.loads((Path(__file__).parent / "protocol.json").read_text())


class LossTests(unittest.TestCase):
    def setUp(self):
        self.loss = SparseLesionLoss(min_negatives=4)

    def test_saturated_false_negatives_have_recovery_gradient(self):
        label = torch.tensor([[[1, 2, 3, 4], [0, 0, 0, 0]]])
        z = torch.full((1, 4, 2, 4), -25., requires_grad=True)
        value, _ = self.loss.one_head(z, label); value.backward()
        for c in range(4):
            self.assertLess(z.grad[0, c, 0, c].item(), -0.01)

    def test_saturated_false_positives_have_suppression_gradient(self):
        label = torch.zeros(1, 8, 8, dtype=torch.long)
        z = torch.full((1, 4, 8, 8), -25.)
        z[:, :, 0, 0] = 25.; z.requires_grad_()
        value, _ = self.loss.one_head(z, label); value.backward()
        self.assertTrue(bool((z.grad[:, :, 0, 0] > .001).all()))

    def test_hard_negative_gradient_does_not_dilute_in_large_background(self):
        gradients = []
        for side in (8, 128):
            label = torch.zeros(1, side, side, dtype=torch.long)
            z = torch.full((1, 4, side, side), -25.)
            z[:, :, 0, 0] = 8.; z.requires_grad_()
            loss, _ = self.loss.one_head(z, label); loss.backward()
            gradients.append(float(z.grad[0, 0, 0, 0]))
        self.assertGreater(gradients[1], gradients[0] * .9)

    def test_ignored_pixels_have_zero_gradient_and_no_value_effect(self):
        label = torch.zeros(1, 8, 8, dtype=torch.long); label[:, :2] = 255
        z = torch.randn(1, 4, 8, 8, requires_grad=True)
        v1, _ = self.loss.one_head(z, label); v1.backward()
        self.assertEqual(int(z.grad[:, :, :2].count_nonzero()), 0)
        changed = z.detach().clone(); changed[:, :, :2] = 100
        v2, _ = self.loss.one_head(changed, label)
        torch.testing.assert_close(v1.detach(), v2)

    def test_empty_valid_batch_is_not_an_optimizer_update(self):
        with self.assertRaises(ValueError):
            self.loss.one_head(torch.zeros(1, 4, 8, 8), torch.full((1, 8, 8), 255))

    def test_illegal_class_rejected(self):
        with self.assertRaises(ValueError):
            self.loss.one_head(torch.zeros(1, 4, 8, 8), torch.full((1, 8, 8), 5))

    def test_absent_class_still_penalizes_background(self):
        low, _ = self.loss.one_head(torch.full((1, 4, 8, 8), -5.), torch.zeros(1, 8, 8, dtype=torch.long))
        high, _ = self.loss.one_head(torch.full((1, 4, 8, 8), 5.), torch.zeros(1, 8, 8, dtype=torch.long))
        self.assertGreater(float(high), float(low) + 1)

    def test_microbatch_loss_and_gradient_equal_batch2(self):
        z = torch.randn(2, 4, 8, 8, requires_grad=True)
        label = torch.randint(0, 5, (2, 8, 8))
        all_loss, _ = self.loss.one_head(z, label)
        full_grad = torch.autograd.grad(all_loss, z)[0]
        pieces = sum(self.loss.one_head(z[i:i+1], label[i:i+1])[0] / 2 for i in range(2))
        micro_grad = torch.autograd.grad(pieces, z)[0]
        torch.testing.assert_close(all_loss, pieces)
        torch.testing.assert_close(full_grad, micro_grad)


class ProtocolTests(unittest.TestCase):
    def test_resumed_exposure_order_crosses_epoch_exactly(self):
        all_batches = list(ResumableBatches(454, 0, 500))
        self.assertEqual(all_batches[226:], list(ResumableBatches(454, 226, 500)))
        self.assertEqual(len({i for pair in all_batches[:227] for i, _ in pair}), 454)

    def test_global_counts_not_mean_of_image_ious(self):
        a = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        b = np.array([[0, 0, 9], [0, 0, 9], [0, 0, 9], [0, 0, 9]])
        self.assertAlmostEqual(summarize(a + b, 2)["mIoU_percent"], 10.)

    def test_counts_ignore_255(self):
        counts = np.zeros((4, 3), np.int64)
        accumulate(counts, np.ones((4, 1, 2), bool), np.array([[1, 255]]))
        self.assertEqual(counts.tolist(), [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])

    def test_exact_threshold_tie_is_negative(self):
        self.assertFalse(bool((torch.tensor(0.).sigmoid() > .5).item()))

    def test_test_access_requires_actual_40k_val_pass(self):
        lock = {"completed": 40000, "primary": "fixed_40000_raw_weights", "source_sha256": "x",
                "status": "VAL_TARGET_PASSED", "test_permitted": True}
        validate_lock(lock, "x")
        for key, value in (("completed", 32000), ("test_permitted", False), ("source_sha256", "y")):
            broken = dict(lock); broken[key] = value
            with self.assertRaises(ValueError):
                validate_lock(broken, "x")

    def test_guard_catches_historical_detail_collapse(self):
        old = {"update": 12000, "mIoU_percent": 36.73585,
               "IoU_percent": [38.91, 43.39, 44.29, 20.36], "recall_percent": [57.34, 53.63, 61.06, 32.11]}
        bad = {"update": 16000, "mIoU_percent": 13.07477,
               "IoU_percent": [6.70, 30.12, 0, 15.48], "recall_percent": [7.20, 34.43, 0, 21.94]}
        reasons = validation_collapse(bad, [old], PROTOCOL)
        self.assertIn("lost_class_SE", reasons)
        self.assertIn("large_validation_miou_drop", reasons)

    def test_guard_allows_ordinary_historical_baseline_oscillation(self):
        old = {"update": 16000, "mIoU_percent": 38.348, "IoU_percent": [40]*4}
        new = {"update": 20000, "mIoU_percent": 32.523, "recall_percent": [45]*4}
        self.assertEqual(validation_collapse(new, [old], PROTOCOL), [])

    def test_training_guard_requires_repeated_saturation(self):
        guard = TrainProbeGuard(PROTOCOL)
        guard.observe({"recall_percent": [30]*4, "positive_logit_median": [0.]*4}, 1000)
        broken = {"recall_percent": [0]*4, "positive_logit_median": [-25.]*4}
        self.assertEqual(guard.observe(broken, 2000), [])
        self.assertEqual(len(guard.observe(broken, 2250)), 4)

    def test_40k_lr_and_warmup(self):
        self.assertAlmostEqual(learning_rate(0, PROTOCOL), 5e-9)
        self.assertAlmostEqual(learning_rate(1500, PROTOCOL), .005 * (1 - 1500/40000))
        self.assertGreater(learning_rate(39999, PROTOCOL), 0)
        with self.assertRaises(ValueError):
            learning_rate(40000, PROTOCOL)

    def test_augmentation_reproducible_and_does_not_change_global_rng(self):
        image = np.arange(80 * 96 * 3, dtype=np.uint8).reshape(80, 96, 3)
        label = np.zeros((80, 96), np.uint8); label[20:50, 30:60] = 4
        np.random.seed(77); before = np.random.get_state()
        a, la = augment(image, label, 64, 42); b, lb = augment(image, label, 64, 42)
        torch.testing.assert_close(a, b); torch.testing.assert_close(la, lb)
        self.assertTrue(np.array_equal(before[1], np.random.get_state()[1]))
        self.assertEqual(set(la.unique().tolist()) - {0, 4, 255}, set())


class DecoderTests(unittest.TestCase):
    def test_first_backward_reaches_every_feature_without_detach(self):
        torch.manual_seed(8)
        decoder = PyramidDecoder()
        image = torch.randn(2, 3, 64, 64)
        stem = torch.randn(2, 64, 32, 32, requires_grad=True)
        features = [torch.randn(2, c, s, s, requires_grad=True)
                    for c, s in ((48, 16), (96, 8), (192, 4), (384, 2))]
        output = decoder(image, stem, features)
        self.assertEqual(tuple(output["logits"].shape), (2, 4, 64, 64))
        loss, _ = SparseLesionLoss(min_negatives=4)(output, torch.randint(0, 5, (2, 64, 64)))
        loss.backward()
        for feature in [stem] + features:
            self.assertTrue(bool(torch.isfinite(feature.grad).all()))
            self.assertGreater(float(feature.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in decoder.parameters()))

    def test_microbatch_performs_one_optimizer_step_and_same_update(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.conv = torch.nn.Conv2d(3, 4, 1)
            def forward(self, x):
                value = self.conv(x)
                return {"logits": value, "auxiliary_logits": value}
        a = Toy(); b = copy.deepcopy(a)
        oa = torch.optim.SGD(a.parameters(), lr=.01, momentum=.9)
        ob = torch.optim.SGD(b.parameters(), lr=.01, momentum=.9)
        batch = {"image": torch.randn(2, 3, 8, 8), "label": torch.randint(0, 5, (2, 8, 8))}
        loss = SparseLesionLoss(min_negatives=4)
        update_once(a, oa, loss, batch, micro_batch=1, amp=False)
        update_once(b, ob, loss, batch, micro_batch=2, amp=False)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y)
        for x, y in zip(oa.state.values(), ob.state.values()):
            torch.testing.assert_close(x["momentum_buffer"], y["momentum_buffer"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
