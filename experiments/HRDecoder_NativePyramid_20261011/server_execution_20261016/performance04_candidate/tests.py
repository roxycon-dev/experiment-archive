"""Mathematical/protocol regressions. Synthetic inputs; no learning run."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F

from data import (ResumableBatches, augment, load_identity, verify_split_files,
                  IDENTITY_SHA, sha256_file)
from evaluate import validate_lock, validate_checkpoint_lock
from losses import SparseLesionLoss, stable_hard_negative_indices
from metrics import accumulate, summarize
from train import learning_rate, update_once

HERE = Path(__file__).parent
PROTOCOL = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


class LossTests(unittest.TestCase):
    def setUp(self):
        self.loss = SparseLesionLoss(min_negatives=4)

    def test_exact_absent_positive_weight_not_transferred(self):
        label = torch.zeros(1, 2, 2, dtype=torch.long)
        value, _ = self.loss.one_head(torch.zeros(1, 4, 2, 2), label)
        self.assertAlmostEqual(float(value), .30 * np.log(2), places=6)

    def test_independent_reference_formula_unequal_positive_images(self):
        label = torch.tensor([[[1, 1, 2], [0, 255, 0]], [[3, 4, 0], [0, 0, 0]]])
        z = torch.linspace(-2, 2, 48).reshape(2, 4, 2, 3)
        reference = 0.
        for c in range(4):
            dice, positive, selected, dense = [], [], [], []
            valid_count = 0
            for i in range(2):
                v = z[i, c][label[i] != 255]
                y = (label[i][label[i] != 255] == c + 1)
                p = v.sigmoid()
                if bool(y.any()):
                    dice.append(1 - (2 * p[y].sum() + 1e-5) / (p.square().sum() + y.sum() + 1e-5))
                positive.extend(F.softplus(-v[y]).tolist())
                negatives = sorted(v[~y].tolist(), reverse=True)
                k = min(len(negatives), max(4, 3 * int(y.sum())))
                selected.extend(F.softplus(torch.tensor(negatives[:k])).tolist())
                dense.extend(F.binary_cross_entropy_with_logits(v, y.float(), reduction="none").tolist())
                valid_count += v.numel()
            reference += ((sum(dice) / max(len(dice), 1)) +
                          .25 * sum(positive) / max(len(positive), 1) +
                          .25 * sum(selected) / max(len(selected), 1) +
                          .05 * sum(dense) / max(valid_count, 1)) / 4
        value, _ = self.loss.one_head(z, label)
        self.assertAlmostEqual(float(value), float(reference), places=6)

    def test_saturated_false_negatives_recover(self):
        label = torch.tensor([[[1, 2, 3, 4], [0, 0, 0, 0]]])
        z = torch.full((1, 4, 2, 4), -25., requires_grad=True)
        self.loss.one_head(z, label)[0].backward()
        for c in range(4):
            self.assertLess(z.grad[0, c, 0, c].item(), -.01)

    def test_saturated_false_positives_suppressed(self):
        z = torch.full((1, 4, 8, 8), -25.); z[:, :, 0, 0] = 25.; z.requires_grad_()
        self.loss.one_head(z, torch.zeros(1, 8, 8, dtype=torch.long))[0].backward()
        self.assertTrue(bool((z.grad[:, :, 0, 0] > .001).all()))

    def test_hard_negative_does_not_dilute_in_large_background(self):
        values = []
        for side in (8, 128):
            z = torch.full((1, 4, side, side), -25.); z[:, :, 0, 0] = 8.; z.requires_grad_()
            self.loss.one_head(z, torch.zeros(1, side, side, dtype=torch.long))[0].backward()
            values.append(float(z.grad[0, 0, 0, 0]))
        self.assertGreater(values[1], values[0] * .9)

    def test_ohem_tie_membership_and_order(self):
        v = torch.tensor([3., 4., 3., 2., 3., 4., 3.])
        self.assertEqual(stable_hard_negative_indices(v, 4).tolist(), [1, 5, 0, 2])
        self.assertEqual(stable_hard_negative_indices(torch.zeros(20), 4).tolist(), [0, 1, 2, 3])

    def test_tied_ignored_negatives_follow_original_flattened_order(self):
        criterion = SparseLesionLoss(min_negatives=2, dense_weight=0)
        label = torch.tensor([[[255, 0, 0, 255, 0, 0]]])
        z = torch.zeros(1, 4, 1, 6, requires_grad=True)
        criterion.one_head(z, label)[0].backward()
        self.assertEqual(torch.nonzero(z.grad[0, 0].flatten()).flatten().tolist(), [1, 2])

    def test_ignore_invariance_and_gradient(self):
        label = torch.zeros(1, 8, 8, dtype=torch.long); label[:, :2] = 255
        z = torch.randn(1, 4, 8, 8, requires_grad=True)
        a, _ = self.loss.one_head(z, label); a.backward()
        self.assertEqual(int(z.grad[:, :, :2].count_nonzero()), 0)
        changed = z.detach().clone(); changed[:, :, :2] = 100
        torch.testing.assert_close(a.detach(), self.loss.one_head(changed, label)[0])

    def test_all_ignore_global_batch_rejected(self):
        with self.assertRaises(ValueError):
            self.loss.one_head(torch.zeros(2, 4, 8, 8), torch.full((2, 8, 8), 255))

    def test_ignore_zero_never_sums_huge_finite_ignored_logits(self):
        label = torch.zeros(2, 2, 2, dtype=torch.long); label[0] = 255
        denom = self.loss.batch_denominators(label)
        ignored = torch.full((1, 4, 2, 2), 1e38, requires_grad=True)
        value, _ = self.loss.one_head(ignored, label[:1], denominators=denom)
        self.assertEqual(float(value), 0.)
        value.backward()
        self.assertEqual(int(ignored.grad.count_nonzero()), 0)

    def test_illegal_class_and_float_labels_rejected(self):
        for label in (torch.full((1, 8, 8), 5), torch.zeros(1, 8, 8)):
            with self.assertRaises(ValueError):
                self.loss.one_head(torch.zeros(1, 4, 8, 8), label)

    def test_loss_always_fp32(self):
        value, _ = self.loss.one_head(torch.zeros(1, 4, 2, 2, dtype=torch.bfloat16),
                                     torch.zeros(1, 2, 2, dtype=torch.long))
        self.assertEqual(value.dtype, torch.float32)

    def test_shared_denominators_gradient_parity_with_unequal_counts(self):
        for ignore_first in (False, True):
            z = torch.randn(2, 4, 8, 8, requires_grad=True)
            label = torch.zeros(2, 8, 8, dtype=torch.long)
            label[0, 0, :4] = torch.tensor([1, 2, 3, 4]); label[1, :5] = 2
            label[1, 5, :2] = 4
            if ignore_first:
                label[0] = 255
            denom = self.loss.batch_denominators(label)
            full = self.loss.one_head(z, label)[0]
            micro = sum(self.loss.one_head(z[i:i+1], label[i:i+1], denominators=denom)[0] for i in range(2))
            torch.testing.assert_close(full, micro)
            torch.testing.assert_close(torch.autograd.grad(full, z)[0], torch.autograd.grad(micro, z)[0])

    def test_two_losses_choose_hard_negatives_independently(self):
        criterion = SparseLesionLoss(min_negatives=1, dense_weight=0)
        z = torch.full((1, 4, 2, 2), -8.); z[:, :, 0, 0] = 8.; z.requires_grad_()
        s = torch.full((1, 4, 2, 2), -8.); s[:, :, 1, 1] = 8.; s.requires_grad_()
        criterion({"logits": z, "auxiliary_logits": s}, torch.zeros(1, 2, 2, dtype=torch.long))[0].backward()
        self.assertTrue(bool((z.grad[:, :, 0, 0] > 0).all()))
        self.assertEqual(int(z.grad[:, :, 1, 1].count_nonzero()), 0)
        self.assertTrue(bool((s.grad[:, :, 1, 1] > 0).all()))


class ProtocolTests(unittest.TestCase):
    def test_resumed_exposure_order_crosses_epoch_exactly(self):
        batches = list(ResumableBatches(454, 0, 500))
        self.assertEqual(batches[226:], list(ResumableBatches(454, 226, 500)))
        self.assertEqual(len({i for pair in batches[:227] for i, _ in pair}), 454)

    def test_global_counts_not_image_iou_average(self):
        self.assertAlmostEqual(summarize(np.tile([1, 0, 9], (4, 1)), 2)["mIoU_percent"], 10.)

    def test_counts_include_absent_image_fp_and_ignore_255(self):
        counts = np.zeros((4, 3), np.int64)
        accumulate(counts, np.ones((4, 1, 2), bool), np.array([[1, 255]]))
        self.assertEqual(counts.tolist(), [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])

    def test_strict_probability_threshold(self):
        self.assertFalse(bool((torch.tensor(0.).sigmoid() > .5).item()))

    def test_test_lock_40k_val_and_equality_eligible(self):
        lock = {"completed": 40000, "primary": "fixed_40000_raw_weights", "source_sha256": "x",
                "status": "VAL_TARGET_PASSED", "test_permitted": True,
                "val": {"update": 40000, "mIoU_percent": PROTOCOL["val_target_percent"]}}
        validate_lock(lock, "x")
        for key, value in (("completed", 32000), ("test_permitted", False), ("source_sha256", "y"),
                           ("val", {"update": 40000, "mIoU_percent": 40.})):
            broken = dict(lock); broken[key] = value
            with self.assertRaises(ValueError):
                validate_lock(broken, "x")

    def test_test_lock_rejects_nan_and_wrong_checkpoint_evidence(self):
        val = {"update": 40000, "mIoU_percent": PROTOCOL["val_target_percent"]}
        identity = {"identity_sha256": "data", "source_sha256": "x"}
        lock = {"completed": 40000, "executed_updates": 40000, "checkpoint_persisted_updates": 40000,
                "primary": "fixed_40000_raw_weights", "source_sha256": "x", "status": "VAL_TARGET_PASSED",
                "test_permitted": True, "val": val, "run_identity": identity}
        state = {**lock, "status": "COMPLETE", "identity_sha256": "data", "validation_history": [val]}
        validate_checkpoint_lock(lock, state, "data", "x")
        nan_lock = copy.deepcopy(lock); nan_lock["val"]["mIoU_percent"] = float("nan")
        with self.assertRaises(ValueError):
            validate_lock(nan_lock, "x")
        for key, value in (("executed_updates", 39999), ("run_identity", {}),
                           ("validation_history", [{"update": 40000, "mIoU_percent": 10.}])):
            bad = copy.deepcopy(state); bad[key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint_lock(lock, bad, "data", "x")

    def test_identity_bytes_locked_not_just_claimed_assignment(self):
        identity = HERE / "preparation_20261016" / "data_identity.json"
        self.assertEqual(sha256_file(identity), IDENTITY_SHA)
        load_identity(identity)
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "identity.json"
            changed.write_bytes(identity.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                load_identity(changed)

    def test_test_sha_audit_needs_explicit_test_entry(self):
        with self.assertRaises(PermissionError):
            verify_split_files("unused", "unused", ("test",))

    def test_duplicate_image_stems_rejected(self):
        from data import image_index
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("same.jpg", "same.png"):
                (Path(tmp) / name).touch()
            with self.assertRaises(ValueError):
                image_index(tmp)

    def test_schedule_budget(self):
        self.assertAlmostEqual(learning_rate(0, PROTOCOL), 5e-9)
        self.assertAlmostEqual(learning_rate(1500, PROTOCOL), .005 * (1 - 1500/40000))
        self.assertGreater(learning_rate(39999, PROTOCOL), 0)
        with self.assertRaises(ValueError):
            learning_rate(40000, PROTOCOL)

    def test_augmentation_reproducible_and_rng_independent(self):
        image = np.arange(80 * 96 * 3, dtype=np.uint8).reshape(80, 96, 3)
        label = np.zeros((80, 96), np.uint8); label[20:50, 30:60] = 4
        np.random.seed(77); before = np.random.get_state()
        a, la = augment(image, label, 64, 42); b, lb = augment(image, label, 64, 42)
        torch.testing.assert_close(a, b); torch.testing.assert_close(la, lb)
        self.assertTrue(np.array_equal(before[1], np.random.get_state()[1]))
        self.assertEqual(set(la.unique().tolist()) - {0, 4, 255}, set())
        self.assertTrue(bool((la == 255).any()))

    def test_microbatch_one_optimizer_step_equivalent(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.conv = torch.nn.Conv2d(3, 4, 1)
            def forward(self, x):
                v = self.conv(x)
                return {"logits": v, "auxiliary_logits": v}
        a = Toy(); b = copy.deepcopy(a)
        oa = torch.optim.SGD(a.parameters(), lr=.01, momentum=.9)
        ob = torch.optim.SGD(b.parameters(), lr=.01, momentum=.9)
        batch = {"image": torch.randn(2, 3, 8, 8), "label": torch.zeros(2, 8, 8, dtype=torch.long)}
        batch["label"][0, 0, :4] = torch.tensor([1, 2, 3, 4]); batch["label"][1, :6] = 2
        criterion = SparseLesionLoss(min_negatives=4)
        with patch.object(oa, "step", wraps=oa.step) as first:
            update_once(a, oa, criterion, batch, micro_batch=1, amp=False)
            self.assertEqual(first.call_count, 1)
        update_once(b, ob, criterion, batch, micro_batch=2, amp=False)
        for x, y in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(x, y)
        for x, y in zip(oa.state.values(), ob.state.values()):
            torch.testing.assert_close(x["momentum_buffer"], y["momentum_buffer"])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
