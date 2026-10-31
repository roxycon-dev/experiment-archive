"""Synthetic execution-contract tests only; no images, W48, or CUDA training."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from data import ResumableBatches
from losses import SparseLesionLoss
from monitoring import HealthGuard, LearningEvidence
import train


def observation(recall=10., tp=10, miou=30.):
    return {"counts": [[tp, 10, max(100 - tp, 0)] for _ in range(4)],
            "recall_percent": [recall] * 4, "mIoU_percent": miou}


class MonitoringTests(unittest.TestCase):
    def test_learning_requires_two_nodes_after_4k(self):
        guard = LearningEvidence()
        guard.observe(observation(), 3500, stream="probe")
        guard.observe(observation(), 3750, stream="probe")
        self.assertEqual(guard.learned, [False] * 4)
        guard.observe(observation(), 4000, stream="probe")
        self.assertEqual(guard.learned, [False] * 4)
        guard.observe(observation(), 4250, stream="probe")
        self.assertEqual(guard.learned, [True] * 4)

    def test_lost_requires_two_consecutive_nodes_and_recovers(self):
        guard = LearningEvidence()
        for update in (4000, 4250):
            guard.observe(observation(), update, stream="probe")
        self.assertEqual(guard.observe(observation(.1, 0), 4500, stream="probe"), [])
        guard.observe(observation(1.), 4750, stream="probe")
        self.assertEqual(guard.observe(observation(.1, 0), 5000, stream="probe"), [])
        self.assertEqual(len(guard.observe(observation(.1, 0), 5250, stream="probe")), 4)

    def test_never_learned_requires_supervision_and_three_zeros(self):
        guard = LearningEvidence()
        guard.observe(observation(0., 0), 3500, stream="probe", recent_positive=[True] * 4)
        guard.observe(observation(0., 0), 3750, stream="probe", recent_positive=[True] * 4)
        self.assertEqual(guard.observe(observation(0., 0), 4000, stream="probe", recent_positive=[False] * 4), [])
        reasons = guard.observe(observation(0., 0), 4250, stream="probe", recent_positive=[True, False, False, False])
        self.assertEqual(reasons, ["probe_persistently_unlearned_EX"])

    def test_validation_evidence_is_independent(self):
        guard = HealthGuard()
        for update in (4000, 4250):
            guard.probe.observe(observation(), update, stream="probe")
        self.assertEqual(guard.observe_validation(observation(0., 0), 4000), [])
        self.assertEqual(guard.observe_validation(observation(0., 0), 6000), [])
        guard.observe_validation(observation(), 8000)
        guard.observe_validation(observation(), 10000)
        self.assertEqual(guard.observe_validation(observation(.1, 0), 12000), [])
        self.assertEqual(len(guard.observe_validation(observation(.1, 0), 14000)), 4)

    def test_exact_fifteen_point_drop(self):
        guard = HealthGuard()
        guard.observe_validation(observation(miou=45.), 2000)
        self.assertIn("validation_miou_drop_at_least_15pp",
                      guard.observe_validation(observation(miou=30.), 4000))

    def test_supervision_window_and_monitor_restore(self):
        guard = HealthGuard()
        for update in range(1, 4001):
            guard.note_supervision(update, [1, 1, 1, 1] if update == 3000 else [0, 0, 0, 0])
        self.assertFalse(any(v[0] <= 3000 for v in guard.supervision))
        guard.observe_probe(observation(0., 0), 4000)
        restored = HealthGuard(); restored.load_state_dict(guard.state_dict())
        self.assertEqual(restored.state_dict(), guard.state_dict())
        with self.assertRaises(ValueError):
            restored.note_supervision(4002, [1] * 4)
        with self.assertRaises(ValueError):
            restored.observe_probe(observation(), 4250)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.direct_classifiers = nn.ModuleList([nn.Conv2d(3, 4, 1, bias=False) for _ in range(4)])
        self.decoder.auxiliary = nn.Conv2d(3, 4, 1, bias=False)
        self.decoder.readout = nn.Conv2d(3, 4, 1)

    def forward(self, image, diagnostics=False):
        semantic = sum(layer(image) for layer in self.decoder.direct_classifiers) + self.decoder.auxiliary(image)
        output = {"logits": semantic + self.decoder.readout(image), "auxiliary_logits": semantic}
        if diagnostics:
            output["diagnostics"] = {"synthetic": True}
        return output


class UpdateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = TinyModel()
        self.batch = {"image": torch.randn(2, 3, 4, 4), "label": torch.tensor([
            [[1, 1, 0, 255], [2, 0, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]],
            [[0, 0, 0, 255], [0, 0, 2, 2], [0, 0, 0, 0], [4, 4, 4, 0]]])}
        self.loss = SparseLesionLoss(min_negatives=2)

    def test_microbatch_uses_complete_denominators_once(self):
        other = copy.deepcopy(self.model)
        a = torch.optim.SGD(self.model.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
        b = torch.optim.SGD(other.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
        av = train.update_once(self.model, a, self.loss, self.batch, micro_batch=1, amp=False)
        bv = train.update_once(other, b, self.loss, self.batch, micro_batch=2, amp=False)
        self.assertAlmostEqual(av["loss"], bv["loss"], places=6)
        for x, y in zip(self.model.parameters(), other.parameters()):
            torch.testing.assert_close(x, y, rtol=2e-6, atol=2e-7)

    def test_single_ignore_image_contributes_zero_but_update_is_valid(self):
        self.batch["label"][0] = 255
        values = train.update_once(self.model, torch.optim.SGD(self.model.parameters(), lr=.005),
                                   self.loss, self.batch, amp=False)
        self.assertEqual(values["supervision_positive_pixels"], [0, 2, 0, 3])

    def test_all_ignore_rejects_before_optimizer(self):
        self.batch["label"].fill_(255)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.005)
        with mock.patch.object(optimizer, "step") as step:
            with self.assertRaises(ValueError):
                train.update_once(self.model, optimizer, self.loss, self.batch, amp=False)
            step.assert_not_called()

    def test_per_class_gradient_buffers_before_and_after_clip(self):
        values = train.update_once(self.model, torch.optim.SGD(self.model.parameters(), lr=.005),
                                   self.loss, self.batch, clip_norm=.01, amp=False)
        before = values["gradients_before_clip"]["direct_scale_class"]
        after = values["gradients_after_clip"]["direct_scale_class"]
        self.assertEqual(np.asarray(before).shape, (4, 4))
        np.testing.assert_allclose(np.asarray(after), np.asarray(before) * values["clip_coefficient"], rtol=3e-6)
        self.assertIn("decoder.auxiliary", values["gradients_before_clip"]["modules"])
        for name in ("decoder.direct_classifiers.0", "decoder.direct_classifiers.1",
                     "decoder.direct_classifiers.2", "decoder.direct_classifiers.3"):
            left = values["gradients_before_clip"]["parameter_modules"][name]
            right = values["gradients_after_clip"]["parameter_modules"][name]
            self.assertAlmostEqual(right, left * values["clip_coefficient"], places=6)

    def test_module_gradient_detail_exposes_dead_sibling(self):
        for i, layer in enumerate(self.model.decoder.direct_classifiers):
            layer.weight.grad = torch.zeros_like(layer.weight) if i == 0 else torch.ones_like(layer.weight)
        report = train.gradient_norms(self.model)
        self.assertGreater(report["modules"]["decoder.direct_classifiers"], 0.)
        self.assertEqual(report["parameter_modules"]["decoder.direct_classifiers.0"], 0.)
        self.assertGreater(report["parameter_modules"]["decoder.direct_classifiers.1"], 0.)

    def test_nonfinite_rejects_before_step(self):
        self.batch["image"].fill_(float("nan"))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.005)
        with mock.patch.object(optimizer, "step") as step:
            with self.assertRaises(FloatingPointError):
                train.update_once(self.model, optimizer, self.loss, self.batch, amp=False)
            step.assert_not_called()


def paused_state():
    identity = {"source_sha256": "source", "micro_batch": 1}
    state = {"status": "PAUSED", "pause_id": "pause", "completed": 4250,
             "executed_updates": 4250, "checkpoint_persisted_updates": 4250,
             "run_identity": identity, "sampler": train.sampler_cursor(454, 4250, 42),
             "model": {}, "optimizer": {}, "rng": train.rng_state(),
             "loader_rng": torch.Generator().get_state(),
             "guard": {"last_supervision_update": 4250}, "validation_history": [], "best": None}
    progress = {key: copy.deepcopy(state[key]) for key in ("status", "pause_id", "executed_updates",
                "checkpoint_persisted_updates", "run_identity")}
    progress["in_flight_update"] = None
    return state, progress, identity


class ResumeTests(unittest.TestCase):
    def test_exclusive_output_lease_and_own_token_release(self):
        with tempfile.TemporaryDirectory() as temp:
            lease = train.acquire_run_lease(temp, {"unit": True})
            with self.assertRaises(FileExistsError):
                train.acquire_run_lease(temp, {"unit": True})
            train.release_run_lease((lease[0], "other_process"))
            self.assertTrue(lease[0].exists())
            train.release_run_lease(lease)
            self.assertFalse(lease[0].exists())
            replacement = train.acquire_run_lease(temp, {"unit": True})
            train.release_run_lease(replacement)

    def test_only_exact_paused_boundary_is_resumable(self):
        state, progress, identity = paused_state()
        self.assertEqual(train.validate_resume(state, progress, identity, "pause"), 4250)
        for status in ("TRAINING", "FAILED", "COMPLETE", "STOPPED_COLLAPSE", "PAUSED_AT_ENDPOINT"):
            altered = {**state, "status": status}
            with self.assertRaises(RuntimeError):
                train.validate_resume(altered, progress, identity, "pause")

    def test_endpoint_pause_can_only_finalize_with_zero_remaining_updates(self):
        state, progress, identity = paused_state()
        for key in ("completed", "executed_updates", "checkpoint_persisted_updates"):
            state[key] = 40000
        for key in ("executed_updates", "checkpoint_persisted_updates"):
            progress[key] = 40000
        state["sampler"] = train.sampler_cursor(454, 40000, 42)
        state["guard"]["last_supervision_update"] = 40000
        with self.assertRaisesRegex(RuntimeError, "already-completed 40k validation"):
            train.validate_resume(state, progress, identity, "pause")
        state["validation_history"] = [{"update": 40000, "mIoU_percent": 46.}]
        self.assertEqual(train.validate_resume(state, progress, identity, "pause"), 40000)
        self.assertEqual(list(ResumableBatches(454, 40000, 40000, 42)), [])
        with self.assertRaises(RuntimeError):
            train.validate_resume(state, progress, identity, "wrong_pause")

    def test_rejects_uncertain_rollback_identity_and_sampler(self):
        state, progress, identity = paused_state()
        for key, value in (("executed_updates", 4251), ("checkpoint_persisted_updates", 4000),
                           ("in_flight_update", 4251), ("run_identity", {})):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                train.validate_resume(state, {**progress, key: value}, identity, "pause")
        for key in ("rng", "optimizer", "loader_rng", "guard"):
            altered = dict(state); altered.pop(key)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                train.validate_resume(altered, progress, identity, "pause")
        altered = copy.deepcopy(state); altered["sampler"]["offset"] += 2
        with self.assertRaises(RuntimeError):
            train.validate_resume(altered, progress, identity, "pause")
        with self.assertRaises(RuntimeError):
            train.validate_resume(state, progress, identity, "wrong")
        with self.assertRaises(RuntimeError):
            train.validate_resume(state, progress, identity, "pause", failure_present=True)

    def test_sticky_failure_precedes_checkpoint_read(self):
        with tempfile.TemporaryDirectory() as temp:
            Path(temp, "failure.json").write_text('{"status":"FAILED"}', encoding="utf-8")
            with mock.patch("train.parse_args", return_value=types.SimpleNamespace(output=temp, resume=True)), mock.patch("torch.load") as load:
                with self.assertRaisesRegex(RuntimeError, "Terminal failure"):
                    train.main()
                load.assert_not_called()

    def test_sampler_boundary_reproduces_next_epoch_and_seeds(self):
        for completed in (0, 226, 227, 4250, 39999):
            cursor = train.sampler_cursor(454, completed, 42)
            batch = next(iter(ResumableBatches(454, completed, completed + 1, 42)))
            expected = [(cursor["epoch_order"][cursor["offset"] + j], 42 + 2 * completed + j) for j in range(2)]
            self.assertEqual(batch, expected)

    def test_rng_roundtrip_and_atomic_persistence(self):
        train.seed_all(42)
        state = train.rng_state()
        first = (torch.rand(2), np.random.rand(2))
        train.restore_rng(state)
        torch.testing.assert_close(torch.rand(2), first[0]); np.testing.assert_equal(np.random.rand(2), first[1])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "state.pt")
            train.save_checkpoint(path, {"rng": state, "completed": 2})
            self.assertEqual(torch.load(path, weights_only=False)["completed"], 2)
            train.write_json(Path(temp, "progress.json"), {"executed": 2, "persisted": 2})
            self.assertEqual(json.loads(Path(temp, "progress.json").read_text())["executed"], 2)

    def test_persisted_sgd_resume_reproduces_next_model_momentum_and_rng(self):
        train.seed_all(42)
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
        criterion = SparseLesionLoss(min_negatives=2)
        guard = HealthGuard()
        def make_batch():
            return {"image": torch.randn(2, 3, 4, 4),
                    "label": torch.from_numpy(np.random.randint(0, 5, (2, 4, 4))).long()}
        initial = train.update_once(model, optimizer, criterion, make_batch(), amp=False)
        guard.note_supervision(1, initial["supervision_positive_pixels"])
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": train.rng_state(), "guard": guard.state_dict()}
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp, "paused.pt")
            train.save_checkpoint(checkpoint, state)
            batch = make_batch()
            expected = train.update_once(model, optimizer, criterion, batch, amp=False)
            guard.note_supervision(2, expected["supervision_positive_pixels"])
            next_rng = torch.rand(3)
            restored = torch.load(checkpoint, weights_only=False)
            other = TinyModel()
            other.load_state_dict(restored["model"])
            resumed_optimizer = torch.optim.SGD(other.parameters(), lr=.005, momentum=.9, weight_decay=.0005)
            resumed_optimizer.load_state_dict(restored["optimizer"])
            resumed_guard = HealthGuard(); resumed_guard.load_state_dict(restored["guard"])
            train.restore_rng(restored["rng"])
            actual_batch = make_batch()
            torch.testing.assert_close(actual_batch["image"], batch["image"], rtol=0, atol=0)
            torch.testing.assert_close(actual_batch["label"], batch["label"], rtol=0, atol=0)
            actual = train.update_once(other, resumed_optimizer, criterion, actual_batch, amp=False)
            resumed_guard.note_supervision(2, actual["supervision_positive_pixels"])
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(optimizer.state[a]["momentum_buffer"],
                                           resumed_optimizer.state[b]["momentum_buffer"], rtol=0, atol=0)
            self.assertEqual(expected, actual)
            self.assertEqual(guard.state_dict(), resumed_guard.state_dict())
            torch.testing.assert_close(torch.rand(3), next_rng, rtol=0, atol=0)

    def test_execution_ledger_rejects_old_state_missing_and_duplicate_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "train.jsonl")
            for update in (1, 2, 3):
                train.append_json(path, {"update": update, "executed_updates": update,
                                         "checkpoint_persisted_updates": 0})
            train.validate_execution_ledger(path, 3)
            with self.assertRaises(RuntimeError):
                train.validate_execution_ledger(path, 2)
            with path.open("a") as stream:
                stream.write('{"update":3,"executed_updates":3,"checkpoint_persisted_updates":0}\n')
            with self.assertRaises(RuntimeError):
                train.validate_execution_ledger(path, 4)


def valid_preflight():
    source = {name: "unit_sha" for name in train.SOURCE_FILES}
    digest = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    sids = [str(i) for i in range(8)]
    losses = np.linspace(2., 1., 256).tolist()
    record = {"version": 3, "protocol_version": 3, "status": "PASS", "source_sha256": digest,
              "source_files": source, "environment": {"unit": True}, "mode": "server_preflight",
              "formal_updates": 0, "test_accessed": False, "train_probe_sids": sids,
              "unit_tests": {"status": "PASS"}, "backbone_integration": {"status": "PASS"},
              "precision_micro_parity": {"status": "PASS"},
              "input_pipeline": {"status": "PASS", "workers": 8, "batches_verified": 2,
                  "multiprocessing_context": "spawn", "after_CUDA_initialization": True,
                  "pin_memory": True, "prefetch_factor": 1, "persistent_workers": True,
                  "SGD_updates": 0, "matches_seeded_samples": True},
              "data_audit": {"status": "PASS", "identity_sha256": "identity", "test_pixels_read": False,
                             "positive_image_coverage": [2] * 4, "train_probe_sids": sids},
              "real_crop_learning": {"status": "PASS", "discarded_updates": 256,
                  "required_nodes": [224, 256], "required_each_class_iou_percent": 20.,
                  "weights_retained": False, "trajectory": [
                      {"update": 224, "IoU_percent": [20.] * 4}, {"update": 256, "IoU_percent": [20.] * 4}],
                  "step_losses": losses, "first_16_loss_mean": np.mean(losses[:16]),
                  "last_16_loss_mean": np.mean(losses[-16:]),
                  "crops": [{"crop_yxhw": [0, 0, 256, 256], "class_positive_pixels": [1] * 4} for _ in range(8)]},
              "full_resolution_smoke": {"status": "PASS", "discarded_updates": 8, "canvas": 2048,
                  "global_batch": 2, "micro_batch": 1, "metrics": [{}] * 8,
                  "internal_parameter_module_gradient_gate": {"status": "PASS", "active_modules": {"decoder.deep.0.0": True}},
                  "all_BN_buffers_unchanged": True, "weights_retained": False}}
    return record, digest


class PreflightContractTests(unittest.TestCase):
    def test_completed_server_gate_is_required(self):
        record, digest = valid_preflight()
        train.validate_preflight(record, digest, "identity", {"unit": True})
        for key, value in (("mode", "static_only"), ("formal_updates", 1), ("test_accessed", True),
                           ("source_files", {}), ("environment", {}), ("protocol_version", 1)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                train.validate_preflight({**record, key: value}, digest, "identity", {"unit": True})
        missing = copy.deepcopy(record); missing.pop("precision_micro_parity")
        with self.assertRaises(RuntimeError):
            train.validate_preflight(missing, digest, "identity", {"unit": True})

    def test_top_level_pass_cannot_hide_failed_numeric_gates(self):
        record, digest = valid_preflight()
        mutations = [("input_pipeline", "workers", 4),
                     ("input_pipeline", "multiprocessing_context", "fork"),
                     ("input_pipeline", "after_CUDA_initialization", False),
                     ("input_pipeline", "matches_seeded_samples", False),
                     ("real_crop_learning", "discarded_updates", 128),
                     ("real_crop_learning", "last_16_loss_mean", 4.),
                     ("real_crop_learning", "trajectory", [{"update": 224, "IoU_percent": [20.] * 4},
                                                             {"update": 256, "IoU_percent": [20., 20., 0., 20.]}]),
                     ("full_resolution_smoke", "discarded_updates", 2),
                     ("full_resolution_smoke", "micro_batch", 2),
                     ("full_resolution_smoke", "all_BN_buffers_unchanged", False),
                     ("full_resolution_smoke", "internal_parameter_module_gradient_gate",
                      {"status": "PASS", "active_modules": {"decoder.deep.0.0": False}}),
                     ("data_audit", "identity_sha256", "changed")]
        for stage, key, value in mutations:
            changed = copy.deepcopy(record); changed[stage][key] = value
            with self.subTest(stage=stage, key=key), self.assertRaises(RuntimeError):
                train.validate_preflight(changed, digest, "identity", {"unit": True})

    def test_formal_protocol_rejects_silent_hyperparameter_or_budget_changes(self):
        protocol = json.loads((train.HERE / "protocol.json").read_text(encoding="utf-8"))
        train.validate_protocol(protocol)
        for key, value in (("updates", 40001), ("global_batch", 1), ("threshold", .4),
                           ("decoder_norm", "GroupNorm"), ("precision", "FP32"),
                           ("val_target_percent", 40.), ("classes", ["MA", "EX", "HE", "SE"]),
                           ("loss", {**protocol["loss"], "positive_weight": .5})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                train.validate_protocol({**protocol, key: value})


class EvaluationTests(unittest.TestCase):
    def test_s_and_z_counts_and_literal_sigmoid_threshold(self):
        class Fixed(nn.Module):
            def __init__(self):
                super().__init__(); self.anchor = nn.Parameter(torch.zeros(()))
            def forward(self, image, diagnostics=False):
                z = torch.full((1, 4, 2, 3), 1e-8) + self.anchor
                out = {"logits": z, "auxiliary_logits": z + 2}
                if diagnostics:
                    out["diagnostics"] = {"synthetic": True}
                return out
        class Dataset:
            split = "train"
            def __len__(self): return 1
            def evaluation(self, index):
                return torch.zeros(3, 2, 3), np.array([[1, 2, 255], [3, 4, 0]]), {"sid": "unit"}
        model = Fixed().train()
        result = train.evaluate(model, Dataset(), training_probe=True, amp=False)
        self.assertTrue(model.training)
        self.assertEqual(result["counts"], [[0, 0, 1]] * 4)
        self.assertEqual(result["heads"]["S"]["counts"], [[1, 4, 0]] * 4)
        self.assertEqual(len(result["heads"]["Z"]["positive_logit_quantiles"]), 4)


if __name__ == "__main__":
    unittest.main()
