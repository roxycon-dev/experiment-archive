"""v3 structural tests, with synthetic inputs and no optimizer/training updates.

Actual W48 checks use NATIVE_PYRAMID_PRETRAINED when provided, or the existing
locked local asset. Missing weights are reported as SKIP, never as a PASS.
"""
import copy
import json
import math
import os
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from model import (BilinearUpsample, ChannelLayerNorm, NativePyramid,
                   PyramidDecoder, logit_statistics, resize)


def synthetic(batch=1, requires_grad=False):
    image = torch.randn(batch, 3, 64, 64)
    stem = torch.randn(batch, 64, 32, 32, requires_grad=requires_grad)
    features = [torch.randn(batch, channels, side, side, requires_grad=requires_grad)
                for channels, side in ((48, 16), (96, 8), (192, 4), (384, 2))]
    return image, stem, features


def objective(output):
    return output["logits"].square().mean() + .2 * output["auxiliary_logits"].square().mean()


def activate_residual_fixture(decoder):
    # An artificial nonzero fixture checks connectivity after zero endpoints.
    # This is not an optimizer update or a crop-learning acceptance experiment.
    with torch.no_grad():
        decoder.auxiliary.weight.normal_(std=.01)
        decoder.readout.weight.normal_(std=.01)


class ChannelLayerNormV3Tests(unittest.TestCase):
    def test_channel_only_population_variance_and_fp32_bf16(self):
        torch.manual_seed(1)
        norm = ChannelLayerNorm(7)
        value = (torch.randn(2, 7, 5, 6) * 10 + 100).bfloat16()
        floating = value.float()
        expected = ((floating - floating.mean(1, keepdim=True)) /
                    (floating.var(1, unbiased=False, keepdim=True) + 1e-5).sqrt()).bfloat16()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = norm(value)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(norm.weight.numel(), 7)
        self.assertEqual(norm.bias.numel(), 7)

    def test_distant_padding_does_not_change_valid_pixel_or_gradient(self):
        torch.manual_seed(2)
        norm = ChannelLayerNorm(8)
        value = torch.randn(2, 8, 3, 4, requires_grad=True)
        padded = F.pad(value.detach(), (0, 20, 0, 17), value=-33).requires_grad_()
        reference = norm(value)
        actual = norm(padded)
        torch.testing.assert_close(reference, actual[:, :, :3, :4], rtol=0, atol=0)
        actual[:, :, 0, 0].square().sum().backward()
        self.assertEqual(padded.grad[:, :, 3:, :].count_nonzero().item(), 0)
        self.assertEqual(padded.grad[:, :, :, 4:].count_nonzero().item(), 0)


class DecoderV3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261011)

    def test_exact_parameter_count_unique_semantic_bias_and_zero_endpoints(self):
        decoder = PyramidDecoder(False)
        self.assertEqual(sum(p.numel() for p in decoder.parameters()), 246132)
        self.assertFalse(hasattr(decoder, "rgb"))
        self.assertEqual(decoder.detail[0][0].in_channels, 96)
        self.assertFalse(any(isinstance(layer, torch.nn.GroupNorm) for layer in decoder.modules()))
        self.assertIsNone(decoder.auxiliary.bias)
        self.assertTrue(all(layer.bias is None for layer in decoder.direct_classifiers))
        torch.testing.assert_close(decoder.direct_bias,
                                   torch.full((4,), math.log(.01 / .99)), rtol=0, atol=0)
        for layer in decoder.direct_classifiers:
            self.assertGreater(layer.weight.count_nonzero().item(), 0)
            self.assertTrue(.008 < layer.weight.std().item() < .012)
        for value in (decoder.auxiliary.weight, decoder.readout.weight, decoder.readout.bias):
            self.assertEqual(value.count_nonzero().item(), 0)

    def test_initial_output_equals_direct_semantics_with_no_phase_residual(self):
        decoder = PyramidDecoder(False)
        image, stem, features = synthetic(batch=2)
        with torch.no_grad():
            actual = decoder(image, stem, features, diagnostics=True)
            direct = 0
            for head, feature in zip(decoder.direct_classifiers, features):
                direct = direct + resize(head(feature), features[0].shape[-2:])
            direct = direct + decoder.direct_bias.view(1, 4, 1, 1)
            torch.testing.assert_close(actual["auxiliary_logits"], direct, rtol=0, atol=0)
            torch.testing.assert_close(actual["logits"], resize(direct, (64, 64)), rtol=0, atol=0)
        self.assertEqual(tuple(actual["logits"].shape), (2, 4, 64, 64))
        for branch in ("semantic_residual", "detail_residual"):
            self.assertEqual(actual["diagnostics"][branch]["abs_max"], [0.] * 4)
        # Scalar-only diagnostics must serialize, with no hidden tensor retention.
        json.dumps(actual["diagnostics"], allow_nan=False)
        self.assertEqual(len(actual["diagnostics"]["direct_logits"]), 4)
        self.assertTrue(all(row["quantile_method"] == "exact"
                            for row in actual["diagnostics"]["direct_logits"]))
        self.assertNotIn("diagnostics", decoder(image, stem, features))

    def test_direct_linear_exchange_equivalence_with_shared_bias_once(self):
        decoder = PyramidDecoder(False)
        _, _, features = synthetic(batch=2)
        grid = features[0].shape[-2:]
        direct = 0
        for head, feature in zip(decoder.direct_classifiers, features):
            direct = direct + resize(head(feature), grid)
        direct = direct + decoder.direct_bias.view(1, 4, 1, 1)
        concatenated = torch.cat([resize(feature, grid) for feature in features], 1)
        weight = torch.cat([head.weight for head in decoder.direct_classifiers], 1)
        reference = F.conv2d(concatenated, weight, decoder.direct_bias)
        torch.testing.assert_close(direct, reference, rtol=2e-6, atol=2e-6)

    def test_shared_bias_is_added_after_scale_cancellation(self):
        decoder = PyramidDecoder(False)
        image, stem, features = synthetic()
        with torch.no_grad():
            decoder.direct_bias.fill_(1.)
            for i, head in enumerate(decoder.direct_classifiers):
                head.weight.zero_()
                head.weight[:, 0] = 1. if i < 2 else 0.
            for i, feature in enumerate(features):
                feature.zero_()
                feature[:, 0] = 1e8 if i == 0 else -1e8 if i == 1 else 0.
            output = decoder(image, stem, features)
        # Adding bias before the cancelling 1e8 terms would lose it to rounding.
        torch.testing.assert_close(output["auxiliary_logits"], torch.ones_like(output["auxiliary_logits"]),
                                   rtol=0, atol=0)

    def test_two_resizes_are_kept_distinct(self):
        value = torch.zeros(1, 1, 2, 2)
        value[:, :, 0, 0] = 1
        self.assertGreater((resize(resize(value, (4, 4)), (16, 16)) -
                            resize(value, (16, 16))).abs().max().item(), .01)

    def test_first_backward_endpoints_direct_paths_and_zero_internal_task_gradient(self):
        decoder = PyramidDecoder(False)
        image, stem, features = synthetic(requires_grad=True)
        objective(decoder(image, stem, features)).backward()
        for layer in [*decoder.direct_classifiers, decoder.auxiliary, decoder.readout]:
            self.assertTrue(torch.isfinite(layer.weight.grad).all().item())
            self.assertTrue((layer.weight.grad.flatten(1).norm(dim=1) > 0).all().item())
        for feature in features:
            self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertEqual(stem.grad.count_nonzero().item(), 0)
        for name, parameter in decoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            if not name.startswith(("direct_", "auxiliary.", "readout.")):
                self.assertEqual(parameter.grad.count_nonzero().item(), 0, name)

    def test_nonzero_endpoint_fixture_opens_internal_and_stem_task_gradients(self):
        decoder = PyramidDecoder(False)
        activate_residual_fixture(decoder)
        image, stem, features = synthetic(requires_grad=True)
        objective(decoder(image, stem, features)).backward()
        self.assertGreater(stem.grad.abs().sum().item(), 0)
        for name, parameter in decoder.named_parameters():
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_checkpoint_matches_output_and_all_decoder_gradients(self):
        ordinary = PyramidDecoder(False)
        activate_residual_fixture(ordinary)
        recomputed = copy.deepcopy(ordinary)
        recomputed.checkpointing = True
        inputs = synthetic()
        a = ordinary(*inputs)
        b = recomputed(*inputs)
        torch.testing.assert_close(a["logits"], b["logits"], rtol=0, atol=0)
        torch.testing.assert_close(a["auxiliary_logits"], b["auxiliary_logits"], rtol=0, atol=0)
        objective(a).backward()
        objective(b).backward()
        for (name, left), (_, right) in zip(ordinary.named_parameters(), recomputed.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)

    def test_full_batch_and_two_microbatches_match_head_gradients(self):
        full = PyramidDecoder(False)
        activate_residual_fixture(full)
        micro = copy.deepcopy(full)
        image, stem, features = synthetic(batch=2)
        objective(full(image, stem, features)).backward()
        for i in range(2):
            objective(micro(image[i:i + 1], stem[i:i + 1],
                            [value[i:i + 1] for value in features])).div(2).backward()
        for (name, left), (_, right) in zip(full.named_parameters(), micro.named_parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_sampled_diagnostics_are_explicit_and_means_are_exact(self):
        value = torch.arange(200).view(1, 4, 5, 10).float()
        result = logit_statistics(value, max_samples=7)
        self.assertEqual(result["quantile_method"], "deterministic_flat_stride_sample")
        self.assertEqual(result["mean"], value.mean((0, 2, 3)).tolist())
        self.assertTrue(all(n <= 7 for n in result["quantile_samples_per_class"]))


class LockedBackboneV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        default = Path(__file__).resolve().parents[1] / "HRDecoder_Ready_20261002/assets/hrnetv2_w48-d2186c55.pth"
        path = Path(os.environ.get("NATIVE_PYRAMID_PRETRAINED", str(default)))
        if not path.is_file():
            raise unittest.SkipTest("Locked ImageNet W48 unavailable; set NATIVE_PYRAMID_PRETRAINED")
        cls.model = NativePyramid(path, checkpointing=False)

    def test_counts_tensor_identity_fusions_and_frozen_bn(self):
        model = self.model
        model.train()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 65571252)
        initialization = model.backbone.initialization
        self.assertEqual(initialization["verified_backbone_tensors"], 1830)
        self.assertEqual(initialization["ignored_classification_tensors"], 126)
        self.assertEqual(len(initialization["ignored_classification_keys"]), 126)
        self.assertEqual(initialization["fusion_bilinear_modules"], 31)
        self.assertEqual(sum(isinstance(m, BilinearUpsample) for m in model.backbone.modules()), 31)
        self.assertFalse(any(isinstance(m, torch.nn.Upsample) for m in model.backbone.modules()))
        bn = [m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        self.assertTrue(bn)
        self.assertTrue(all(not m.training and m.weight.requires_grad and m.bias.requires_grad for m in bn))

    def test_shapes_bn_buffer_identity_and_direct_gradient_reaches_backbone(self):
        model = self.model
        model.zero_grad(set_to_none=True)
        model.train()
        before = {name: value.clone() for name, value in model.backbone.named_buffers()}
        image = torch.randn(1, 3, 64, 64)
        stem, features = model.backbone(image)
        self.assertEqual(tuple(stem.shape), (1, 64, 32, 32))
        self.assertEqual([tuple(f.shape) for f in features],
                         [(1, 48, 16, 16), (1, 96, 8, 8), (1, 192, 4, 4), (1, 384, 2, 2)])
        output = model.decoder(image, stem, features)
        objective(output).backward()
        self.assertGreater(model.backbone.net.conv1.weight.grad.abs().sum().item(), 0)
        for name, value in model.backbone.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
        model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
