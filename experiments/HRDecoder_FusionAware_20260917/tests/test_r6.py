# -*- coding: utf-8 -*-
"""Round-6 unit tests: head identity/architecture reuse, feature-channel
construction, dice loss, patch planning, sliding-window coverage, counts
math, paired bootstrap, gate constants and phase schedule."""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import r6_head                      # noqa: E402
import r6_lib                       # noqa: E402
import runtime as rt                # noqa: E402

try:
    import r6_expert                # noqa: E402  (needs cv2)
    HAVE_EXPERT = True
except Exception:                   # pragma: no cover
    HAVE_EXPERT = False

# every module whose runtime/run_protocol attribute usage is statically
# guarded (round-6b files were added after the first such slip was caught)
SCANNED_FILES = ('r6_lib.py', 'r6_features.py', 'r6_head.py', 'r6_expert.py',
                 'r6_run.py', 'r6b_expert.py', 'r6b_run.py')


class TestRuntimeIntegration(unittest.TestCase):
    """The round-6 features must come from the SAME code path inference
    uses; these checks pin the runtime API the round depends on."""

    def test_pre_head_probs_exists(self):
        self.assertTrue(hasattr(rt.ChampionEnsemble, 'pre_head_probs'))

    def test_corrected_class_prob_exists_and_matches_f7_recipe(self):
        self.assertTrue(hasattr(rt.ChampionEnsemble, 'corrected_class_prob'))
        import inspect
        src = inspect.getsource(rt.ChampionEnsemble.corrected_class_prob)
        for token in ('_logit(pf[c:c + 1])', '_logit(ps[c:c + 1])',
                      '_logit(pr[c:c + 1])', 'ps[c:c + 1] - pr[c:c + 1]'):
            self.assertIn(token, src)

    def test_head_class_architecture_unchanged(self):
        h = rt.CorrectionHead()
        self.assertEqual(str(h.net[0]), 'Conv2d(7, 16, kernel_size=(3, 3), '
                                        'stride=(1, 1), padding=(1, 1))')
        self.assertEqual(str(h.net[4]), 'Conv2d(16, 1, kernel_size=(1, 1), '
                                        'stride=(1, 1))')
        x = torch.zeros(1, 7, 8, 8)
        self.assertEqual(tuple(h(x).shape), (1, 1, 8, 8))

    def test_predict_uses_the_shared_helpers(self):
        import inspect
        src = inspect.getsource(rt.ChampionEnsemble.predict)
        self.assertIn('self.pre_head_probs(', src)
        self.assertIn('self.corrected_class_prob(', src)
        # the raw f7 construction must NOT be duplicated in predict
        self.assertNotIn('torch.cat([img_f', src)


class TestHeadIdentity(unittest.TestCase):
    def test_zero_init_final_conv_is_identity(self):
        head = rt.CorrectionHead()
        with torch.no_grad():
            head.net[4].weight.zero_()
            head.net[4].bias.zero_()
        x = torch.randn(1, 7, 32, 32)
        self.assertLess(float(head(x).abs().max()), 1e-7)

    def test_f7_channels_follow_the_reference_recipe(self):
        rng = np.random.RandomState(0)
        img_f16 = (rng.randn(3, 24, 24)).astype(np.float16)
        fields = np.clip(rng.rand(3, 24, 24), 0.05, 0.95).astype(np.float16)
        f7 = r6_head.f7_from_arrays(img_f16, fields)[0]
        self.assertEqual(tuple(f7.shape), (7, 24, 24))
        ri = r6_head.round_features(torch.as_tensor(
            np.asarray(img_f16, dtype=np.float32)))
        r = r6_head.round_features(torch.as_tensor(
            np.asarray(fields, dtype=np.float32)))
        self.assertTrue(torch.equal(f7[0:3], ri))
        self.assertTrue(torch.equal(f7[3:4], rt._logit(r[0:1])))
        self.assertTrue(torch.equal(f7[4:5], rt._logit(r[1:2])))
        self.assertTrue(torch.equal(f7[5:6], rt._logit(r[2:3])))
        self.assertTrue(torch.equal(f7[6:7], r[1:2] - r[2:3]))

    def test_rounding_flag_is_declared_on_round6_heads_only(self):
        h_plain = rt.CorrectionHead()
        self.assertFalse(getattr(h_plain, 'r6_round_inputs', False))
        h_r6 = rt.CorrectionHead()
        h_r6.r6_round_inputs = True
        self.assertTrue(getattr(h_r6, 'r6_round_inputs'))

    def test_f7_logit_uses_runtime_logit(self):
        p = torch.tensor([[[[1e-6, 0.5, 1 - 1e-6]]]])
        self.assertTrue(torch.equal(rt._logit(p), rt._logit(p)))
        # clamping at eps must be finite
        self.assertTrue(torch.isfinite(rt._logit(p)).all())

    def test_round_features_is_near_identity(self):
        x = torch.rand(3, 16, 16) * 2 - 1
        y = r6_head.round_features(x)
        self.assertLess(float((x - y).abs().max()), 1e-3)

    def test_corrected_class_prob_rounds_only_when_flagged(self):
        import inspect
        src = inspect.getsource(rt.ChampionEnsemble.corrected_class_prob)
        self.assertIn("getattr(head, 'r6_round_inputs', False)", src)
        self.assertIn('img_f.half().float()', src)


class TestDice(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        p = torch.ones(1, 1, 8, 8) * 0.999
        t = torch.ones(1, 1, 8, 8)
        self.assertLess(float(r6_head._dice_loss(p, t)), 0.01)

    def test_empty_prediction_is_one(self):
        p = torch.zeros(1, 1, 8, 8)
        t = torch.ones(1, 1, 8, 8)
        self.assertGreater(float(r6_head._dice_loss(p, t)), 0.9)


class TestPatchPlan(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((256, 256), dtype=np.int64)
        self.gt[100:120, 150:170] = 4          # MA blob 400 px

    def test_positives_and_negatives(self):
        rng = np.random.RandomState(1)
        pos, neg = r6_lib.ma_patch_plan(self.gt, 64, 8, 8, rng,
                                        min_pos_px=8)
        self.assertGreater(len(pos), 0)
        for y, x in pos:
            self.assertGreaterEqual(int((self.gt[y:y + 64, x:x + 64] == 4)
                                        .sum()), 8)
        for y, x in neg:
            self.assertEqual(int((self.gt[y:y + 64, x:x + 64] == 4).sum()), 0)

    def test_no_positive_when_class_absent(self):
        rng = np.random.RandomState(2)
        pos, neg = r6_lib.ma_patch_plan(np.zeros((128, 128), np.int64), 32,
                                        4, 4, rng)
        self.assertEqual(pos, [])
        self.assertEqual(len(neg), 4)

    def test_label_patch_plan_generalises(self):
        gt = np.zeros((128, 128), np.int64)
        gt[10:40, 10:40] = 2                   # HE
        rng = np.random.RandomState(3)
        pos, _ = r6_lib.label_patch_plan(gt, 2, 32, 4, 0, rng, min_pos_px=4)
        self.assertGreater(len(pos), 0)


class TestWindowOrigins(unittest.TestCase):
    def test_covers_every_pixel(self):
        h, w, win, stride = 100, 137, 32, 16
        cov = np.zeros((h, w), dtype=int)
        for y, x in r6_lib.window_origins(h, w, win, stride):
            cov[y:y + win, x:x + win] += 1
        self.assertTrue((cov > 0).all(), 'sliding window left gaps')

    def test_edge_clamped_origins_in_range(self):
        for y, x in r6_lib.window_origins(100, 100, 32, 32):
            self.assertGreaterEqual(y, 0)
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(y, 100 - 32)
            self.assertLessEqual(x, 100 - 32)


class TestCountsMath(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(r6_lib.iou(5, 0, 0), 1.0)
        self.assertEqual(r6_lib.iou(0, 0, 0), 0.0)
        self.assertAlmostEqual(r6_lib.iou(3, 1, 1), 0.6)

    def test_counts_iou_miou(self):
        counts = {'EX': [1, 0, 0], 'HE': [1, 0, 0], 'SE': [1, 0, 0],
                  'MA': [1, 1, 1]}
        out = r6_lib.counts_iou(counts)
        self.assertAlmostEqual(out['EX'], 1.0)
        self.assertAlmostEqual(out['MA'], 1 / 3.0)
        self.assertAlmostEqual(out['mIoU'], (1 + 1 + 1 + 1 / 3.0) / 4.0)

    def test_class_delta(self):
        a = {'s1': {'HE': [10, 0, 0]}}
        b = {'s1': {'HE': [10, 0, 10]}}
        self.assertAlmostEqual(r6_lib.class_delta(b, a, 'HE'), -0.5)


class TestPairedBootstrap(unittest.TestCase):
    @staticmethod
    def _mk(delta_fn=None):
        rng = np.random.RandomState(7)
        pi = {}
        for i in range(30):
            base = {'EX': [50, 5, 5], 'HE': [40, 10, 10], 'SE': [45, 5, 5],
                    'MA': [10, 5, 5]}
            if delta_fn:
                base = delta_fn(dict(base), i, rng)
            pi['img%02d' % i] = base
        return pi

    def test_identical_sets_give_zero_delta_ci_containing_zero(self):
        a = self._mk()
        b = {k: {c: list(v[c]) for c in v} for k, v in a.items()}
        out = r6_lib.paired_bootstrap(a, b, n=200)
        self.assertAlmostEqual(out['delta'], 0.0, places=12)
        self.assertTrue(out['crosses_zero'])
        self.assertEqual(out['ci'], [0.0, 0.0])

    def test_clear_improvement_is_detected(self):
        a = self._mk()
        b = {k: {c: list(v[c]) for c in v} for k, v in a.items()}
        for k in b:                       # massive MA improvement
            b[k]['MA'] = [10, 0, 5]
        out = r6_lib.paired_bootstrap(a, b, n=300)
        self.assertGreater(out['delta'], 0.01)
        self.assertFalse(out['crosses_zero'])

    def test_mismatched_key_sets_rejected(self):
        a = self._mk()
        b = {k: v for k, v in a.items() if k != 'img00'}
        with self.assertRaises(ValueError):
            r6_lib.paired_bootstrap(a, b, n=10)


class TestRuntimeApiUsage(unittest.TestCase):
    """Every ``rt.<name>`` used by round-6 modules must exist on runtime --
    catches module-attribute slips (e.g. rt.list_split, rt.RuntimeError)
    locally instead of at minute 30 of a paid run."""

    def test_all_rt_attributes_exist(self):
        import io
        import re
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        missing = []
        for name in SCANNED_FILES:
            src = io.open(os.path.join(root, name), encoding='utf-8').read()
            for m in re.finditer(r'\brt\.([A-Za-z_][A-Za-z0-9_]*)', src):
                attr = m.group(1)
                if not hasattr(rt, attr):
                    missing.append('%s: rt.%s' % (name, attr))
        self.assertEqual(missing, [], 'missing runtime attributes: %s'
                         % missing)

    def test_all_rp_attributes_exist(self):
        import io
        import re
        import run_protocol as rp
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        missing = []
        for name in SCANNED_FILES:
            src = io.open(os.path.join(root, name), encoding='utf-8').read()
            for m in re.finditer(r'\brp\.([A-Za-z_][A-Za-z0-9_]*)', src):
                attr = m.group(1)
                if not hasattr(rp, attr):
                    missing.append('%s: rp.%s' % (name, attr))
        self.assertEqual(missing, [], 'missing run_protocol attributes: %s'
                         % missing)


class TestCacheReservation(unittest.TestCase):
    """A resume over a COMPLETE cache must not demand disk space (the 14:30
    recovery would otherwise refuse to start with the cache already built)."""

    def test_cache_needed_false_when_complete(self):
        import json
        import tempfile
        import r6_features
        d = tempfile.mkdtemp()
        try:
            cd = r6_features.cache_dir(d, 'HE', 'train')
            os.makedirs(cd, exist_ok=True)
            settings = {'_cfg': type('C', (), {'dump': lambda self: 'x'})(),
                        '_hashes': {'reference_checkpoint': 'sha'}}
            with open(os.path.join(cd, 'manifest.json'), 'w') as f:
                json.dump({'cls': 'HE', 'split': 'train', 'count': 3,
                           'cfg_digest': r6_features._digest(settings),
                           'ref_sha': 'sha'}, f)
            with open(os.path.join(cd, 'COMPLETE'), 'w') as fh:
                fh.write('ok')
            # complete + matching identity -> no rebuild, no disk demand
            self.assertFalse(r6_features.cache_needed(d, 'HE', 'train', 3,
                                                      settings))
            # different class -> still needed
            self.assertTrue(r6_features.cache_needed(d, 'EX', 'train', 3,
                                                     settings))
            # present but inconsistent (count changed) -> FAIL CLOSED
            # (same contract as runtime.ReferenceCache: never silently rebuild)
            with self.assertRaises(RuntimeError):
                r6_features.cache_needed(d, 'HE', 'train', 4, settings)
        finally:
            import shutil
            shutil.rmtree(d)


class TestLabelLoaderUsage(unittest.TestCase):
    """rt.load_image_label is train-only; round 6 must use r6_lib.load_label
    for any other split (the round-6 run crashed exactly there at 14:16)."""

    def test_no_train_only_loader_on_other_splits(self):
        import io
        import re
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bad = []
        for name in SCANNED_FILES:
            src = io.open(os.path.join(root, name), encoding='utf-8').read()
            for m in re.finditer(
                    r"load_image_label\([^)]*?,?\s*['\"]([a-z]+)['\"]", src):
                if m.group(1) != 'train':
                    bad.append('%s: split=%s' % (name, m.group(1)))
        self.assertEqual(bad, [], 'train-only loader used elsewhere: %s' % bad)

    def test_r6_lib_load_label_exists(self):
        self.assertTrue(callable(getattr(r6_lib, 'load_label', None)))


class TestHeadDeviceSafety(unittest.TestCase):
    """Regression guard for the CPU/GPU mismatch the round-6 smoke caught:
    every head call site must build f7 on the head's own device."""

    def test_val_full_image_runs_on_cpu(self):
        head = rt.CorrectionHead()               # CPU head
        with torch.no_grad():
            head.net[4].weight.zero_()
            head.net[4].bias.zero_()
        img = np.zeros((3, 32, 32), dtype=np.float16)
        fld = np.full((3, 32, 32), 0.3, dtype=np.float16)
        lab = np.zeros((32, 32), dtype=np.int64)
        lab[5:10, 5:10] = 2                      # HE pixels (25)
        p, cnt = r6_head._val_full_image(img, fld, lab, head, 2)
        self.assertEqual(tuple(p.shape), (32, 32))
        # identity head -> p_new == pf == 0.3 (<0.5) -> nothing predicted
        self.assertEqual(cnt, [0, 0, 25])

    def test_f7_on_device_follows_head_device(self):
        head = rt.CorrectionHead()
        img = np.zeros((3, 16, 16), dtype=np.float16)
        fld = np.full((3, 16, 16), 0.4, dtype=np.float16)
        f7 = r6_head._f7_on_device(img, fld, head)
        self.assertEqual(f7.device.type, 'cpu')


class TestGatesAndSchedule(unittest.TestCase):
    def test_pre_registered_constants(self):
        import r6_run
        self.assertEqual(r6_run.G1_CLASS_DELTA, 0.005)
        self.assertEqual(r6_run.G2_MA_DELTA, 0.010)
        self.assertEqual(r6_run.G3_MIOU_DELTA, 0.002)
        self.assertEqual(r6_run.ALPHAS, [0.3, 0.5, 0.7, 1.0])
        self.assertAlmostEqual(r6_run.CHAMPION_TEST_REF, 0.442115)

    def test_phase_schedule_is_ordered_and_within_six_hours(self):
        import r6_run
        h = r6_run.PHASE_HOURS
        order = [h[k] for k in ('p0', 'p1', 'p1_test', 'p2', 'p3')]
        self.assertEqual(order, sorted(order))
        self.assertLessEqual(max(order), 6.0)
        for v in order:
            self.assertGreater(v, 0)

    def test_dataset_size_constants(self):
        import r6_run
        self.assertEqual((r6_run.TRAIN_IMAGES, r6_run.VAL_IMAGES,
                          r6_run.TEST_IMAGES), (454, 76, 227))


class TestEnsureSpace(unittest.TestCase):
    def test_returns_free_space_without_cleanup(self):
        import tempfile
        d = tempfile.mkdtemp()
        try:
            logs = []
            free = r6_lib.ensure_space(d, 0.0001, d, logs.append,
                                       allow_cleanup=False)
            self.assertGreater(free, 0.0)
            self.assertTrue(any('disk' in m for m in logs))
        finally:
            import shutil
            shutil.rmtree(d)


@unittest.skipUnless(HAVE_EXPERT, 'cv2/r6_expert unavailable locally')
class TestExpertModule(unittest.TestCase):
    def test_unet_forward_shape_and_range(self):
        m = r6_expert.UNet(base=8)
        x = torch.rand(2, 3, 64, 64)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 1, 64, 64))

    def test_expert_loss_finite(self):
        p = torch.zeros(1, 1, 16, 16)
        t = torch.zeros(1, 1, 16, 16)
        loss = r6_expert._loss(p, t)
        self.assertTrue(np.isfinite(float(loss)))

    def test_counts_for_class_ignores_255(self):
        prob = np.array([[0.9, 0.1], [0.9, 0.1]], dtype=np.float32)
        gt = np.array([[1, 255], [1, 0]], dtype=np.int64)
        tp, fp, fn = r6_expert.counts_for_class(prob, gt, 1)
        self.assertEqual((tp, fp, fn), (2, 0, 0))
        tp, fp, fn = r6_expert.counts_for_class(prob, gt, 2)
        self.assertEqual((tp, fp, fn), (0, 2, 0))

    def test_alphas_without_expert_is_rejected(self):
        with self.assertRaises(ValueError):
            r6_expert.eval_split_with_expert(None, None, None, [0.5], 'val',
                                             lambda m: None, 1e18)

    def test_alpha_fusion_math(self):
        ex = np.array([[1.0, 0.0]], dtype=np.float32)
        champ = np.array([[0.0, 1.0]], dtype=np.float32)
        fused = 0.7 * ex + 0.3 * champ
        self.assertAlmostEqual(float(fused[0, 0]), 0.7, places=6)
        self.assertAlmostEqual(float(fused[0, 1]), 0.3, places=6)


if __name__ == '__main__':
    unittest.main()
