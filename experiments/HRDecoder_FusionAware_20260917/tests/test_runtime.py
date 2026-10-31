# tests/test_runtime.py
# -*- coding: utf-8 -*-
"""Runtime unit tests.  No mmcv: dataset builders are injected, members are
fake models, cache fixtures are tiny."""
import hashlib
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

import runtime


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


class TestCounts(unittest.TestCase):
    def test_resolve_repo_path(self):
        # relative -> repo-joined; absolute -> untouched (guard runs with
        # CWD=glm, so raw relative paths must never reach torch.load)
        self.assertEqual(runtime.resolve_repo_path('/r', 'weights/a.pth'),
                         os.path.join('/r', 'weights/a.pth'))
        self.assertEqual(runtime.resolve_repo_path('/r', '/abs/a.pth'),
                         '/abs/a.pth')

    def test_per_image_counts_exact(self):
        prob = np.zeros((4, 2, 2), np.float32)
        prob[0, 0, 0] = 0.6   # EX tp
        prob[1, 1, 1] = 0.4   # HE nothing (strict >0.5)
        gt = np.zeros((2, 2), np.int64)
        gt[0, 0] = 1
        gt[1, 0] = 255        # ignore region, EX fp there ignored
        prob[0, 1, 0] = 0.9
        cnt = runtime.per_image_counts(prob, gt)
        self.assertEqual(cnt[0], [1, 0, 0])
        self.assertEqual(cnt[1], [0, 0, 0])

    def test_global_miou_and_union_zero(self):
        with self.assertRaises(RuntimeError):
            runtime.global_miou([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
        miou, per = runtime.global_miou([(2, 0, 2), (0, 1, 0)][:2] +
                                        [(1, 1, 1), (4, 0, 0)])
        self.assertAlmostEqual(miou, np.mean([0.5, 0.0, 1.0 / 3.0, 1.0]))
        self.assertEqual(set(per), {'EX', 'HE', 'SE', 'MA'})


class TestCorrectionHead(unittest.TestCase):
    def test_flat_head_load_and_shape(self):
        h = runtime.CorrectionHead()
        flat = {k: v.clone() for k, v in h.state_dict().items()}
        h2 = runtime.load_flat_head(flat)
        self.assertEqual(sorted(h2.state_dict()), sorted(flat))
        f = torch.randn(1, 7, 8, 8)
        out = h2(f)
        self.assertEqual(out.shape, (1, 1, 8, 8))
        self.assertTrue((out.abs() <= 6.0 + 1e-5).all())

    def test_flat_head_rejects_bad_state(self):
        with self.assertRaises(RuntimeError):
            runtime.load_flat_head({'something': torch.zeros(1)})
        with self.assertRaises(RuntimeError):
            runtime.load_flat_head('not-a-dict')


class FakeMember(object):
    """Records calls; simulates mmseg flip-undo: returns the identity prob
    regardless of input flip (output already un-flipped).

    fork=True reproduces the REAL server structure verified against GLM's
    eval_ens.py / h5_ensemble_corrector.py: simple_test (compute_aupr=True
    path) returns [(prob(4,H,W) ndarray, use_sigmoid, compute_aupr)] --
    a list wrapping a (array, bool, bool) tuple."""
    def __init__(self, val, fork=True):
        self.val = val
        self.fork = fork
        self.last_flip = None

    def __call__(self, return_loss, rescale, img, img_metas):
        meta = img_metas[0]
        if not isinstance(meta, dict):
            meta = meta[0]
        self.last_flip = bool(meta.get('flip', False))
        prob = np.full((4, 4, 4), self.val, np.float32)
        if self.fork:
            return [(prob, True, True)]      # real structure
        return [prob]                        # legacy/plain structure


class TestMemberProbNoDoubleFlip(unittest.TestCase):
    def _check(self, fork):
        m = FakeMember(0.5, fork=fork)
        img = torch.rand(1, 3, 4, 4)
        meta = [{'ori_shape': (4, 4)}]
        p = runtime.member_prob(m, img, meta, flip=True, device='cpu')
        self.assertTrue(m.last_flip)                 # flag was set
        self.assertEqual(p.shape, (4, 4, 4))
        self.assertTrue(np.allclose(p.numpy(), 0.5))  # NOT flipped again
        p2 = runtime.member_prob(m, img, meta, flip=False, device='cpu')
        self.assertFalse(m.last_flip)
        self.assertTrue(np.allclose(p2.numpy(), 0.5))

    def test_flip_undo_once_fork_structure(self):
        # real server output: [(prob, use_sigmoid, compute_aupr)]
        self._check(fork=True)

    def test_flip_undo_once_plain_structure(self):
        # backward-compatible: [prob]
        self._check(fork=False)

    def test_member_prob_rejects_bad_shape(self):
        class BadMember(object):
            def __call__(self, return_loss, rescale, img, img_metas):
                return [(np.zeros((2, 4, 4), np.float32), True, True)]
        m = BadMember()
        img = torch.rand(1, 3, 4, 4)
        meta = [{'ori_shape': (4, 4)}]
        with self.assertRaises(RuntimeError):
            runtime.member_prob(m, img, meta, flip=False, device='cpu')


class TestReferenceCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # v3 contract: float16 on disk at the fixed 2048 test-time resolution
        self.prob = np.full((2048, 2048, 4), 0.25, np.float16)
        self.npy = os.path.join(self.tmp, 'img1.npy')
        np.save(self.npy, self.prob)
        self.manifest = {'version': runtime.ReferenceCache.VERSION,
                         'split': 'train', 'image_ids': ['img1'],
                         'dtype': runtime.ReferenceCache.DTYPE,
                         'resolution': list(runtime.ReferenceCache.RESOLUTION),
                         'source_images_sha256': 'X',
                         'reference_weight_sha256': 'R',
                         'config_sha256': 'C',
                         'normalization': {
                             'mean': runtime.IMG_MEAN.tolist(),
                             'std': runtime.IMG_STD.tolist()},
                         'entries': {'img1': {'npy': self.npy,
                                              'shape': [2048, 2048, 4],
                                              'sha256': _sha(self.npy)}}}

    def _settings(self):
        s = {'data_root': self.tmp, '_cfg': None, '_hashes':
             {'reference_checkpoint': 'R'}}
        return s

    def test_missing_cache_returns_false(self):
        c = runtime.ReferenceCache(os.path.join(self.tmp, 'none'), {})
        self.assertFalse(c.complete())

    def test_get_validates(self):
        import types
        c = runtime.ReferenceCache(self.tmp, self._settings())
        # monkeypatch identity material for fixture (tiny test setup)
        c._train_image_ids = lambda: ['img1']
        c._source_images_digest = lambda: 'X'
        c._cfg_digest = lambda: 'C'
        json.dump(self.manifest,
                  open(c.manifest_path, 'w', encoding='utf-8'))
        open(c.complete_path, 'w').write('ok')
        arr = c.get('img1')
        self.assertTrue(np.allclose(arr, self.prob))
        # corrupt file bytes -> hash mismatch on get
        with open(self.npy, 'ab') as f:
            f.write(b'\x00')
        with self.assertRaises(RuntimeError):
            c.get('img1')

    def test_corruption_raises(self):
        c = runtime.ReferenceCache(self.tmp, self._settings())
        c._train_image_ids = lambda: ['img1']
        c._source_images_digest = lambda: 'X'
        c._cfg_digest = lambda: 'C'
        bad = json.loads(json.dumps(self.manifest))
        bad['entries']['img1']['sha256'] = 'deadbeef'
        json.dump(bad, open(c.manifest_path, 'w', encoding='utf-8'))
        open(c.complete_path, 'w').write('ok')
        with self.assertRaises(RuntimeError):
            c.complete()
        # wrong version raises too
        bad = json.loads(json.dumps(self.manifest))
        bad['version'] = 'nope'
        json.dump(bad, open(c.manifest_path, 'w', encoding='utf-8'))
        with self.assertRaises(RuntimeError):
            c.complete()


class FakeDS(object):
    def __init__(self, n=2, size=(2, 2)):
        self.n = n
        self.size = size
        self.img_infos = [{'filename': 'v%d.jpg' % i} for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        h, w = self.size
        return {'img': torch.rand(3, h, w),
                'img_metas': {'ori_shape': (h, w),
                              'ori_filename': 'v%d.jpg' % i}}

    def get_gt_seg_maps(self):
        return [np.full(self.size, 255, np.int64) for _ in range(self.n)]


class TestEvaluateSplit(unittest.TestCase):
    def test_predictor_alignment_nonMA_exact(self):
        h, w = 2, 3
        gt = np.array([[1, 2, 3], [4, 0, 255]], dtype=np.int64)
        prob = np.full((1, 4, h, w), 0.1, np.float32)
        for c in range(4):
            prob[0, c][gt == c+1] = 0.9
        prob[0, 2] = 0.9  # SE predicts every pixel, one ignored

        ds = FakeDS(n=1, size=(h, w))
        ds._gt = gt
        ds.get_gt_seg_maps = lambda: [gt]

        def builder(root, split):
            return ds

        def predictor(img, meta):
            self.assertEqual(tuple(img.shape), (1, 3, h, w))
            return torch.from_numpy(prob)      # (1,4,H,W) batch accepted

        log = lambda m: None
        miou, per, per_image = runtime.evaluate_split(
            predictor, 'val', 'x', None, float('inf'), log,
            expected_count=1, _ds_builder=builder)
        # SE: tp=1, fp=3, fn=0 (ignored pixel not fp)
        self.assertEqual(per_image['v0']['SE'], [1, 4, 0])
        for c in ('EX', 'HE', 'MA'):
            self.assertEqual(per_image['v0'][c], [1, 0, 0])
        self.assertAlmostEqual(miou, np.mean([1.0, 1.0, 0.2, 1.0]))

    def test_predictor_shape_strict(self):
        ds = FakeDS()

        def builder(root, split):
            return ds

        def predictor(img, meta):
            return torch.zeros(2, 4, 2, 2)     # batch 2 -> reject

        with self.assertRaises(RuntimeError):
            runtime.evaluate_split(predictor, 'val', 'x', None,
                                   float('inf'), lambda m: None,
                                   expected_count=1, _ds_builder=builder)


class TestFirstUpdateEvidence(unittest.TestCase):
    def _model(self):
        class Inner(torch.nn.Module):
            def __init__(self):
                super(Inner, self).__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.decode_head = Inner()
                self.backbone = torch.nn.Module()
                self.backbone.stage4 = Inner()
                self.reference = Inner()
                self.other = Inner()

        return M()

    def test_evidence(self):
        m = self._model()
        ps = runtime.snapshot_state(m)
        bs = runtime.snapshot_buffers(m)
        for n, p in m.named_parameters():
            p.grad = torch.full_like(p, 0.1)
        with torch.no_grad():
            m.decode_head.conv.weight.add_(0.01)
            m.backbone.stage4.conv.weight.add_(0.01)
        ev = runtime.first_update_evidence(m, ps, bs)
        self.assertTrue(all(ev.values()))

    def test_frozen_changed_fails(self):
        m = self._model()
        ps = runtime.snapshot_state(m)
        bs = runtime.snapshot_buffers(m)
        for n, p in m.named_parameters():
            p.grad = torch.full_like(p, 0.1)
        with torch.no_grad():
            m.decode_head.conv.weight.add_(0.01)
            m.backbone.stage4.conv.weight.add_(0.01)
            m.reference.conv.weight.add_(0.01)   # forbidden
        ev = runtime.first_update_evidence(m, ps, bs)
        self.assertFalse(ev['frozen_params_unchanged'])
        self.assertFalse(all(ev.values()))


class TestAuditSplit(unittest.TestCase):
    def test_audit_tiny(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for s in ('train', 'val', 'test'):
            os.makedirs(os.path.join(tmp, 'images', s))
            os.makedirs(os.path.join(tmp, 'labels', s))
        files = {'train': ['a.jpg'], 'val': ['b.jpg'], 'test': ['c.jpg']}
        for s, fs in files.items():
            for f in fs:
                open(os.path.join(tmp, 'images', s, f), 'w').write('x')
            with open(os.path.join(tmp, s + '.txt'), 'w') as fh:
                fh.write('\n'.join(fs))
            if True:
                open(os.path.join(tmp, 'labels', s,
                                  os.path.splitext(f)[0] + '.png'),
                     'w').write('x')
        a = runtime.audit_split(tmp, expected=(1, 1, 1))
        self.assertEqual(a['a.jpg']['split'], 'train')
        self.assertEqual(a['c.jpg']['split'], 'test')
        # label mismatch raises
        os.remove(os.path.join(tmp, 'labels', 'train', 'a.png'))
        with self.assertRaises(RuntimeError):
            runtime.audit_split(tmp, expected=(1, 1, 1))


class TestMisc(unittest.TestCase):
    def test_parse_deadline(self):
        self.assertEqual(runtime.parse_deadline(123.5), 123.5)
        t = runtime.parse_deadline('2026-01-01T00:00:00Z')
        self.assertIsInstance(t, float)
        with self.assertRaises(ValueError):
            runtime.parse_deadline('tomorrow')

    def test_mix_probs(self):
        a = torch.full((4, 2, 2), 0.8)
        b = torch.full((4, 2, 2), 0.2)
        self.assertTrue(torch.allclose(runtime.mix_probs(a, b),
                                       torch.full((4, 2, 2), 0.5)))


if __name__ == '__main__':
    unittest.main()
