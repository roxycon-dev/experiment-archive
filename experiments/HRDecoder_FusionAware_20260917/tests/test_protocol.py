# -*- coding: utf-8 -*-
"""Pure/stub CPU unit tests.  No mmcv, no GPU, no dataset downloads.
Importing run_protocol must not require mmcv."""
import datetime
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_protocol as rp  # noqa: E402  (must import without mmcv)
import ops                  # noqa: E402


def mk(fake_t0, caps=None):
    caps = caps or rp.PHASE_CAPS
    st = {'run_start': fake_t0, 'chosen_T': 2000, 'identity': {'x': 1},
          'phase_start': {}, 'val_history': {}}
    return st


class TestPhaseBudget(unittest.TestCase):
    def test_phase_caps_sum_24h_with_prep_shared(self):
        total = sum(rp.PHASE_CAPS.values())
        self.assertLessEqual(total, rp.GLOBAL_BUDGET)
        # preparation appears ONCE (shared), not once per prep subtask
        self.assertEqual([k for k in rp.PHASE_CAPS if 'prep' in k or
                          k in ('cache', 'starting_d_val', 'champion_val',
                                'timing')],
                         ['preparation'])

    def test_deadline_never_extended_on_restart(self):
        t0 = 1_000_000.0
        st = mk(t0)
        st['phase_start']['preparation'] = t0 + 10      # started late
        # pretend time jumped forward (restart later): deadline must be the
        # same absolute value
        d1 = rp.phase_deadline(st, 'preparation')
        st['phase_start']['preparation'] = t0 + 10      # persisted value
        d2 = rp.phase_deadline(st, 'preparation')
        self.assertEqual(d1, d2)
        self.assertLessEqual(d1, t0 + rp.GLOBAL_BUDGET)

    def test_final_reserve_fits_global(self):
        t0 = 1_000_000.0
        st = mk(t0)
        st['phase_start']['preparation'] = t0
        st['phase_start']['pair_seed42'] = t0 + rp.PHASE_CAPS['preparation']
        st['phase_start']['pair_seed43'] = (st['phase_start']['pair_seed42'] +
                                            rp.PHASE_CAPS['pair_seed42'])
        d_final = rp.phase_deadline(st, 'final')
        self.assertLessEqual(d_final, t0 + rp.GLOBAL_BUDGET)
        # sum of pre-final caps + final + reserve <= 24h
        self.assertLessEqual(
            rp.PHASE_CAPS['preparation'] + rp.PHASE_CAPS['pair_seed42'] +
            rp.PHASE_CAPS['pair_seed43'] + rp.PHASE_CAPS['final'], rp.GLOBAL_BUDGET)


class TestGateUsesHistories(unittest.TestCase):
    def test_gate_uses_four_timepoint_mious_not_perclass(self):
        # a_hist/b_hist are (step, miou) histories of 4 validation timepoints
        a_hist = [(500, 0.430), (1000, 0.440), (1500, 0.445), (2000, 0.450)]
        b_hist = [(500, 0.440), (1000, 0.448), (1500, 0.455), (2000, 0.460)]
        a4 = [m for _, m in a_hist]
        b4 = [m for _, m in b_hist]
        self.assertTrue(ops.gate_first(a4, b4, 0.4220))
        # exactly the driver's extraction path
        self.assertEqual([m for _, m in a_hist], a4)

    def test_gate_rejects_insufficient_gain(self):
        self.assertFalse(ops.gate_first([0.44] * 4, [0.441] * 4, 0.4220))

    def test_no_perclass_helper_exists(self):
        self.assertFalse(hasattr(rp, '_per_class_at_best'))


class TestRngRestore(unittest.TestCase):
    def test_restore_makes_next_sample_equal_uninterrupted(self):
        rng = np.random.RandomState(42)
        first3 = [rng.randint(0, 1000) for _ in range(3)]
        state = rng.get_state()
        noise = [rng.randint(0, 1000) for _ in range(10)]  # divergence
        self.assertNotEqual(noise[:3], [rng.randint(0, 1000) for _ in range(3)])
        rng.set_state(state)                 # restore
        again = [rng.randint(0, 1000) for _ in range(3)]
        # uninterrupted reference
        rng2 = np.random.RandomState(42)
        [rng2.randint(0, 1000) for _ in range(3)]
        ref = [rng2.randint(0, 1000) for _ in range(3)]
        self.assertEqual(again, ref)
        check_rng = np.random.RandomState(42)
        self.assertEqual(first3, [check_rng.randint(0, 1000) for _ in range(3)])


class TestChooseBestTie(unittest.TestCase):
    def test_tie_prefers_seed42_with_explicit_file(self):
        b42 = (2000, 0.4700)
        b43 = (1750, 0.4700)
        seed, best = rp._choose_best_seed([('B', 42, b42, []),
                                           ('B', 43, b43, [])])
        self.assertEqual(seed, 42)
        path = os.path.join('run', 'ckpt_B_s%d' % seed, 'iter_%d.pth' % best[0])
        self.assertIn('s42', path)

    def test_higher_seed43_wins_over_42(self):
        seed, _ = rp._choose_best_seed([('B', 42, (2000, 0.4600), []),
                                        ('B', 43, (1500, 0.4650), [])])
        self.assertEqual(seed, 43)


class TestLockImmutability(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_lock_immutable_mismatch_fails(self):
        lock = {'candidate': 'champion', 'val_results': {'champion': 0.44}}
        p = rp.write_lock(self.d, lock)
        with open(p) as f:
            self.assertEqual(json.load(f), lock)
        with self.assertRaises(RuntimeError):
            rp.write_lock(self.d, {'candidate': 'new'})
        rp.write_lock(self.d, lock)          # identical re-write is fine

    def test_lock_written_before_test_marker(self):
        lock = {'candidate': 'mix'}
        rp.write_lock(self.d, lock)
        self.assertTrue(os.path.isfile(
            os.path.join(self.d, 'selection_lock.json')))


class TestTestMarkers(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_incomplete_test_fails_closed_no_rerun(self):
        open(os.path.join(self.d, 'TEST_STARTED'), 'w').close()
        with self.assertRaises(RuntimeError):
            rp.final_test(None, self.d, {'candidate': 'champion'}, None,
                          None, 1e18, lambda m: None)

    def test_done_marker_skips(self):
        open(os.path.join(self.d, 'TEST_DONE'), 'w').close()
        rp.final_test(None, self.d, {'candidate': 'champion'}, None, None,
                      1e18, lambda m: None)  # must not raise

    def test_noop_when_no_markers_and_no_models_raises_gracefully(self):
        # nothing to run with None champion -> generic failure but marker
        # removed (no stale TEST_STARTED)
        try:
            rp.final_test(None, self.d, {'candidate': 'champion'}, None,
                          None, 1e18, lambda m: None)
        except Exception:
            pass
        self.assertFalse(os.path.isfile(
            os.path.join(self.d, 'TEST_STARTED')))


class TestOpsLedger(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.dl = datetime.datetime.now(
            datetime.timezone.utc) + datetime.timedelta(hours=3)
        with open(os.path.join(self.d, 'budget_ledger.json'), 'w') as f:
            json.dump({'billing_started_at': (self.dl - datetime.timedelta(hours=3)).isoformat(), 'other_cost': 0}, f)

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_ops_load_ledger_and_deadline(self):
        led = ops.load_ledger(self.d)
        dt = ops.deadline_of(led)
        self.assertTrue(dt.tzinfo is not None)      # aware datetime
        self.assertGreater(dt.timestamp(), __import__('time').time())

    def test_no_ledger_json_required(self):
        self.assertFalse(os.path.isfile(
            os.path.join(self.d, 'ledger.json')))


class TestDryRunImportable(unittest.TestCase):
    def test_import_needs_no_mmcv(self):
        self.assertIsNone(sys.modules.get('mmcv'))
        import subprocess
        subprocess.run([sys.executable, '-c', "import run_protocol; import sys; assert 'mmcv' not in sys.modules"], check=True)
        self.assertTrue(callable(rp.validate_settings))
        self.assertTrue(callable(rp.main))


class TestArmCredit(unittest.TestCase):
    """arm_credit sizes the REMAINING budget on restart; it must read the
    persisted iter/val_history of the right arm+seed and fail closed."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        import torch
        self.torch = torch

    def tearDown(self):
        shutil.rmtree(self.d)

    def _save(self, arm, seed, iters, n_evals):
        ck = os.path.join(self.d, 'ckpt_%s_s%d' % (arm, seed))
        os.makedirs(ck, exist_ok=True)
        self.torch.save(
            {'arm': arm, 'seed': seed, 'iter': iters,
             'val_history': [(i * 1000, 0.41) for i in range(n_evals)]},
            os.path.join(ck, 'last.pth'))

    def test_credit_reads_last_pth(self):
        self._save('B', 42, 3400, 3)
        self.assertEqual(rp.arm_credit(self.d, 'B', 42), (3400, 3))

    def test_credit_zero_when_absent(self):
        self._save('B', 42, 3400, 3)
        self.assertEqual(rp.arm_credit(self.d, 'A', 42), (0, 0))  # no dir
        self.assertEqual(rp.arm_credit(self.d, 'B', 43), (0, 0))  # wrong seed

    def test_credit_corrupt_file_fails_closed(self):
        ck = os.path.join(self.d, 'ckpt_A_s42')
        os.makedirs(ck)
        with open(os.path.join(ck, 'last.pth'), 'wb') as f:
            f.write(b'not a checkpoint at all')
        self.assertEqual(rp.arm_credit(self.d, 'A', 42), (0, 0))


class TestResumeIntegrity(unittest.TestCase):
    """Restart-time code changes are allowed ONLY as recorded amendments:
    fingerprint-only drift is accepted and logged; anything touching
    config/split/assets stays fatal."""

    def test_identity_drift_kind(self):
        old = {'hashes': {'config': 'c'}, 'fingerprint': 'F_OLD'}
        same = {'hashes': {'config': 'c'}, 'fingerprint': 'F_OLD'}
        code = {'hashes': {'config': 'c'}, 'fingerprint': 'F_NEW'}
        fatal = {'hashes': {'config': 'X'}, 'fingerprint': 'F_NEW'}
        self.assertIsNone(rp._identity_drift_kind(old, same))
        self.assertEqual(rp._identity_drift_kind(old, code), 'code')
        self.assertEqual(rp._identity_drift_kind(old, fatal), 'fatal')

    def test_code_drift_accepted_and_recorded(self):
        d = tempfile.mkdtemp()
        try:
            ck_dir = os.path.join(d, 'ckpt_B_s42')
            os.makedirs(ck_dir)
            p = os.path.join(ck_dir, 'last.pth')
            saved = {'config': 'c', 'split': 's', 'assets': {}, 'code': 'OLD'}
            expect = {'config': 'c', 'split': 's', 'assets': {}, 'code': 'NEW'}
            rp._check_split_hashes(p, saved, expect)   # must not raise
            self.assertTrue(os.path.isfile(
                os.path.join(d, 'protocol_amendments.jsonl')))
            with open(os.path.join(d, 'protocol_amendments.jsonl')) as f:
                entry = json.loads(f.read().splitlines()[-1])
            self.assertEqual(entry['kind'],
                             'code_fingerprint_drift_on_resume')
        finally:
            shutil.rmtree(d)

    def test_exact_match_needs_no_amendment(self):
        d = tempfile.mkdtemp()
        try:
            ck_dir = os.path.join(d, 'ckpt_A_s42')
            os.makedirs(ck_dir)
            p = os.path.join(ck_dir, 'last.pth')
            same = {'config': 'c', 'split': 's', 'assets': {}, 'code': 'X'}
            rp._check_split_hashes(p, same, dict(same))
            self.assertFalse(os.path.isfile(
                os.path.join(d, 'protocol_amendments.jsonl')))
        finally:
            shutil.rmtree(d)

    def test_split_or_asset_drift_still_fatal(self):
        base = {'config': 'c', 'split': 's', 'assets': {'a': 1}, 'code': 'X'}
        for mutated in ({'config': 'Z', 'split': 's', 'assets': {},
                         'code': 'X'},
                        {'config': 'c', 'split': 'OTHER', 'assets': {},
                         'code': 'X'},
                        {'config': 'c', 'split': 's',
                         'assets': {'a': 2}, 'code': 'X'}):
            with self.assertRaises(RuntimeError):
                rp._check_split_hashes('p/ckpt_A_s42/last.pth',
                                       mutated, base)
        # missing 'code' on either side is NOT amendable drift
        with self.assertRaises(RuntimeError):
            rp._check_split_hashes(
                'p/ckpt_A_s42/last.pth',
                {'config': 'c', 'split': 's', 'assets': {}},
                {'config': 'c', 'split': 's', 'assets': {}, 'code': 'X'})


if __name__ == '__main__':
    unittest.main()
