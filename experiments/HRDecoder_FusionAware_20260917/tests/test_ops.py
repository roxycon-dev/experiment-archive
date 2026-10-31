"""ops tests: budget, watchdog, steps, gates, checkpoints, metrics. unittest."""
import json, os, sys, tempfile, unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ops


def _led(start, cost):
    return {"billing_started_at": start.isoformat(), "other_cost": cost}


class TestBudget(unittest.TestCase):
    def test_deadline_min_of_cap_and_budget(self):
        start = datetime(2026, 9, 16, tzinfo=timezone.utc)
        self.assertEqual(ops.deadline_of(_led(start, 0.0)),
                         start + timedelta(hours=ops.CAP_HOURS))
        self.assertEqual(ops.deadline_of(_led(start, 40.0)),
                         start + timedelta(hours=10 / 1.98))

    def test_invalid_budget_rejected(self):
        start = datetime(2026, 9, 16, tzinfo=timezone.utc)
        for bad in (float("nan"), float("inf"), -1.0, 50.0, "x"):
            with self.assertRaises((ValueError, TypeError)):
                ops.deadline_of(_led(start, bad))

    def test_init_persists_and_rejects_naive(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = os.path.join(tmp, "run")
            ops.main(["init", "--started-at", "2026-09-16T00:00:00+00:00",
                      "--other-cost", "5.5", "--run-dir", rd])
            led = ops.load_ledger(rd)
            self.assertTrue(led["billing_started_at"]
                            .startswith("2026-09-16T00:00:00+00:00"))
            with self.assertRaises(ValueError):
                ops.main(["init", "--started-at", "2026-09-16T00:00:00",
                          "--other-cost", "0", "--run-dir", rd])

    def test_restart_never_resets_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = os.path.join(tmp, "run")
            ops.main(["init", "--started-at", "2026-09-16T00:00:00+00:00",
                      "--other-cost", "1.0", "--run-dir", rd])
            with self.assertRaises(ValueError):
                ops.main(["init", "--started-at", "2030-01-01T00:00:00+00:00",
                          "--other-cost", "2.0", "--run-dir", rd])
            ops.main(["init", "--started-at", "2026-09-16T00:00:00+00:00",
                      "--other-cost", "2.0", "--run-dir", rd])
            led = ops.load_ledger(rd)
            self.assertTrue(led["billing_started_at"].startswith("2026-09-16"))
            self.assertEqual(led["other_cost"], 2.0)


class TestWatchdog(unittest.TestCase):
    def test_dry_run_deadline_expired_no_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = os.path.join(tmp, "run")
            ops.main(["init", "--started-at", "2020-01-01T00:00:00+00:00",
                      "--other-cost", "0", "--run-dir", rd])
            ops.main(["watchdog", "--run-dir", rd, "--dry-run"])

    def test_windows_real_shutdown_forbidden(self):
        with mock.patch.object(ops.sys, "platform", "win32"):
            with self.assertRaises(RuntimeError):
                ops.shutdown_system(False)


class TestChooseSteps(unittest.TestCase):
    def test_feasible_and_multiple_of_four(self):
        self.assertEqual(ops.choose_steps(0.001, 100.0), 4000)
        t = ops.choose_steps(0.002, 50.0)
        self.assertTrue(1000 <= t <= 4000 and t % 4 == 0)

    def test_infeasible_raises(self):
        with self.assertRaises(ValueError):
            ops.choose_steps(20.0, 100.0)


class TestGates(unittest.TestCase):
    def test_gate_four_timepoints(self):
        a = [0.50, 0.51, 0.52, 0.52]
        self.assertTrue(ops.gate_first(a, [0.525, 0.526, 0.53, 0.53], 0.50))
        self.assertFalse(ops.gate_first(a, [0.522, 0.522, 0.522, 0.522], 0.50))
        self.assertFalse(ops.gate_first([0.52] * 4, [0.53, 0.50, 0.50, 0.50], 0.50))
        self.assertFalse(ops.gate_first([0.40] * 4, [0.45] * 4, 0.50))
        with self.assertRaises(ValueError):
            ops.gate_first([0.5, 0.5, 0.5], [0.6] * 4, 0.4)


class TestChooseCheckpoint(unittest.TestCase):
    def test_tie_prefers_earlier_regardless_of_order(self):
        self.assertEqual(ops.choose_checkpoint([(100, 0.60), (200, 0.62),
                                                (300, 0.62)]), (200, 0.62))
        self.assertEqual(ops.choose_checkpoint([(300, 0.62), (200, 0.62),
                                                (100, 0.60)]), (200, 0.62))


class TestMetrics(unittest.TestCase):
    def test_true_aggregates_across_images(self):
        imgs = [[[10, 5, 5], [4, 4, 2], [0, 1, 1], [3, 0, 3]],
                [[0, 0, 0], [1, 0, 1], [2, 2, 0], [0, 0, 0]]]
        m = ops.metrics_from_counts(imgs)
        self.assertAlmostEqual(m["per_class"]["EX"], 10 / 20)
        self.assertAlmostEqual(m["per_class"]["HE"], 5 / 12)
        self.assertAlmostEqual(m["per_class"]["SE"], 2 / 6)
        self.assertAlmostEqual(m["per_class"]["MA"], 3 / 6)
        self.assertAlmostEqual(m["mIoU"],
                               sum(m["per_class"].values()) / 4)
        self.assertEqual(m["counts"]["HE"],
                         {"tp": 5.0, "fp": 4.0, "fn": 3.0})

    def test_reject_zero_union(self):
        with self.assertRaises(ValueError):
            ops.metrics_from_counts({c: {"tp": 0, "fp": 0, "fn": 0}
                                     for c in ops.CLASSES})


class TestSplitAssignment(unittest.TestCase):
    def test_wrapper_and_direct(self):
        import runtime
        for wrapper in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); mapping={}
                for split,name in [('train','a'),('val','b'),('test','c')]:
                    (root/'images'/split).mkdir(parents=True)
                    (root/'labels'/split).mkdir(parents=True)
                    (root/'images'/split/(name+'.jpg')).write_bytes(b'image')
                    (root/'labels'/split/(name+'.png')).write_bytes(b'label')
                    (root/(split+'.txt')).write_text(name+'.jpg')
                    mapping[name+'.jpg']={'split':split}
                (root/'split_assignment.json').write_text(json.dumps({'images':mapping} if wrapper else mapping))
                self.assertEqual(len(runtime.audit_split(str(root), expected=(1,1,1))), 3)
                mapping['a.jpg']['split']='test'
                (root/'split_assignment.json').write_text(json.dumps(mapping))
                with self.assertRaises(RuntimeError): runtime.audit_split(str(root),expected=(1,1,1))


if __name__ == "__main__":
    unittest.main()
