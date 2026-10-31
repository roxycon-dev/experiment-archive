"""Local fault injection for the unexecuted approved 03 -> 04 transition.

Executes only the extracted Python body under fake process, signal, clock and
/proc APIs. Files are disposable fixtures in TemporaryDirectory. Never imports
training code, connects to a server, signals real processes, or runs model SGD.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid  # Resolve this before replacing selected imported modules.


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "approved_stop03_at_boundary.sh"
SOURCE = SCRIPT.read_text(encoding="utf-8")
BODY = SOURCE.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
COMPILED = compile(BODY, str(SCRIPT) + "::<mocked heredoc>", "exec")
OLD = "d3032fc8b20f30a058adbcf3e6bc5a3485b83adafe9b7cdce0682a2d55d7f627"
NEW = "14e47c3749db5dd28c2938ddbb227aad66b07c40b4fb9d049fdb0564b91c95d0"
PID, OWNER = 2562, 2160
STOP, CONT, KILL = 19, 18, 9


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class ProcPath:
    """Only explicitly supported fake /proc operations are allowed."""

    def __init__(self, simulation, path):
        self.sim = simulation
        self.parts = str(path).split("/")
        self.pid = int(self.parts[2])

    def read_bytes(self):
        if self.parts[-1] != "cmdline":
            raise AssertionError("Unexpected fake proc read")
        if self.pid == PID:
            return b"python train.py --output " + str(self.sim.run).encode()
        if self.pid == OWNER:
            return b"python controller.py"
        raise AssertionError("Unowned proc access")

    def resolve(self):
        if self.pid == OWNER and self.parts[-1] == "cwd":
            return self.sim.base
        raise AssertionError("Unexpected proc resolve")

    def glob(self, pattern):
        if self.parts[-1] != "task" or pattern != "*/status":
            raise AssertionError("Unexpected proc glob")
        sim, pid = self.sim, self.pid

        class TaskStatus:
            def read_text(self):
                return "State:\t" + sim.table[pid]["state"] + " (mock)\n"

        return [TaskStatus(), TaskStatus()]


class Simulation:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="performance04_mock_")
        self.base = Path(self.temp.name)
        self.run = self.base / "run_seed42_launch02"
        self.run.mkdir()
        (self.base / "receipts").mkdir()
        self.table = {
            PID: {"pid": PID, "state": "R", "ppid": OWNER, "pgrp": PID, "starttime": 111},
            OWNER: {"pid": OWNER, "state": "S", "ppid": 1, "pgrp": OWNER, "starttime": 222},
        }
        self.evidence = {"new_source_sha256": NEW, "new_preflight_sha256": "preflight-bytes",
                         "production_parity_sha256": "production-bytes"}
        self.signals = []
        self.candidate_calls = 0
        self.clock = 0.
        self.freeze_timeout_pid = None
        self.fail_resume_pid = None
        self.fail_kill_pid = None
        self.fail_prefix_write = False
        self.fail_canonical_write = False
        self.fail_stdout = False
        self.fail_candidate = False
        self.error = None
        self.namespace = {}
        self.write_json(self.base / "job_status.json", {"source_sha256": OLD})
        (self.base / "controller.py").write_text("# archived mock controller\n")
        self.write_json(self.base / "controller_v3_started.json", {"pid": OWNER})
        self.write_json(self.run / "progress.json", {
            "executed_updates": 5, "completed": 5, "checkpoint_persisted_updates": 3,
            "in_flight_update": None, "run_identity": {"source_sha256": OLD}})
        self.write_ledger([1, 2, 3, 4, 5])

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_ledger(self, updates):
        (self.run / "train.jsonl").write_text("".join(
            json.dumps({"update": n, "executed_updates": n}) + "\n" for n in updates), encoding="utf-8")

    def process(self, pid):
        item = self.table.get(pid)
        return None if item is None else dict(item)

    def alive(self, identity):
        current = self.table.get(identity["pid"])
        return bool(current and current["starttime"] == identity["starttime"] and current["state"] != "Z")

    def active_group(self, group):
        return [dict(v) for v in self.table.values() if v["pgrp"] == group and v["state"] != "Z"]

    def signal(self, target, value, *, group):
        if target not in (PID, OWNER):
            raise AssertionError("Attempted signal to an unowned target")
        if group and target != PID:
            raise AssertionError("Attempted signal to an unowned group")
        self.signals.append((target, value, group))
        if value == CONT and target == self.fail_resume_pid:
            raise ProcessLookupError("injected resume signal failure")
        if value == KILL and target == self.fail_kill_pid:
            raise PermissionError("injected kill signal failure")
        if value == STOP:
            if target != self.freeze_timeout_pid:
                self.table[target]["state"] = "T"
        elif value == CONT:
            self.table[target]["state"] = "R"
        elif value == KILL:
            self.table[target]["state"] = "Z"
        else:
            raise AssertionError("Unexpected signal")

    def validate_candidate(self):
        self.candidate_calls += 1
        if self.fail_candidate:
            raise RuntimeError("injected candidate evidence mismatch")
        return dict(self.evidence)

    def monotonic(self):
        self.clock += .02
        return self.clock

    def sleep(self, seconds):
        self.clock += seconds

    def path(self, value):
        return ProcPath(self, value) if str(value).startswith("/proc/") else Path(value)

    def run_script(self):
        support = types.ModuleType("performance04_transition_support")
        support.BASE, support.OLD_SOURCE = self.base, OLD
        support.require = lambda condition, message: None if condition else (_ for _ in ()).throw(RuntimeError(message))
        support.read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
        support.process, support.alive, support.active_group = self.process, self.alive, self.active_group
        support.validate_candidate = self.validate_candidate
        fake_os = types.ModuleType("os")
        fake_os.fsync = os.fsync
        fake_os.kill = lambda target, value: self.signal(target, value, group=False)
        fake_os.killpg = lambda target, value: self.signal(target, value, group=True)
        fake_signal = types.ModuleType("signal")
        fake_signal.SIGSTOP, fake_signal.SIGCONT, fake_signal.SIGKILL = STOP, CONT, KILL
        fake_time = types.ModuleType("time")
        fake_time.monotonic, fake_time.sleep = self.monotonic, self.sleep
        fake_time.time = lambda: 1792200000. + self.clock
        fake_pathlib = types.ModuleType("pathlib")
        fake_pathlib.Path = self.path
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path.name == "abandoned_for_approved_performance04.writing" and self.fail_prefix_write:
                raise OSError("injected old-prefix persistence failure")
            if path.name == "stopped03_for_approved_performance04.writing" and self.fail_canonical_write:
                raise OSError("injected final commit persistence failure")
            return original_open(path, *args, **kwargs)

        def output(*args, **kwargs):
            if self.fail_stdout:
                raise BrokenPipeError("injected stdout failure after commit")

        modules = {"performance04_transition_support": support, "os": fake_os, "signal": fake_signal,
                   "time": fake_time, "pathlib": fake_pathlib}
        self.namespace = {"__name__": "mocked_transition_only", "print": output}
        with mock.patch.dict(sys.modules, modules), mock.patch.object(Path, "open", guarded_open):
            try:
                exec(COMPILED, self.namespace, self.namespace)
            except BaseException as error:
                self.error = error

    def transaction(self):
        paths = list((self.base / "receipts").glob("approved_stop_transaction_*.json"))
        if not paths:
            return None
        if len(paths) != 1:
            raise AssertionError("Unexpected transaction count")
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def canonical(self):
        return self.base / "receipts/stopped03_for_approved_performance04.json"

    def actual_start_guard(self):
        """Execute the real read-only guard, with only its process/candidate inputs mocked."""
        module = load_module("transition_guard_under_test", "performance04_transition_support.py")
        module.BASE = self.base
        module.process, module.alive, module.active_group = self.process, self.alive, self.active_group
        module.validate_candidate = lambda: dict(self.evidence)
        return module.validate_committed_stop()

    def close(self):
        self.temp.cleanup()


class TransitionFaultTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation()
        self.addCleanup(self.sim.close)

    def assert_no_commit(self):
        self.assertFalse(self.sim.canonical().exists())
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            self.sim.actual_start_guard()

    def assert_both_running(self):
        self.assertNotIn(self.sim.table[PID]["state"], ("T", "t", "Z"))
        self.assertNotIn(self.sim.table[OWNER]["state"], ("T", "t", "Z"))

    def test_shell_requires_user_confirmation_before_python(self):
        self.assertEqual(SOURCE.splitlines()[:2], [
            "set -eu", 'test "${ALLOW_ENDING_CURRENT_FORMAL_RUN:-}" = USER_CONFIRMED'])

    def test_invalid_candidate_does_not_send_any_signal(self):
        self.sim.fail_candidate = True
        self.sim.run_script()
        self.assertIsInstance(self.sim.error, RuntimeError)
        self.assertEqual(self.sim.signals, [])
        self.assertIsNone(self.sim.transaction())
        self.assert_both_running()
        self.assert_no_commit()

    def test_training_freeze_timeout_resumes_training(self):
        self.sim.freeze_timeout_pid = PID
        self.sim.run_script()
        self.assertIn("did not confirm stopped", str(self.sim.error))
        self.assertIn((PID, CONT, True), self.sim.signals)
        self.assert_both_running()
        self.assertEqual(self.sim.transaction()["status"], "ABORTED_CURRENT_RUN_CONTINUES")
        self.assert_no_commit()

    def test_controller_freeze_timeout_attempts_both_resumes(self):
        self.sim.freeze_timeout_pid = OWNER
        self.sim.run_script()
        self.assertIn((OWNER, CONT, False), self.sim.signals)
        self.assertIn((PID, CONT, True), self.sim.signals)
        self.assert_both_running()
        self.assertTrue(self.sim.transaction()["recovery_confirmed"])
        self.assert_no_commit()

    def test_ledger_gap_restores_both_owned_processes(self):
        self.sim.write_ledger([1, 2, 4, 5])
        self.sim.run_script()
        self.assertIn("not contiguous", str(self.sim.error))
        self.assertEqual([item for item in self.sim.signals if item[1] == CONT],
                         [(OWNER, CONT, False), (PID, CONT, True)])
        self.assert_both_running()
        self.assertEqual(self.sim.transaction()["status"], "ABORTED_CURRENT_RUN_CONTINUES")
        self.assert_no_commit()

    def test_first_resume_failure_does_not_skip_second_resume(self):
        self.sim.write_ledger([1, 2, 4])
        self.sim.fail_resume_pid = OWNER
        self.sim.run_script()
        self.assertIn((OWNER, CONT, False), self.sim.signals)
        self.assertIn((PID, CONT, True), self.sim.signals)
        self.assertEqual(self.sim.table[OWNER]["state"], "T")
        self.assertEqual(self.sim.table[PID]["state"], "R")
        self.assertEqual(self.sim.transaction()["status"], "ABORTED_RECOVERY_FAILED_NO_NEW_TRAINING")
        self.assertFalse(self.sim.transaction()["recovery_confirmed"])
        self.assert_no_commit()

    def test_second_resume_failure_is_not_reported_as_continuing(self):
        self.sim.write_ledger([1, 2, 4])
        self.sim.fail_resume_pid = PID
        self.sim.run_script()
        self.assertEqual(self.sim.table[OWNER]["state"], "R")
        self.assertEqual(self.sim.table[PID]["state"], "T")
        self.assertEqual(self.sim.transaction()["status"], "ABORTED_RECOVERY_FAILED_NO_NEW_TRAINING")
        self.assert_no_commit()

    def test_first_kill_failure_does_not_resume_or_commit(self):
        self.sim.fail_kill_pid = PID
        self.sim.run_script()
        self.assertIsInstance(self.sim.error, PermissionError)
        self.assertFalse(any(value == CONT for _, value, _ in self.sim.signals))
        self.assertEqual(self.sim.transaction()["status"], "PARTIAL_STOP_NO_NEW_TRAINING")
        self.assert_no_commit()

    def test_controller_kill_failure_cannot_resume_uncontrolled_training(self):
        self.sim.fail_kill_pid = OWNER
        self.sim.run_script()
        self.assertEqual(self.sim.table[PID]["state"], "Z")
        self.assertEqual(self.sim.table[OWNER]["state"], "T")
        self.assertFalse(any(value == CONT for _, value, _ in self.sim.signals))
        self.assertEqual(self.sim.transaction()["status"], "PARTIAL_STOP_NO_NEW_TRAINING")
        self.assert_no_commit()

    def test_old_prefix_write_failure_leaves_no_canonical_commit(self):
        self.sim.fail_prefix_write = True
        self.sim.run_script()
        self.assertIsInstance(self.sim.error, OSError)
        self.assertEqual(self.sim.transaction()["status"], "PARTIAL_STOP_NO_NEW_TRAINING")
        self.assert_no_commit()

    def test_final_commit_write_failure_blocks_even_with_old_prefix_mirror(self):
        self.sim.fail_canonical_write = True
        self.sim.run_script()
        self.assertTrue((self.sim.run / "abandoned_for_approved_performance04.json").is_file())
        self.assertEqual(self.sim.transaction()["status"], "PARTIAL_STOP_NO_NEW_TRAINING")
        self.assert_no_commit()

    def test_success_commits_all_three_equal_records_and_exact_prefix(self):
        self.sim.run_script()
        self.assertIsNone(self.sim.error)
        record = self.sim.actual_start_guard()
        self.assertEqual(record, self.sim.transaction())
        self.assertEqual(record, json.loads((self.sim.run / "abandoned_for_approved_performance04.json").read_text()))
        self.assertEqual(record["executed_updates"], 5)
        self.assertEqual(record["checkpoint_persisted_updates"], 3)
        self.assertEqual(record["additional_executed_updates_uncertain"], [0, 0])
        self.assertEqual(record["status"], "STOP_CONFIRMED")
        self.assertTrue(record["old_executed_steps_are_additional_disclosed_budget"])
        self.assertTrue(record["new_trial_must_start_from_ImageNet"])

    def test_stdout_failure_after_commit_does_not_downgrade_records(self):
        self.sim.fail_stdout = True
        self.sim.run_script()
        self.assertIsInstance(self.sim.error, BrokenPipeError)
        self.assertEqual(self.sim.actual_start_guard(), self.sim.transaction())
        self.assertEqual(self.sim.transaction()["status"], "STOP_CONFIRMED")

    def test_mismatched_old_prefix_mirror_blocks_new_controller(self):
        self.sim.run_script()
        prefix = self.sim.run / "abandoned_for_approved_performance04.json"
        broken = json.loads(prefix.read_text()); broken["status"] = "PARTIAL_STOP_NO_NEW_TRAINING"
        self.sim.write_json(prefix, broken)
        with self.assertRaisesRegex(RuntimeError, "incomplete/inconsistent"):
            self.sim.actual_start_guard()

    def test_living_old_process_blocks_new_controller(self):
        self.sim.run_script()
        self.sim.table[OWNER]["state"] = "R"
        with self.assertRaisesRegex(RuntimeError, "remains alive"):
            self.sim.actual_start_guard()

    def test_changed_candidate_evidence_blocks_new_controller(self):
        self.sim.run_script()
        self.sim.evidence["new_preflight_sha256"] = "different-preflight-bytes"
        with self.assertRaisesRegex(RuntimeError, "evidence changed"):
            self.sim.actual_start_guard()


if __name__ == "__main__":
    unittest.main(verbosity=2)
