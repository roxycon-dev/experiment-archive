"""Completion/shutdown guards only; no SSH, signals, GPU or SGD."""
import copy
import unittest
from unittest import mock
import controller_performance04 as c
import performance04_transition_support as support

class PerformanceControllerTests(unittest.TestCase):
    def setUp(self):
        original=copy.deepcopy(c.state)
        def restore():c.state.clear();c.state.update(original)
        self.addCleanup(restore)
        c.state.update(formal_training_started=True,shutdown_armed=True,outcome='COMPLETE_WITH_LOCKED_TEST')

    def test_final_result_failure_preserves_instance_after_completed_compute(self):
        with mock.patch.object(c,'save_final_result',side_effect=OSError('disk full')), \
             mock.patch.object(c,'request_shutdown') as shutdown,mock.patch.object(c,'write'):
            c.finish_and_shutdown()
        shutdown.assert_not_called()
        self.assertEqual(c.state['outcome'],'FINALIZATION_FAILED')
        self.assertEqual(c.state['status'],'INSTANCE_PRESERVED')
        self.assertEqual(c.state['job_outcome_before_finalization_failure'],'COMPLETE_WITH_LOCKED_TEST')

    def test_successful_finalization_after_complete_requests_shutdown(self):
        with mock.patch.object(c,'save_final_result'),mock.patch.object(c,'request_shutdown') as shutdown:
            c.finish_and_shutdown()
        shutdown.assert_called_once()

    def test_unapproved_transition_cannot_start_formal_stage(self):
        with mock.patch.object(support,'validate_committed_stop',side_effect=RuntimeError('No approved stop')), \
             mock.patch.object(c,'stage') as stage:
            with self.assertRaisesRegex(RuntimeError,'No approved stop'):c.run_job()
        stage.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
