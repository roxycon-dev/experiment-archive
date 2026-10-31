"""Local simulated control-flow checks; never connects or runs model SGD."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import controller as c


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        (root/'receipts').mkdir();(root/'receipts/environment_ready').touch()
        self.original=copy.deepcopy(c.state)
        def restore():
            c.state.clear();c.state.update(self.original)
        self.addCleanup(restore)
        c.state.update(formal_training_started=False,shutdown_armed=False)
        c.state.pop('outcome',None)
        self.patch=mock.patch.multiple(c,BASE=root,CODE=root/'code',PREFLIGHT=root/'preflight',RUN=root/'run',STATUS=root/'job_status.json')
        self.patch.start();self.addCleanup(self.patch.stop)

    def test_report_failure_during_preflight_preserves_instance(self):
        with mock.patch.object(c,'save_final_result',side_effect=OSError('disk full')),mock.patch.object(c,'request_shutdown') as shutdown:
            c.finish_and_shutdown()
            shutdown.assert_not_called()
            self.assertEqual(c.state['status'],'INSTANCE_PRESERVED')

    def test_completed_formal_job_requests_shutdown_even_if_report_fails(self):
        c.observe_formal_start({'executed_updates':1,'run_identity':{'source_sha256':c.SOURCE_SHA}})
        c.state['outcome']='COMPLETE_WITH_LOCKED_TEST'
        with mock.patch.object(c,'save_final_result',side_effect=OSError('disk full')),mock.patch.object(c,'request_shutdown') as shutdown:
            c.finish_and_shutdown()
            shutdown.assert_called_once()

    def test_only_completed_current_source_formal_update_arms_shutdown(self):
        for progress in (None,{}, {'executed_updates':0,'run_identity':{'source_sha256':c.SOURCE_SHA}},
                         {'executed_updates':1,'run_identity':{'source_sha256':'wrong'}},
                         {'executed_updates':True,'run_identity':{'source_sha256':c.SOURCE_SHA}}):
            c.observe_formal_start(progress)
            self.assertFalse(c.state['shutdown_armed'])
        c.observe_formal_start({'executed_updates':1,'run_identity':{'source_sha256':c.SOURCE_SHA}})
        self.assertTrue(c.state['shutdown_armed'])
        self.assertTrue(c.state['formal_training_started'])

    def test_failure_or_pause_after_formal_start_preserves_instance(self):
        c.observe_formal_start({'executed_updates':1,'run_identity':{'source_sha256':c.SOURCE_SHA}})
        for outcome in ('SETUP_FAILED','PREFLIGHT_FAILED','CONTROLLER_FAILED','FORMAL_TRAINING_FAILED',
                        'FORMAL_TRAINING_PAUSED','LOCKED_TEST_FAILED'):
            with self.subTest(outcome=outcome):
                c.state['outcome']=outcome
                with mock.patch.object(c,'save_final_result'),mock.patch.object(c,'request_shutdown') as shutdown:
                    c.finish_and_shutdown()
                    shutdown.assert_not_called()

    def test_completion_without_formal_start_preserves_instance(self):
        c.state['outcome']='COMPLETE_WITH_LOCKED_TEST'
        with mock.patch.object(c,'save_final_result'),mock.patch.object(c,'request_shutdown') as shutdown:
            c.finish_and_shutdown()
            shutdown.assert_not_called()

    def test_provider_call_is_itself_guarded_before_training(self):
        with mock.patch.object(c.subprocess,'run') as command:
            c.request_shutdown()
            command.assert_not_called()

    def test_failed_gate_never_starts_formal_training(self):
        with mock.patch.object(c,'stage',return_value=1) as stage:
            c.run_job()
            self.assertEqual(stage.call_count,1)
            self.assertEqual(c.state['outcome'],'PREFLIGHT_FAILED')

    def test_source_mismatch_never_starts_formal_training(self):
        def preflight(*args):
            c.PREFLIGHT.mkdir();c.write(c.PREFLIGHT/'preflight.json',{'status':'PASS','source_sha256':'wrong'})
            return 0
        with mock.patch.object(c,'stage',side_effect=preflight) as stage:
            with self.assertRaisesRegex(RuntimeError,'different source'):
                c.run_job()
            self.assertEqual(stage.call_count,1)

    def test_paused_training_is_not_complete_or_tested(self):
        def stages(name,*args):
            if name=='PREFLIGHT':
                c.PREFLIGHT.mkdir();c.write(c.PREFLIGHT/'preflight.json',{'status':'PASS','source_sha256':c.SOURCE_SHA})
            else:
                c.RUN.mkdir();c.write(c.RUN/'result.json',{'status':'PAUSED'})
            return 0
        with mock.patch.object(c,'stage',side_effect=stages) as stage:
            c.run_job()
            self.assertEqual(stage.call_count,2)
            self.assertEqual(c.state['outcome'],'FORMAL_TRAINING_PAUSED')

    def test_test_runs_only_for_eligible_fixed_40k(self):
        def stages(name,*args):
            if name=='PREFLIGHT':
                c.PREFLIGHT.mkdir();c.write(c.PREFLIGHT/'preflight.json',{'status':'PASS','source_sha256':c.SOURCE_SHA})
            elif name=='FORMAL_TRAINING':
                c.RUN.mkdir();c.write(c.RUN/'result.json',{'status':'VAL_TARGET_NOT_REACHED'})
                c.write(c.RUN/'endpoint_lock.json',{'completed':40000,'test_permitted':False})
            else:self.fail('ineligible test was started')
            return 0
        with mock.patch.object(c,'stage',side_effect=stages) as stage:
            c.run_job()
            self.assertEqual(stage.call_count,2)
            self.assertEqual(c.state['outcome'],'COMPLETE_VAL_TARGET_NOT_REACHED')

    def test_monitor_failure_signals_owned_child_before_shutdown(self):
        child=mock.Mock();child.poll.return_value=None;child.returncode=1
        with mock.patch.object(c.subprocess,'Popen',return_value=child),mock.patch.object(c,'write',side_effect=[None,OSError('write failure')]):
            with self.assertRaises(OSError):c.stage('PREFLIGHT',['verify.py'],'dummy.log')
        child.send_signal.assert_called_once_with(c.signal.SIGINT)
        child.wait.assert_called_once_with(timeout=45)


if __name__=='__main__':
    unittest.main(verbosity=2)
