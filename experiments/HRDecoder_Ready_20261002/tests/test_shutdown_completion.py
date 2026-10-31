import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/'campaign/shutdown_after_complete.py'
spec = importlib.util.spec_from_file_location('completion_guard', SOURCE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.main = Path(self.tmp.name)/'main'
        self.root = self.main/'round2_highres'
        self.root.mkdir(parents=True)
        self.save('shutdown_policy.json', {'enabled': True, 'job_order': guard.TAGS, 'remote_root': '/test'})
        protocol = {'job_order': guard.TAGS, 'config_sha256': {}, 'max_updates_per_arm': 4000,
                    'baseline_val_miou': .5, 'test_policy': 'closed'}
        state = {'status': 'complete', 'active_job': None, 'jobs': {}, 'started_at_epoch': 100, 'finished_at_epoch': 400}
        result = {'status': 'complete', 'results': {}}
        journal = {}
        for tag in guard.TAGS:
            config = self.root/'configs'/f'{tag}.yaml'
            config.parent.mkdir(exist_ok=True)
            config.write_text('tag: '+tag)
            mirrored = self.root/'server_results/configs'/config.name
            mirrored.parent.mkdir(parents=True, exist_ok=True)
            mirrored.write_bytes(config.read_bytes())
            digest = guard.sha(config)
            protocol['config_sha256'][tag] = digest
            row = {'update': 1000, 'mIoU_lesion4': .4, 'fixed_fusion_miou': .5, 'fixed_fusion_gain': 0,
                   'fixed_fusion_baseline_miou': .5, 'fixed_fusion_counts': [[1, 1, 0]]*4,
                   'n_images': 76, 'fixed_fusion_images': 76, 'eval_inference': 'whole_image', 'split': 'val'}
            self.save('server_results/work_dirs/'+tag+'/eval_u1000.json', row)
            self.save('server_results/work_dirs/'+tag+'/eval_u2000.json', dict(row, update=2000))
            summary = {'status': 'stopped', 'stop_reason': 'paired_no_gain_or_learning', 'updates_done': 2000,
                       'best': {'update': 1000, 'selection_score': .5, 'selection_metric': 'fixed_fusion_miou'},
                       'config_file_sha256': digest}
            self.save('server_results/work_dirs/'+tag+'/completion_summary.json', summary)
            state['jobs'][tag] = {'summary': summary, 'started_at_epoch': 100, 'finished_at_epoch': 250}
            result['results'][tag] = {'best_update': 1000}
            weight = self.root/'server_results/work_dirs'/tag/'ckpt_best.pth'
            weight.write_bytes(b'checked model '+tag.encode())
            journal[tag+'/ckpt_best.pth'] = {'local': str(weight), 'bytes': weight.stat().st_size, 'sha256': guard.sha(weight)}
        self.save('protocol.json', protocol)
        self.save('server_results/work_dirs/campaign_state.json', state)
        self.save('server_results/work_dirs/finish_state.json', result)
        self.save('server_results/reports/round_result.json', result)
        self.save('reports/verified_backups.json', journal)
        historical = self.main/'best_pipeline/original_deployment_lock.json'
        historical.parent.mkdir(parents=True)
        historical.write_bytes(b'original evaluated lock')
        historical_weight = historical.parent/'old.pth'
        historical_weight.write_bytes(b'previous best')
        guard.write(historical.parent/'local_weight_manifest.json', {
            'original_lock_sha256': guard.sha(historical),
            'weights': {'old': {'path': str(historical_weight), 'bytes': historical_weight.stat().st_size, 'sha256': guard.sha(historical_weight)}}})
        self.lock_patch = patch.object(guard, 'LOCK_SHA', guard.sha(historical))
        self.lock_patch.start()

    def tearDown(self):
        self.lock_patch.stop()
        self.tmp.cleanup()

    def save(self, rel, value):
        guard.write(self.root/rel, value)

    def test_running_or_failed_queue_cannot_shutdown(self):
        for phase in ('running', 'engineering_attention_required'):
            state = guard.read(self.root/'server_results/work_dirs/campaign_state.json')
            state['status'] = phase
            self.save('server_results/work_dirs/campaign_state.json', state)
            with self.assertRaises(guard.Waiting):
                guard.prepare(self.root)

    def test_partial_backup_blocks_shutdown(self):
        (self.root/'server_results/work_dirs'/guard.TAGS[1]/'ckpt_best.pth').unlink()
        with self.assertRaises(guard.Waiting):
            guard.prepare(self.root)

    def test_corrupt_backup_is_rejected(self):
        (self.root/'server_results/work_dirs'/guard.TAGS[0]/'ckpt_best.pth').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            guard.prepare(self.root)

    def test_complete_validated_queue_finalizes_idempotently(self):
        plan = guard.prepare(self.root)
        self.assertEqual(sum(x['updates_done'] for x in plan['jobs'].values()), 4000)
        guard.finalize(self.root, plan)
        guard.finalize(self.root, plan)
        archive = self.main.parent/'_全实验档案_20260920/01_全目录实验清单.md'
        self.assertEqual(archive.read_text(encoding='utf-8').count('<!-- HRD_ROUND2_COMPLETE_20261003 -->'), 1)

    def test_changed_validation_counts_are_rejected(self):
        p = self.root/'server_results/work_dirs'/guard.TAGS[1]/'eval_u2000.json'
        row = guard.read(p)
        row['fixed_fusion_counts'][0][0] += 100
        guard.write(p, row)
        with self.assertRaises(ValueError):
            guard.prepare(self.root)

    def test_changed_evaluated_deployment_lock_blocks_shutdown(self):
        (self.main/'best_pipeline/original_deployment_lock.json').write_bytes(b'changed evaluated pipeline')
        with self.assertRaises(ValueError):
            guard.prepare(self.root)


if __name__ == '__main__':
    unittest.main()
