"""Protocol, failure classification, identity, and real-draw integrity tests without GPUs."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location(
    'init_controller', Path(__file__).parents[1] / 'scripts/inspire/run_prefeval_unet_init_ablation.py')
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def config(root):
    return controller.configure(controller.parser().parse_args([
        '--repo', str(root / 'repo'), '--run-root', str(root / 'run'),
        '--python', str(root / 'python'), '--base', str(root / 'base'),
        '--reader', str(root / 'reader'), '--official-source', str(root / 'official'),
        '--parent-checkpoint', str(root / 'parent.pt'), '--teachers', str(root / 'teachers'),
        '--train-variants', str(root / 'train-variants.json'), '--dev-variants', str(root / 'dev-variants.json')]))


def mock_runner():
    runner = Mock()
    runner.stop = threading.Event()
    runner.terminate.side_effect = runner.stop.set
    runner.wait_for_shutdown.side_effect = runner.stop.set
    return runner


def outcome(args, arm, rate, attempt=None):
    shared = {'conditions_sha256': 'conditions', 'targets_sha256': 'targets', 'training_rows_sha256': 'rows',
              'schedule_sha256': 'schedule', 'train_seed': controller.TRAIN_SEED, 'effective_batch': 4}
    manifest = {'shared_training_identity': shared, 'architecture_sha256': 'arch',
                'frozen_module_hashes': {'vae': 'vae', 'text': 'text'}, 'runtime': {'torch': 'test'},
                'conditioning_seed': 0, 'train_seed': controller.TRAIN_SEED, 'init_seed': controller.INIT_SEED,
                'parent_sha256': 'parent', 'unet_init': 'random' if arm in controller.RANDOM_ARMS else 'parent',
                'initial_state_sha256': 'random-state' if arm in controller.RANDOM_ARMS else 'parent-state', 'learning_rate': rate}
    return {'status': 'complete', 'output': str(controller.stage_output(args, arm, attempt)), 'manifest': manifest,
            'actual_draw_signature': 'prefix' if attempt is not None else 'all-draws',
            'first_128_draw_signature': 'prefix'}


class ControllerProtocolTests(unittest.TestCase):
    def test_plan_has_no_io_and_all_twenty_evaluations_are_fixed(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as stream:
            root = Path(directory)
            self.assertEqual(controller.main(['--repo', str(root / 'absent-repo'), '--run-root', str(root / 'run'),
                                              '--plan-only']), 0)
            result = json.loads(stream.getvalue())
            self.assertFalse((root / 'run').exists())
        self.assertEqual(len(result['prechecks']), 4)
        self.assertEqual(len(result['fallback_prechecks']), 2)
        self.assertEqual(len(result['evaluations']), 16)
        self.assertEqual(len(result['official_evaluations_after_train_dev']), 4)
        self.assertEqual(result['protocol']['training']['total_formal_updates'], 93440)
        self.assertEqual({r['gpu'] for r in result['prechecks']}, {'0', '1', '2', '3'})
        self.assertEqual(result['resources']['evaluation_gpus'], ['0', '1', '2', '3'])
        self.assertEqual(result['resources']['evaluation_workers'], 4)
        self.assertEqual(result['resources']['max_simultaneous_gpus'], 4)
        self.assertFalse(result['resources']['training_evaluation_overlap'])
        self.assertEqual([j['split'] for j in result['evaluations']], ['train'] * 8 + ['dev'] * 8)
        for job in result['official_evaluations_after_train_dev']:
            self.assertEqual(job['step'], 23360)
            self.assertIn('--history-file', job['command'])
            self.assertNotIn('--initial-variants', job['command'])
            self.assertEqual(job['eligible_gpus'], ['0', '1', '2', '3'])
            self.assertNotIn('gpu', job)  # Assigned only when a dedicated GPU worker dequeues it.

    def test_evaluation_pool_reuses_training_gpus_and_accepts_legacy_single_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory))
            self.assertEqual(args.eval_gpus, ('0', '1', '2', '3'))
            self.assertEqual(args.eval_gpu, '0')
            controller.configure(args)
            args.eval_gpus = ('not-a-gpu',)
            with self.assertRaisesRegex(controller.ProtocolError, 'numeric GPU'):
                controller.configure(args)
        single = controller.configure(controller.parser().parse_args(['--eval-gpu', '2']))
        self.assertEqual(single.eval_gpus, ('2',))
        self.assertEqual(single.eval_gpu, '2')
        subset = controller.configure(controller.parser().parse_args(['--eval-gpus', '3,1']))
        self.assertEqual(subset.eval_gpus, ('3', '1'))
        self.assertEqual(subset.eval_gpu, '3')
        with self.assertRaisesRegex(controller.ProtocolError, 'reuse'):
            controller.configure(controller.parser().parse_args(['--eval-gpus', '0,4']))
        with self.assertRaisesRegex(controller.ProtocolError, 'numeric GPU'):
            controller.configure(controller.parser().parse_args(['--eval-gpu', 'invalid']))
        with self.assertRaises(Exception):
            controller.parse_gpus('0,0,1,2')
        for invalid in ('0,0', '', 'x', '0,1,2,3,4'):
            with self.assertRaises(Exception):
                controller.parse_eval_gpus(invalid)

    def test_formal_fresh_commands_reference_successful_precheck_without_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory))
            for arm in controller.ARMS:
                rate = controller.LOW_LR if arm in controller.LOW_ARMS else controller.FALLBACK_LR
                command = controller.train_command(args, arm, rate, controller.stage_output(args, arm))
                self.assertIn('--fresh-start', command)
                self.assertNotIn('--continue-from', command)
                self.assertEqual(command[command.index('--steps') + 1], '23360')
                self.assertEqual(command[command.index('--snapshot-steps') + 1], '2048')
                attempt = 0 if arm in controller.LOW_ARMS else 1
                self.assertEqual(command[command.index('--expected-init-audit') + 1],
                                 str(controller.stage_output(args, arm, attempt) / 'manifest.json'))
                self.assertNotEqual(controller.stage_output(args, arm), controller.stage_output(args, arm, attempt))

    def test_identity_cannot_cross_directories_or_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); args.run_root.mkdir()
            identity = controller.register(args, {'source': 'same'})
            with self.assertRaisesRegex(controller.ProtocolError, '--resume'):
                controller.register(args, {'source': 'same'})
            args.resume = True
            self.assertEqual(controller.register(args, {'source': 'same'}), identity)
            with self.assertRaisesRegex(controller.ProtocolError, 'identity changed'):
                controller.register(args, {'source': 'changed'})
            args.eval_gpus = ('0',)
            with self.assertRaisesRegex(controller.ProtocolError, 'identity changed'):
                controller.register(args, {'source': 'same'})

    def test_joint_fallback_occurs_once_and_official_is_after_main_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); calls, order = [], []
            def train(_args, _runner, _identity, arms, rates, attempt=None):
                calls.append((tuple(arms), dict(rates), attempt))
                result = {a: outcome(args, a, rates[a], attempt) for a in arms}
                if attempt == 0:
                    result['P-H'] = {'status': 'numeric_failure', 'output': str(controller.stage_output(args, 'P-H', 0)),
                                     'failure': {'step': 7, 'reason': 'nonfinite_fm_loss'}}
                return result
            def evaluate(_args, _runner, job):
                self.assertEqual(calls[-1][2], None, 'All formal training finishes before any evaluation')
                order.append(job['split']); return {'label': job['label'], 'complete': {}}
            def history(*_):
                self.assertEqual(sorted(order), ['dev'] * 8 + ['train'] * 8)
                return {'sha256': 'history'}
            with patch.object(controller, 'parallel_training', side_effect=train), \
                 patch.object(controller, 'evaluate_job', side_effect=evaluate), \
                 patch.object(controller, 'freeze_official_history', side_effect=history):
                result = controller.execute(args, mock_runner(), 'identity')
            self.assertEqual([x[2] for x in calls], [0, 1, None])
            self.assertEqual(calls[1][0], controller.HIGH_ARMS)
            for _, rates, _ in calls[1:]:
                self.assertEqual(rates, {'P-L': 5e-5, 'P-H': 1.5e-4, 'R-L': 5e-5, 'R-H': 1.5e-4})
            self.assertEqual(order[-4:], ['official'] * 4)
            self.assertEqual(result['evaluation_count'], 20)
            cost = result['precheck_cost']['runs']
            self.assertTrue(any(x['numeric_failure'] and x['numeric_failure']['step'] == 7 for x in cost))

    def test_nonnumeric_exception_has_no_lr_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory))
            with patch.object(controller, 'parallel_training', side_effect=RuntimeError('missing dependency')) as run:
                with self.assertRaisesRegex(RuntimeError, 'missing dependency'):
                    controller.execute(args, Mock(), 'identity')
                self.assertEqual(run.call_count, 1)
            self.assertFalse((args.run_root / 'lr-lock.json').exists())

    def test_second_numeric_failure_stops_before_formal(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory))
            def train(_a, _r, _i, arms, rates, attempt=None):
                self.assertIsNotNone(attempt, 'Formal training must never start')
                result = {a: outcome(args, a, rates[a], attempt) for a in arms}
                result['R-H'] = {'status': 'numeric_failure', 'output': str(controller.stage_output(args, 'R-H', attempt)),
                                 'failure': {'step': 1, 'reason': 'nonfinite_gradient'}}
                return result
            with patch.object(controller, 'parallel_training', side_effect=train) as run:
                with self.assertRaisesRegex(controller.ProtocolError, 'single joint reduction'):
                    controller.execute(args, Mock(), 'identity')
                self.assertEqual(run.call_count, 2)

    def test_same_manifest_but_changed_actual_noise_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory))
            results = {a: outcome(args, a, 5e-5, 0) for a in controller.ARMS}
            controller.compare_training(results)
            results['R-H']['actual_draw_signature'] = 'different-noise'
            with self.assertRaisesRegex(controller.ProtocolError, 'Actual paired'):
                controller.compare_training(results)

    def test_both_random_groups_share_initial_weights_distinct_from_both_parent_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory))
            rates = controller.learning_rates(controller.HIGH_LR)
            results = {a: outcome(args, a, rates[a], 0) for a in controller.ARMS}
            controller.compare_training(results)
            results['R-L']['manifest']['initial_state_sha256'] = 'different-random'
            with self.assertRaisesRegex(controller.ProtocolError, 'Random initial states differ'):
                controller.compare_training(results)
            results['R-L']['manifest']['initial_state_sha256'] = 'parent-state'
            results['R-H']['manifest']['initial_state_sha256'] = 'parent-state'
            with self.assertRaisesRegex(controller.ProtocolError, 'equals parent'):
                controller.compare_training(results)

    def test_low_random_numeric_failure_cannot_trigger_the_high_lr_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); output = controller.stage_output(args, 'R-L', 0)
            def fail(_label, _gpu, _command):
                controller.save(output / 'numeric-failure.json',
                                {'schema': 'prefeval-k1-numeric-failure/v1', 'reason': 'nonfinite_gradient', 'step': 2})
                return 1
            runner = Mock(); runner.run.side_effect = fail
            with self.assertRaisesRegex(controller.ProtocolError, 'outside high-LR precheck'):
                controller.train_stage(args, runner, 'same', 'R-L', 5e-5, 0)
            self.assertEqual(runner.run.call_count, 1)

    def test_partial_stage_without_resume_checkpoint_is_not_restarted(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); args.resume = True
            output = controller.stage_output(args, 'P-L', 0)
            output.mkdir(parents=True); (output / 'manifest.json').write_text('{}')
            receipt = args.run_root / 'bindings/precheck-0-P-L.json'
            controller.save(receipt, {'identity': 'same', 'command_without_fresh_flag': controller.train_command(
                args, 'P-L', 5e-5, output, precheck=True, fresh=False), 'output': str(output.resolve())})
            runner = Mock()
            with self.assertRaisesRegex(controller.ProtocolError, 'no safe same-directory'):
                controller.train_stage(args, runner, 'same', 'P-L', 5e-5, 0)
            runner.run.assert_not_called()

    def test_numeric_marker_must_have_the_explicit_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller.save(root / 'numeric-failure.json', {'reason': 'nonfinite_fm_loss'})
            with self.assertRaisesRegex(controller.ProtocolError, 'Unrecognized numeric'):
                controller.numeric_failure(root)


class EvaluationPoolTests(unittest.TestCase):
    def test_four_dedicated_gpu_workers_share_train_first_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); runner = mock_runner()
            jobs = list(reversed(list(controller.evaluation_jobs(args))))
            original = copy.deepcopy(jobs)
            first_wave = threading.Barrier(4)
            faster_worker_progress = threading.Event()
            mutex = threading.Lock()
            active, visits, dispatched, finished = set(), {}, [], []
            peak = 0

            class TracedQueue(controller.queue.Queue):
                def get_nowait(self):
                    job = super().get_nowait()
                    dispatched.append(job['split'])
                    return job

            def evaluate(_args, _runner, job):
                nonlocal peak
                gpu = job['gpu']
                with mutex:
                    self.assertNotIn(gpu, active, 'Two endpoint jobs must never share a GPU')
                    active.add(gpu); peak = max(peak, len(active))
                    visits[gpu] = visits.get(gpu, 0) + 1
                    first = visits[gpu] == 1
                if first:
                    first_wave.wait(timeout=5)
                if gpu == '0' and first:
                    self.assertTrue(faster_worker_progress.wait(5), 'Other GPU workers must drain the shared queue')
                with mutex:
                    if gpu != '0' and visits[gpu] >= 2:
                        faster_worker_progress.set()
                    active.remove(gpu); finished.append(job['label'])
                return {'label': job['label'], 'complete': {}}

            with patch.object(controller, 'evaluate_job', side_effect=evaluate), \
                 patch.object(controller.queue, 'Queue', TracedQueue):
                result = controller.evaluation_pool(args, runner, 'identity', jobs, phase='evaluation')
            self.assertEqual(peak, 4)
            self.assertEqual(set(visits), set(args.eval_gpus))
            self.assertEqual(dispatched, ['train'] * 8 + ['dev'] * 8)
            self.assertEqual((len(finished), len(set(finished))), (16, 16))
            self.assertEqual(set(finished), {j['label'] for j in jobs})
            ordered = sorted(jobs, key=lambda j: j['split'] != 'train')
            self.assertEqual([r['label'] for r in result], [j['label'] for j in ordered])
            self.assertEqual(jobs, original, 'Dispatch must not mutate immutable commands or outputs')
            status = controller.read_json(args.run_root / 'status.json')
            self.assertEqual((status['completed'], status['pending'], status['active']), (16, 0, {}))
            runner.wait_for_shutdown.assert_not_called()

    def test_official_jobs_run_on_four_gpus_after_one_frozen_history(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); runner = mock_runner()
            jobs = list(controller.evaluation_jobs(args, official=True))
            barrier = threading.Barrier(4)
            def evaluate(_args, _runner, job):
                command = job['command']
                self.assertEqual(command[command.index('--history-file') + 1], str(args.run_root / 'official-history.json'))
                self.assertEqual(job['step'], 23360)
                barrier.wait(timeout=5)
                return {'label': job['label'], 'complete': {}}
            with patch.object(controller, 'evaluate_job', side_effect=evaluate):
                result = controller.evaluation_pool(args, runner, 'identity', jobs, phase='official_final_only')
            self.assertEqual(len(result), 4)

    def test_worker_failure_stops_peers_and_does_not_dequeue_more_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); runner = mock_runner()
            barrier = threading.Barrier(4)
            calls = []
            def evaluate(_args, _runner, job):
                calls.append(job['label'])
                barrier.wait(timeout=5)
                if job['gpu'] == '0':
                    raise RuntimeError('simulated primary failure')
                self.assertTrue(runner.stop.wait(5), 'Running peers must receive cancellation')
                raise controller.ProtocolError('peer stopped')
            with patch.object(controller, 'evaluate_job', side_effect=evaluate):
                with self.assertRaisesRegex(RuntimeError, 'simulated primary failure'):
                    controller.evaluation_pool(args, runner, 'identity', controller.evaluation_jobs(args), phase='evaluation')
            self.assertEqual(len(calls), 4)
            self.assertTrue(runner.stop.is_set())
            runner.wait_for_shutdown.assert_called_once()

    def test_resume_reenters_each_evaluator_and_single_gpu_keeps_commands_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); args.resume = True
            original = list(controller.evaluation_jobs(args))
            args.eval_gpus = ('2',); args.eval_gpu = '2'
            jobs = list(controller.evaluation_jobs(args))
            self.assertEqual([j['command'] for j in jobs], [j['command'] for j in original])
            self.assertEqual([j['output'] for j in jobs], [j['output'] for j in original])
            seen = []
            def evaluate(_args, _runner, job):
                self.assertEqual(job['gpu'], '2')
                seen.append(job['label'])
                return {'label': job['label'], 'complete': {}}
            with patch.object(controller, 'evaluate_job', side_effect=evaluate):
                for _ in range(2):
                    controller.evaluation_pool(args, mock_runner(), 'identity', jobs, phase='evaluation')
            self.assertEqual(seen, [j['label'] for j in jobs] * 2)

    def test_stopped_pool_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            args = config(Path(directory)); runner = mock_runner(); runner.stop.set()
            with patch.object(controller, 'evaluate_job') as evaluate:
                with self.assertRaisesRegex(controller.ProtocolError, 'stopped before all jobs'):
                    controller.evaluation_pool(args, runner, 'identity', controller.evaluation_jobs(args), phase='evaluation')
            evaluate.assert_not_called()


class DrawAuditTests(unittest.TestCase):
    def test_real_draw_signature_ignores_loss_but_catches_noise_and_count_changes(self):
        ids = {f'p:{i:04d}' for i in range(730)}
        row = {'step': 1, 'seconds': 10, 'draws': [
            {'pair_id': f'p:{i:04d}', 'position': 0, 'initial_variant': 0, 'draw': i,
             'sigma': 0.5, 'noise_seed': i, 'noise_sha256': 'a' * 64, 'mse': 2.0} for i in range(4)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'optimization.jsonl'
            path.write_text(json.dumps(row) + '\n')
            first = controller.audit_optimization(path, 1, ids)
            row['seconds'] = 500; row['draws'][0]['mse'] = 100
            path.write_text(json.dumps(row) + '\n')
            second = controller.audit_optimization(path, 1, ids)
            self.assertEqual(first['actual_draw_signature'], second['actual_draw_signature'])
            row['draws'][0]['noise_sha256'] = 'b' * 64
            path.write_text(json.dumps(row) + '\n')
            self.assertNotEqual(first['actual_draw_signature'], controller.audit_optimization(path, 1, ids)['actual_draw_signature'])
            with self.assertRaisesRegex(controller.ProtocolError, 'expected 2'):
                controller.audit_optimization(path, 2, ids)
            row['draws'][0]['draw'] = 99
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(controller.ProtocolError, 'draw index'):
                controller.audit_optimization(path, 1, ids)

    def test_formal_exposure_checks_reject_unbalanced_data(self):
        ids = {f'p:{i:04d}' for i in range(730)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'optimization.jsonl'
            # Shrink only the test's terminal step; the fixed exposure assertion stays 128/64.
            row = {'step': 1, 'draws': [{'pair_id': 'p:0000', 'position': 0, 'initial_variant': 0,
                    'draw': i, 'sigma': 0.5, 'noise_seed': i, 'noise_sha256': 'a' * 64} for i in range(4)]}
            path.write_text(json.dumps(row) + '\n')
            with patch.object(controller, 'STEPS', 1), self.assertRaisesRegex(controller.ProtocolError, 'exposure'):
                controller.audit_optimization(path, 1, ids)


class PreflightTests(unittest.TestCase):
    def test_preflight_hashes_the_last_of_all_730_teachers_without_loading_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args = config(root)
            train = [f'p:{i:04d}' for i in range(730)]; dev = [f'd:{i:04d}' for i in range(90)]
            controller.save(args.split_file, {'train_ids': train, 'internal_dev_ids': dev})
            args.train_data.parent.mkdir(parents=True, exist_ok=True)
            records = [{'base_pair_id': p, 'history': [{'role': 'user', 'content': p},
                                                     {'role': 'assistant', 'content': 'original'}]} for p in train + dev]
            with gzip.open(args.train_data, 'wt', encoding='utf-8') as handle:
                for row in records:
                    handle.write(json.dumps(row) + '\n')
            for path, ids in [(args.train_variants, train), (args.dev_variants, dev)]:
                controller.save(path, {'review_complete': True, 'items': {p: {'preference': p,
                                'variants': ['original', 'neutral', 'heldout']} for p in ids}})
            for pid in train:
                folder = args.teachers / pid.replace(':', '_'); folder.mkdir(parents=True)
                (folder / 'latent.pt').write_bytes(pid.encode())
                controller.save(folder / 'complete.json', {'step': 288, 'binding': {'arm': 'B'},
                                                           'latent_sha256': controller.sha(folder / 'latent.pt')})
            for path in (args.writer, args.evaluator, args.benchmark_ack, args.python, args.parent_checkpoint):
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'placeholder')
            for model in (args.base, args.reader):
                model.mkdir(); (model / 'config.json').write_text('{}')
            forms = args.repo / 'reports/prefeval-k1-l0-l2-20260924'
            for name in ('train-question-forms.json', 'dev-question-forms.json', 'official-question-forms.json'):
                controller.save(forms / name, {})
            benchmark = args.train_data.parent / 'benchmark-disclosures.jsonl.gz'; benchmark.write_bytes(b'benchmark')
            with patch.object(controller.subprocess, 'check_output', side_effect=[controller.OFFICIAL_COMMIT, '']), \
                 patch.object(controller, 'inspect_checkpoint', return_value={'architecture_sha256': 'arch'}):
                binding = controller.preflight(args)
            self.assertEqual(len(binding['targets']), 730)
            last = args.teachers / train[-1].replace(':', '_') / 'latent.pt'
            last.write_bytes(b'changed')
            with self.assertRaisesRegex(controller.ProtocolError, 'Teacher latent hash mismatch: p:0729'):
                controller.preflight(args)

    def test_posix_group_termination_propagates_to_all_live_children(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = controller.Runner(config(Path(directory)))
            live = Mock(pid=123); live.poll.return_value = None
            finished = Mock(pid=456); finished.poll.return_value = 0
            runner.children = {'live': live, 'finished': finished}
            with patch.object(controller.os, 'killpg', create=True) as kill:
                runner.terminate()
            kill.assert_called_once_with(123, controller.signal.SIGTERM)
            self.assertTrue(runner.stop.is_set())


if __name__ == '__main__':
    unittest.main()
