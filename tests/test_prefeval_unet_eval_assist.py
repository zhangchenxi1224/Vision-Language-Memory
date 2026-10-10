"""CPU-only safety tests for external multi-host evaluation scheduling."""
import concurrent.futures
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('eval_assist', Path(__file__).parents[1] / 'scripts/inspire/prefeval_unet_eval_assist.py')
assist = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assist)


class AssistanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(assist_root=self.root / 'assist', run_root=self.root / 'run',
                                    repo=self.root / 'repo', extra_host='extra', gpus=('0', '1', '2', '3'),
                                    stale_seconds=180, poll_seconds=.01)

    def test_atomic_claim_has_exactly_one_owner_and_never_reclaims(self):
        path = self.root / 'claim'
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(lambda i: assist.atomic_task_claim(path, {'owner': i, 'time': 0}), range(32)))
        self.assertEqual(sum(results), 1)
        self.assertFalse(assist.atomic_task_claim(path, {'owner': 'new', 'time': 10**20}))
        self.assertIn(assist.read(path / 'owner.json')['owner'], range(32))

    def test_atomic_registration_serializes_changes(self):
        count = self.root / 'count.json'
        assist.save(count, {'value': 0})
        def increment(_):
            with assist.atomic_registration(self.root / 'mutex'):
                assist.save(count, {'value': assist.read(count)['value'] + 1})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(increment, range(8)))
        self.assertEqual(assist.read(count)['value'], 8)

    def test_reused_pid_is_never_signaled(self):
        receipt = {'pid': 123, 'start_token': 'old', 'command': ['original'], 'host': socket.gethostname()}
        with patch.object(assist, 'process', return_value={'pid': 123, 'start_token': 'new', 'command': ['original']}), \
             patch.object(assist.os, 'kill', create=True) as kill:
            with self.assertRaisesRegex(RuntimeError, 'identity changed'):
                assist.signal_exact(receipt, 19)
            kill.assert_not_called()

    def test_original_signal_targets_only_pid(self):
        receipt = {'pid': 123, 'start_token': 'ok', 'command': ['original'], 'host': socket.gethostname()}
        with patch.object(assist, 'process', return_value=receipt), patch.object(assist.os, 'kill') as kill:
            assist.signal_exact(receipt, 19)
            kill.assert_called_once_with(123, 19)

    def test_zombie_writer_no_longer_occupies_gpu(self):
        receipt = {'pid': 123, 'start_token': 'ok'}
        with patch.object(assist, 'process', return_value=dict(receipt, state='Z')):
            self.assertFalse(assist.live(receipt))

    def test_remote_ack_required_even_when_heartbeat_expired(self):
        assist.save(self.args.assist_root / 'lease.json', {'controller': {'host': 'primary'}})
        for host in ('primary', 'extra'):
            assist.save(self.args.assist_root / 'hosts' / host / 'worker.json', {'status': 'running'})
        assist.save(self.args.assist_root / 'hosts/primary/drained.json', {'helper_gpu_children_terminated': True, 'host': 'primary'})
        self.assertFalse(assist.all_hosts_drained(self.args))
        assist.save(self.args.assist_root / 'hosts/extra/drained.json', {'helper_gpu_children_terminated': False, 'host': 'extra'})
        self.assertFalse(assist.all_hosts_drained(self.args))
        assist.save(self.args.assist_root / 'hosts/extra/drained.json', {'helper_gpu_children_terminated': True, 'host': 'extra'})
        self.assertTrue(assist.all_hosts_drained(self.args))

    def test_missing_or_stale_heartbeat_aborts(self):
        self.assertTrue(assist.should_abort(self.args))
        assist.save(self.args.assist_root / 'heartbeat.json', {'state': 'active', 'time': 0})
        self.assertTrue(assist.should_abort(self.args))
        assist.save(self.args.assist_root / 'heartbeat.json', {'state': 'active', 'time': assist.time.time()})
        self.assertFalse(assist.should_abort(self.args))
        assist.save(self.args.assist_root / 'abort.json', {'reason': 'failure'})
        self.assertTrue(assist.should_abort(self.args))

    def test_final_checkpoint_cannot_run_before_formal_complete(self):
        path = self.root / 'final.pt'
        path.write_bytes(b'not yet ready')
        module = SimpleNamespace(STEPS=23360, checkpoint=Mock(return_value=path), stage_output=Mock(return_value=self.root / 'formal'))
        self.assertIsNone(assist.checkpoint_ready(module, None, {'arm': 'P-L', 'step': 23360}, {}))

    def test_checkpoint_mmap_step_manifest_and_final_hash(self):
        import torch
        output = self.root / 'formal'
        output.mkdir()
        path = output / 'final.pt'
        manifest = {'steps': 23360}
        assist.save(output / 'manifest.json', manifest)
        torch.save({'schema_version': 1, 'optimizer_step': 23360, 'manifest': manifest,
                    'trainable_state': {'x': torch.ones(2)}}, path)
        assist.save(output / 'complete.json', {'steps': 23360, 'checkpoint_sha256': assist.sha(path), 'frozen_verified': True})
        module = SimpleNamespace(STEPS=23360, checkpoint=Mock(return_value=path), stage_output=Mock(return_value=output))
        binding = assist.checkpoint_ready(module, None, {'arm': 'P-L', 'step': 23360}, {})
        self.assertEqual(binding['sha256'], assist.sha(path))
        with self.assertRaisesRegex(RuntimeError, 'step/schema'):
            assist.checkpoint_ready(module, None, {'arm': 'P-L', 'step': 2048}, {})
        assist.save(output / 'complete.json', {'steps': 23360, 'checkpoint_sha256': 'bad', 'frozen_verified': True})
        with self.assertRaisesRegex(RuntimeError, 'completion differs'):
            assist.checkpoint_ready(module, None, {'arm': 'P-L', 'step': 23360}, {})

    def test_completed_receipt_binds_identity_command_artifacts_and_checkpoint(self):
        output = self.root / 'evaluation'
        output.mkdir()
        for name in ('complete.json', 'binding.json', 'summary.json'):
            assist.save(output / name, {'ok': True})
        job = {'label': 'job', 'output': str(output), 'command': ['frozen']}
        receipt = {'identity_sha256': 'identity', 'label': 'job', 'command': ['frozen'], 'checkpoint_sha256': 'checkpoint',
                   'artifacts': {name: assist.sha(output / name) for name in ('complete.json', 'binding.json', 'summary.json')}}
        assist.save(self.args.assist_root / 'completed/job.json', receipt)
        self.assertTrue(assist.completed(self.args, job, 'identity', {'sha256': 'checkpoint'}))
        with self.assertRaises(RuntimeError):
            assist.completed(self.args, job, 'other')
        with self.assertRaises(RuntimeError):
            assist.completed(self.args, job, 'identity', {'sha256': 'changed'})
        assist.save(output / 'summary.json', {'modified': True})
        with self.assertRaises(RuntimeError):
            assist.completed(self.args, job, 'identity')

    def test_drain_never_acknowledges_failed_cleanup(self):
        with patch.object(assist, 'cleanup_receipts', side_effect=RuntimeError('still alive')):
            with self.assertRaisesRegex(RuntimeError, 'still alive'):
                assist.drain_local(self.args)
        self.assertFalse((self.args.assist_root / 'hosts' / socket.gethostname() / 'drained.json').exists())

    def test_resume_requires_same_live_controller(self):
        with patch.object(assist, 'matches', return_value=False), patch.object(assist, 'signal_exact') as send:
            with self.assertRaisesRegex(RuntimeError, 'manual recovery'):
                assist.resume_original(self.args, {'controller': {'pid': 1}})
            send.assert_not_called()
        self.assertFalse((self.args.assist_root / 'resumed.json').exists())

    @unittest.skipUnless(os.name == 'posix', 'Linux process-group recovery integration test')
    def test_guard_kills_orphan_and_waits_for_explicit_remote_ack(self):
        host = socket.gethostname()
        root = self.args.assist_root
        controller = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(120)'], start_new_session=True)
        owner = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(120)'], start_new_session=True)
        children = [controller, owner]
        def receipt(child, **extra):
            value = assist.process(child.pid)
            value.update(host=host, **extra)
            return value
        def wait_until(predicate, seconds=15):
            deadline = time.monotonic() + seconds
            while not predicate():
                self.assertLess(time.monotonic(), deadline, 'Timed out waiting for recovery condition')
                time.sleep(.1)
        try:
            controller_receipt = receipt(controller)
            owner_receipt = receipt(owner, role='coordinate', status='running')
            assist.save(root / 'lease.json', {'controller': controller_receipt, 'coordinator': owner_receipt})
            assist.save(root / 'heartbeat.json', {'state': 'active', 'time': time.time()})
            assist.save(root / 'hosts' / host / 'worker.json', owner_receipt)
            assist.save(root / 'hosts' / host / 'guard-owner.json', owner_receipt)
            assist.save(root / 'hosts/extra/worker.json', {'host': 'extra', 'status': 'running'})
            marker_env = dict(os.environ, PREFEVAL_EVAL_ASSIST_ROOT=str(root.resolve()))
            pidfile = self.root / 'orphan.json'
            code = ('import subprocess,sys,json;from pathlib import Path;'
                    'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]);'
                    f'Path({str(pidfile)!r}).write_text(json.dumps(p.pid))')
            wrapper = subprocess.Popen([sys.executable, '-c', code], env=marker_env, start_new_session=True)
            children.append(wrapper)
            wrapper.wait(timeout=5)
            orphan_pid = json.loads(pidfile.read_text())
            orphan = assist.process(orphan_pid)
            # No process receipt exists: exercise the launch-to-receipt crash gap.
            os.kill(controller.pid, signal.SIGSTOP)
            wait_until(lambda: assist.process(controller.pid)['state'] == 'T')
            guard = subprocess.Popen([sys.executable, str(Path(assist.__file__)), 'guard',
                                      '--repo', str(self.args.repo), '--run-root', str(self.args.run_root),
                                      '--assist-root', str(root), '--extra-host', 'extra', '--poll-seconds', '.1'],
                                      env=marker_env, start_new_session=True)
            children.append(guard)
            wait_until(lambda: (root / 'hosts' / host / 'guard-ready.json').exists())
            owner.kill()
            owner.wait(timeout=5)
            wait_until(lambda: (root / 'hosts' / host / 'drained.json').exists())
            self.assertFalse(assist.live(orphan))
            self.assertEqual(assist.process(controller.pid)['state'], 'T')
            self.assertFalse((root / 'resumed.json').exists())
            assist.save(root / 'hosts/extra/drained.json', {'host': 'extra', 'helper_gpu_children_terminated': True})
            wait_until(lambda: (root / 'recovered.json').exists())
            guard.wait(timeout=5)
            self.assertEqual(guard.returncode, 0)
            self.assertNotEqual(assist.process(controller.pid)['state'], 'T')
        finally:
            for child in children:
                if child.poll() is None:
                    with assist.contextlib.suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)

    def test_plan_only_creates_no_files(self):
        with patch('sys.stdout', new_callable=io.StringIO) as out:
            result = assist.main(['worker', '--repo', str(self.args.repo), '--run-root', str(self.args.run_root),
                                  '--assist-root', str(self.args.assist_root), '--extra-host', 'extra', '--plan-only'])
        self.assertEqual(result, 0)
        self.assertFalse(self.args.assist_root.exists())
        self.assertFalse(json.loads(out.getvalue())['formal_training_changed'])


if __name__ == '__main__':
    unittest.main()
