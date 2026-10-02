"""Check device exclusion, registered coverage and failure cancellation."""
import importlib.util
from pathlib import Path
import threading
import tempfile
import sys
import time
import unittest

spec = importlib.util.spec_from_file_location('scheduler', Path(__file__).parents[1] / 'scripts/inspire/run_prefeval_b730_exposure512_two_gpu.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


class ScheduleTests(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', 'Adoption identity uses Linux /proc')
    def test_adopts_live_process_without_claiming_unobserved_exit_code(self):
        import json
        import socket
        import subprocess
        from unittest.mock import patch
        sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/inspire'))
        import run_prefeval_b730_exposure512_distributed as distributed
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frozen=root/'code'
            frozen.mkdir()
            (root/'processes').mkdir()
            (root/'train').mkdir()
            command=[sys.executable,'-c','import time; time.sleep(0.3)',str(frozen/'scripts/experiments/prefeval_k1_write_extension.py'),
                     '--steps','93440','--snapshot-steps','46720','70080']
            process=subprocess.Popen(command,cwd=frozen)
            receipt=dict(pid=process.pid,host=socket.gethostname(),status='running',gpu=0,command=command)
            (root/'processes/train.json').write_text(json.dumps(receipt))
            class Stop:
                def wait(self,seconds):
                    time.sleep(0.05)
                    return False
            class Core:
                STOP=Stop()
                @staticmethod
                def save(path,value): path.write_text(json.dumps(value))
                def train(self): raise AssertionError('Must not spawn a second training process')
            try:
                with patch.object(distributed,'RUN',root),patch.object(distributed,'FROZEN',frozen):
                    core=Core()
                    distributed.adopt_existing_train(core,process.pid)
                    (root/'train/complete.json').write_text('{"steps":93440}')
                    (root/'train/checkpoint-final.pt').touch()
                    core.train()
                after=json.loads((root/'processes/train.json').read_text())
                self.assertEqual(after['pid'],process.pid)
                self.assertEqual(after['status'],'complete')
                self.assertIsNone(after['exit_code'])
            finally:
                process.wait(timeout=5)

    @unittest.skipIf(sys.platform == 'win32', 'Production orchestration requires POSIX')
    def test_primary_waits_for_reader_without_executing_its_work(self):
        sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/inspire'))
        import run_prefeval_b730_exposure512_distributed as distributed
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            class Stop:
                def wait(self, seconds):
                    for step in distributed.ENDPOINTS:
                        for split in ('pilot','dev'):
                            path=root/f'readback/step-{step:06d}-{split}-V1/summary.json'
                            path.parent.mkdir(parents=True,exist_ok=True)
                            path.write_text('{}')
                    (root/'reader-complete.json').write_text('{}')
                    return False
            class Core:
                STOP=Stop()
                def evaluate(self,*args):
                    raise AssertionError('Primary tried to run reader work')
            with patch.object(distributed,'RUN',root):
                distributed.wait_for_reader(Core())

    @unittest.skipIf(sys.platform == 'win32', 'Production orchestration requires POSIX')
    def test_four_plus_two_covers_same_registration_with_one_training_lane(self):
        sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts/inspire'))
        import run_prefeval_b730_exposure512_distributed as distributed
        calls, training = [], []
        class Core:
            def train(self):
                training.append('train')
            def evaluate(self, *args):
                calls.append(args)
        for role, count in [('primary', 4), ('reader', 2)]:
            for gpu in range(count):
                distributed.lane(Core(), role, gpu)
        self.assertEqual(training, ['train'])
        self.assertEqual(len(calls), 19)
        self.assertEqual({c[:3] for c in calls}, set(scheduler.registered_evaluations()))

    @unittest.skipIf(sys.platform == 'win32', 'Shared POSIX lock is exercised on Linux')
    def test_shared_endpoint_lock_prevents_duplicate_work(self):
        with tempfile.TemporaryDirectory() as directory:
            writes = []
            class Core:
                RUN = Path(directory)
                STOP = threading.Event()
                def evaluate(self, *args):
                    marker = self.RUN / 'summary.json'
                    if marker.exists():
                        return
                    time.sleep(0.1)
                    writes.append(args)
                    marker.write_text('{}')
            first, second = Core(), Core()
            scheduler.lock_evaluations(first)
            scheduler.lock_evaluations(second)
            with scheduler.concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(c.evaluate, 93440, 'pilot', 0, gpu) for gpu, c in enumerate((first, second))]
                for job in jobs:
                    job.result()
            self.assertEqual(len(writes), 1)

    def test_each_registered_evaluation_once_and_gpu_zero_reused_after_training(self):
        trained = threading.Event()
        calls = []
        class Core:
            def train(self):
                trained.set()
            def evaluate(self, step, split, variant, gpu):
                if gpu == 0:
                    assert trained.is_set()
                calls.append((step, split, variant, gpu))
            def terminate_children(self):
                raise AssertionError('Unexpected cancellation')
        scheduler.execute(Core())
        keys = [c[:3] for c in calls]
        self.assertEqual(len(keys), 19)
        self.assertEqual(set(keys), set(scheduler.registered_evaluations()))
        self.assertEqual({c[3] for c in calls}, {0, 1})
        for step, split, variant, gpu in calls:
            if step != 23360:
                self.assertEqual(variant, gpu)

    def test_failed_training_cancels_waiting_evaluation_lane(self):
        stopped = threading.Event()
        class Core:
            def train(self):
                raise RuntimeError('simulated training failure')
            def evaluate(self, *args):
                assert stopped.wait(5), 'Waiting lane was not cancelled'
                raise RuntimeError('cancelled')
            def terminate_children(self):
                stopped.set()
        with self.assertRaises(RuntimeError):
            scheduler.execute(Core())
        self.assertTrue(stopped.is_set())


if __name__ == '__main__':
    unittest.main()
